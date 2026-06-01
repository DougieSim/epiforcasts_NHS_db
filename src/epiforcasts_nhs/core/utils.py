"""
Utility functions for the Bayesian pressure model.

Covers data preparation, encoding, season mapping, and posterior extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az

import epiforcasts_nhs.core.constants as constants


# ─────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove England aggregate row. Use all ICB data — no holdout."""
    return df[df["icb"] != "England"].copy()


def encode(df: pd.DataFrame) -> dict:
    """
    Encode the weekly panel into model-ready arrays.

    Season is derived from the calendar month column so Winter always maps to
    Dec/Jan/Feb regardless of which week the dataset starts on.

    Returns dict with keys:
        icb_codes, week_idx, season_idx, beds, n_icb, n_weeks, categories
    """
    from datetime import date as _date, timedelta as _td
    _START_DATE = _date(2023, 1, 2)

    cat        = df["icb"].astype("category")
    categories = cat.cat.categories
    icb_codes  = cat.cat.codes.values
    week_idx   = (df["week"].values - df["week"].min()).astype(int)
    beds       = df["bed_occupancy"].values
    n_icb      = len(categories)
    n_weeks    = int(week_idx.max()) + 1

    month_vals    = df["month"].values if "month" in df.columns else np.full(len(df), np.nan)
    week_vals     = df["week"].values
    week_date_col = df["week_date"] if "week_date" in df.columns else None

    month_resolved: list[int] = []
    for i, m in enumerate(month_vals):
        if not (m != m):  # fast NaN check
            month_resolved.append(int(m))
        elif week_date_col is not None and pd.notna(week_date_col.iloc[i]):
            try:
                month_resolved.append(pd.to_datetime(week_date_col.iloc[i]).month)
            except Exception:
                month_resolved.append((_START_DATE + _td(weeks=int(week_vals[i]))).month)
        else:
            month_resolved.append((_START_DATE + _td(weeks=int(week_vals[i]))).month)

    season_idx = np.array([constants.SEASONS.month_to_season[m] for m in month_resolved])

    return dict(
        icb_codes=icb_codes,
        week_idx=week_idx,
        season_idx=season_idx,
        beds=beds,
        n_icb=n_icb,
        n_weeks=n_weeks,
        categories=categories,
    )


def check_season_alignment(enc: dict, df: pd.DataFrame) -> None:
    """Print season and date for the first N weeks as a sanity check."""
    min_week   = df["week"].min()
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    has_date   = "week_date" in df.columns
    n_display  = constants.SUMMARY.season_alignment_display_weeks

    print("Season alignment check (first %d weeks):" % n_display)
    for w in range(min(n_display, enc["n_weeks"])):
        mask = np.where(week_idx == w)[0]
        if len(mask) == 0:
            continue
        s        = int(season_idx[mask[0]])
        date_str = f"  ({df['week_date'].values[mask[0]]})" if has_date else ""
        print(f"  Week {min_week + w}{date_str}: {constants.SEASONS.names[s]}")


def current_season_from_enc(enc: dict) -> int:
    """Return the season index for the final week recorded in an encoding dict."""
    last_week_pos = np.where(enc["week_idx"] == enc["week_idx"].max())[0][0]
    return int(enc["season_idx"][last_week_pos])


# ─────────────────────────────────────────
# Posterior extraction
# ─────────────────────────────────────────

def current_pressure_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    """
    Posterior samples for the current underlying level of one ICB (ex-seasonal).
    Shape: (n_chains * n_draws,).
    """
    level = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
    return level[:, :, -1, icb_idx].astype(float).ravel()


def current_total_pressure_samples(
    idata: az.InferenceData,
    icb_idx: int,
    current_season: int,
) -> np.ndarray:
    """
    Posterior samples for total pressure (level + seasonal effect) for one ICB.

    Parameters
    ----------
    current_season : 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    """
    level          = current_pressure_samples(idata, icb_idx)
    season_effects = idata.posterior["season_effects"].values.reshape(-1, constants.SEASONS.n_seasons)
    return level + season_effects[:, current_season]


def direction_of_travel(
    idata: az.InferenceData,
    icb_idx: int,
    lookback_weeks: int = constants.SUMMARY.lookback_weeks,
) -> np.ndarray:
    """
    Posterior samples for change in underlying level over the last N weeks.
    Positive = rising pressure. Seasonal component excluded.
    """
    level    = idata.posterior["level"].values
    current  = level[:, :, -1, icb_idx].ravel()
    previous = level[:, :, -lookback_weeks, icb_idx].ravel()
    return current - previous


def pressure_summary(
    idata: az.InferenceData,
    enc: dict,
    lookback_weeks: int = constants.SUMMARY.lookback_weeks,
    current_season: int = 0,
) -> pd.DataFrame:
    """
    Clinical summary table — one row per ICB, sorted by descending pressure.

    Columns: icb, pressure_median, pressure_lo, pressure_hi, bed_occ_median,
             p_above_concern, p_above_high, dot_median, p_rising
    """
    ls = constants.LATENT_SCALE
    th = constants.PRESSURE_THRESHOLDS
    ci = constants.SUMMARY

    rows = []
    for i, icb_name in enumerate(enc["categories"]):
        level_samples = current_pressure_samples(idata, i)
        total_samples = current_total_pressure_samples(idata, i, current_season)
        dot           = direction_of_travel(idata, i, lookback_weeks)

        rows.append(dict(
            icb=icb_name,
            pressure_median=float(np.median(level_samples)),
            pressure_lo=float(np.percentile(level_samples, ci.ci_lower_pct)),
            pressure_hi=float(np.percentile(level_samples, ci.ci_upper_pct)),
            bed_occ_median=float(ls.baseline + np.median(total_samples) * ls.scale),
            p_above_concern=float(np.mean(total_samples > th.concern)),
            p_above_high=float(np.mean(total_samples > th.elevated)),
            dot_median=float(np.median(dot)),
            p_rising=float(np.mean(dot > 0)),
        ))

    return pd.DataFrame(rows).sort_values("pressure_median", ascending=False)
