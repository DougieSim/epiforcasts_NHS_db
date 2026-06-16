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


# ─────────────────────────────────────────
# Forward projection — weeks to occupancy thresholds
# ─────────────────────────────────────────

def threshold_occupancies() -> dict[str, float]:
    """
    The three occupancy (%) targets used for weeks-to-threshold projection.

    "concern" and "elevated" are the model's latent PRESSURE_THRESHOLDS mapped to
    occupancy via the latent scale; "capacity" is the 100% reference line.
    """
    ls = constants.LATENT_SCALE
    th = constants.PRESSURE_THRESHOLDS
    return {
        "concern":  ls.baseline + th.concern * ls.scale,
        "elevated": ls.baseline + th.elevated * ls.scale,
        "capacity": 100.0,
    }


def future_season_indices(
    horizon: int,
    *,
    last_date: "pd.Timestamp | None" = None,
    current_season: int = 0,
) -> np.ndarray:
    """
    Season index (0–3) for each of the next `horizon` weeks.

    If a calendar `last_date` is given, advance the real calendar one week at a
    time and map month → season. Otherwise fall back to equal-length season
    blocks starting from `current_season`.
    """
    n = constants.SEASONS.n_seasons
    if last_date is not None and pd.notna(last_date):
        last_date = pd.Timestamp(last_date)
        months = [(last_date + pd.Timedelta(weeks=k)).month for k in range(1, horizon + 1)]
        return np.array([constants.SEASONS.month_to_season[m] for m in months], dtype=int)

    wps = constants.FORECAST.weeks_per_season
    return np.array([(current_season + (k // wps)) % n for k in range(1, horizon + 1)], dtype=int)


def forward_simulate_occupancy(
    idata: az.InferenceData,
    icb_idx: int,
    future_seasons: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Forward-simulate expected bed occupancy (%) for one ICB.

    Propagates the model's own AR(1) dynamics from the posterior draws:
        level_{k} = rho * level_{k-1} + Normal(0, sigma_drift)
    (no drift term — the process mean-reverts to 0, so crossings arise from the
    sigma_drift innovations rather than a fictitious linear trend), then maps
    level + seasonal effect to occupancy via the latent scale.

    Returns an array of shape (n_samples, horizon) of expected occupancy %.
    Uses the expected occupancy (no sigma_obs observation noise), consistent
    with how thresholds are applied in `pressure_summary`.
    """
    ls = constants.LATENT_SCALE
    post = idata.posterior

    rho         = post["rho"].values.reshape(-1)
    sigma_drift = post["sigma_drift"].values.reshape(-1)
    level       = post["level"].values
    level_last  = level[:, :, -1, icb_idx].reshape(-1).astype(float)
    season_f    = post["season_effects"].values.reshape(-1, constants.SEASONS.n_seasons)

    n_samples = level_last.shape[0]
    horizon   = len(future_seasons)
    occ       = np.empty((n_samples, horizon), dtype=float)

    lvl = level_last
    for k in range(horizon):
        lvl = rho * lvl + rng.normal(0.0, sigma_drift)
        season_k   = season_f[:, future_seasons[k]]
        occ[:, k]  = ls.baseline + (lvl + season_k) * ls.scale

    return occ


def weeks_to_thresholds(
    occ_samples: np.ndarray,
    current_occ_median: float,
    targets: dict[str, float],
) -> dict[str, dict]:
    """
    First-passage weeks to each occupancy target from forward-simulated paths.

    Parameters
    ----------
    occ_samples        : (n_samples, horizon) forward-simulated occupancy %.
    current_occ_median : current median occupancy % (for the "already above" case).
    targets            : mapping of target name → occupancy % cut-point.

    Returns, per target name, a dict with:
        weeks    : int median first-passage week, 0 if already above, or None if
                   fewer than half the samples cross within the horizon.
        p_within : posterior probability of crossing within the horizon.
        already  : True if current median occupancy is already at/above the target.
    """
    horizon = occ_samples.shape[1]
    out: dict[str, dict] = {}
    for name, target in targets.items():
        if current_occ_median >= target:
            out[name] = {"weeks": 0, "p_within": 1.0, "already": True}
            continue

        crossed   = occ_samples >= target           # (S, horizon)
        any_cross = crossed.any(axis=1)
        first_idx = crossed.argmax(axis=1) + 1       # 1-based week; meaningless where no cross
        weeks_arr = np.where(any_cross, first_idx, np.inf)

        p_within = float(any_cross.mean())
        median_w = float(np.median(weeks_arr))
        weeks    = int(median_w) if np.isfinite(median_w) else None

        out[name] = {"weeks": weeks, "p_within": p_within, "already": False}
    return out


def project_weeks_to_thresholds(
    idata: az.InferenceData,
    icb_idx: int,
    current_season: int,
    *,
    last_date: "pd.Timestamp | None" = None,
    horizon: int | None = None,
) -> dict[str, dict]:
    """
    Convenience wrapper: forward-simulate one ICB and return weeks-to-threshold
    for the standard concern / elevated / capacity occupancy targets.
    """
    horizon = horizon or constants.FORECAST.horizon_weeks
    rng     = np.random.default_rng(constants.FORECAST.seed)

    future_seasons = future_season_indices(
        horizon, last_date=last_date, current_season=current_season
    )
    occ = forward_simulate_occupancy(idata, icb_idx, future_seasons, rng)

    total_now = current_total_pressure_samples(idata, icb_idx, current_season)
    current_occ_median = float(constants.LATENT_SCALE.baseline
                               + np.median(total_now) * constants.LATENT_SCALE.scale)

    return weeks_to_thresholds(occ, current_occ_median, threshold_occupancies())
