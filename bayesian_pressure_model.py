"""
Bayesian NHS system pressure model — AR(1) with seasonal decomposition.

Model structure:
    level[icb, 0]   ~ Normal(mu_national, sigma_icb)
    level[icb, t]   = rho * level[icb, t-1] + N(0, sigma_drift)
    season_effects  = [winter, spring, summer, autumn]  sum-to-zero
    bed_obs[i]      ~ Normal(85 + (level[icb, week] + season[week]) * 6, sigma_obs)

The AR(1) level captures each ICB's underlying pressure trajectory.
The seasonal component captures shared winter/spring/summer/autumn patterns.
rho controls persistence — Beta(2,2) prior lets the data determine how
mean-reverting vs random-walk the dynamics are.

Clinical outputs
----------------
level[icb, T]                     — current underlying pressure (ex-seasonal)
level[icb, T] - level[icb, T-k]  — direction of travel
season_effects                    — shared seasonal pattern across all ICBs
P(level[icb,T] + season[T] > threshold)  — probability of elevated pressure
"""

from __future__ import annotations

import os
import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

from plots import (
    plot_prior_predictive,
    plot_posterior_predictive,
    plot_residuals,
    plot_seasonal_effects,
    plot_pressure_trajectories,
    plot_direction_of_travel,
    plot_clinical_summary,
)

_FAST = os.environ.get("PRESSURE_MODEL_FAST", "1").strip().lower() not in (
    "0", "false", "no",
)

rng = np.random.default_rng(42)

SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]


# ─────────────────────────────────────────
# Data
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove England aggregate. Use all data — no holdout for monitoring."""
    return df[df["icb"] != "England"].copy()


# Month → season mapping: Dec/Jan/Feb=Winter, Mar/Apr/May=Spring,
# Jun/Jul/Aug=Summer, Sep/Oct/Nov=Autumn
MONTH_TO_SEASON: dict[int, int] = {
    12: 0, 1: 0, 2: 0,   # Winter
     3: 1, 4: 1, 5: 1,   # Spring
     6: 2, 7: 2, 8: 2,   # Summer
     9: 3, 10: 3, 11: 3, # Autumn
}


def encode(df: pd.DataFrame) -> dict:
    """
    Encode all data into model-ready arrays.

    Season is derived from the calendar month in the `month` column
    (added by the data generator). This ensures winter always maps to
    Dec/Jan/Feb regardless of which week the dataset starts on.

    Returns a dict containing:
      icb_codes  — integer ICB index per observation  (n_obs,)
      week_idx   — 0-based week index per observation  (n_obs,)
      season_idx — season 0..3 per observation  (n_obs,)
                   0=Winter, 1=Spring, 2=Summer, 3=Autumn
      beds       — observed bed occupancy  (n_obs,)
      n_icb      — number of ICBs
      n_weeks    — number of weeks
      categories — ordered ICB names
    """
    cat        = df["icb"].astype("category")
    categories = cat.cat.categories
    icb_codes  = cat.cat.codes.values
    week_idx   = (df["week"].values - df["week"].min()).astype(int)
    beds       = df["bed_occupancy"].values
    n_icb      = len(categories)
    n_weeks    = int(week_idx.max()) + 1

    from datetime import date as _date, timedelta as _td
    _START_DATE = _date(2023, 1, 2)

    if "month" in df.columns:
        month_vals = df["month"].values
    else:
        month_vals = np.full(len(df), np.nan)

    # Resolve NaN months: try week_date first, then derive from week number.
    month_resolved = []
    week_vals = df["week"].values
    week_date_col = df["week_date"] if "week_date" in df.columns else None
    for i, m in enumerate(month_vals):
        if not (m != m):  # fast NaN check without pandas
            month_resolved.append(int(m))
        elif week_date_col is not None and pd.notna(week_date_col.iloc[i]):
            try:
                month_resolved.append(
                    pd.to_datetime(week_date_col.iloc[i]).month
                )
            except Exception:
                month_resolved.append(
                    (_START_DATE + _td(weeks=int(week_vals[i]))).month
                )
        else:
            month_resolved.append(
                (_START_DATE + _td(weeks=int(week_vals[i]))).month
            )

    season_idx = np.array([MONTH_TO_SEASON[m] for m in month_resolved])

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
    """
    Print the season and date for the first 8 weeks as a sanity check.
    Season is derived from calendar month so no offset adjustment needed.
    """
    min_week   = df["week"].min()
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    has_date   = "week_date" in df.columns

    print("Season alignment check (first 8 weeks):")
    for w in range(min(8, enc["n_weeks"])):
        mask = np.where(week_idx == w)[0]
        if len(mask) == 0:
            continue
        s        = int(season_idx[mask[0]])
        date_str = f"  ({df['week_date'].values[mask[0]]})" if has_date else ""
        print(f"  Week {min_week + w}{date_str}: {SEASON_NAMES[s]}")


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

def build_model(enc: dict) -> pm.Model:
    """
    AR(1) local level model with shared four-season component.

    Variables
    ---------
    mu_national    ~ Normal(0, 1)
        National average pressure on latent scale. At 0 → 85% occupancy.

    sigma_icb      ~ Exponential(1)
        Cross-ICB spread at initialisation.

    rho            ~ Beta(2, 2)
        AR(1) persistence. rho=1 is pure random walk; rho=0 is white noise.
        Beta(2,2) is symmetric around 0.5; data determines the value.

    level_init     ~ Normal(mu_national, sigma_icb)  shape=(n_icb,)
        Starting pressure level for each ICB at week 0.

    sigma_drift    ~ Exponential(lam=10)  [mean 0.1]
        Week-to-week innovation noise on the AR(1) level.
        0.1 latent ≈ 0.6% occupancy per week.

    innovations    ~ Normal(0, sigma_drift)  shape=(n_weeks-1, n_icb)
        Random shocks driving the AR(1) level at each step.

    level          Deterministic  shape=(n_weeks, n_icb)
        Full AR(1) trajectory: level[t] = rho * level[t-1] + innovations[t].

    sigma_season   ~ Exponential(lam=5)  [mean 0.2]
        Scale of seasonal effects. 0.2 latent ≈ 1.2% occupancy swing.

    season_raw     ~ Normal(0, sigma_season)  shape=(3,)
        Free seasonal effects for winter, spring, summer.

    season_effects  Deterministic  shape=(4,)
        [winter, spring, summer, autumn] with sum-to-zero constraint.
        autumn = -(winter + spring + summer).

    sigma_obs      ~ Exponential(lam=5)  [mean 0.2]
        Observation noise — measurement error and reporting noise.

    bed_obs        ~ Normal(85 + (level[week, icb] + season[week]) * 6, sigma_obs)
        Likelihood. Fixed to observed data; drives the posterior.
    """
    icb_codes  = enc["icb_codes"]
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    beds       = enc["beds"]
    n_icb      = enc["n_icb"]
    n_weeks    = enc["n_weeks"]

    with pm.Model() as model:

        # ── National anchor ───────────────────────────────────────────
        mu_national = pm.Normal("mu_national", 0, 1)
        sigma_icb   = pm.Exponential("sigma_icb", 1)

        # ── AR(1) persistence ─────────────────────────────────────────
        rho = pm.Beta("rho", alpha=2, beta=2)

        # ── Initial pressure level per ICB ────────────────────────────
        level_init = pm.Normal(
            "level_init",
            mu=mu_national,
            sigma=sigma_icb,
            shape=n_icb,
        )

        # ── Week-to-week AR(1) innovations ────────────────────────────
        sigma_drift = pm.Exponential("sigma_drift", lam=1)

        innovations = pm.Normal(
            "innovations",
            0,
            sigma_drift,
            shape=(n_weeks - 1, n_icb),
        )

        # AR(1) scan: level[t] = rho * level[t-1] + innovations[t]
        levels_rest = pytensor.scan(
            fn=lambda innov, prev, rho: rho * prev + innov,
            sequences=innovations,
            outputs_info=level_init,
            non_sequences=rho,
            return_updates=False,
        )

        level = pm.Deterministic(
            "level",
            pt.concatenate([level_init[None, :], levels_rest], axis=0),
        )   # shape: (n_weeks, n_icb)

        # ── Shared seasonal component ─────────────────────────────────
        # Four free parameters, explicitly centred via sum-to-zero
        # reparameterisation. All seasons estimated freely — no season
        # is forced to be the residual of the other three.
        # The sum-to-zero constraint is applied as a shift so the
        # seasonal component has zero net annual effect on the level.
        sigma_season = pm.Exponential("sigma_season", lam=5)

        season_raw = pm.Normal(
            "season_raw",
            0,
            sigma_season,
            shape=4,    # winter, spring, summer, autumn — all free
        )

        # Enforce sum-to-zero by subtracting the mean from each effect.
        # This is equivalent to the constraint but gives each season
        # equal freedom in the posterior.
        season_effects = pm.Deterministic(
            "season_effects",
            season_raw - pt.mean(season_raw),
        )   # shape: (4,)  [winter, spring, summer, autumn]

        # ── Latent pressure ───────────────────────────────────────────
        latent = (
            level[week_idx, icb_codes]
            + season_effects[season_idx]
        )

        # ── Likelihood ────────────────────────────────────────────────
        sigma_obs = pm.Exponential("sigma_obs", lam=5)

        pm.Normal(
            "bed_obs",
            mu=85 + latent * 6,
            sigma=sigma_obs,
            observed=beds,
        )

    return model


# ─────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────

def sample_prior(model: pm.Model, draws: int = 100) -> az.InferenceData:
    with model:
        idata = pm.sample_prior_predictive(draws=draws, random_seed=rng)
    return idata


def sample_posterior(model: pm.Model, draws: int, tune: int) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.95,
            progressbar=True,
            compute_convergence_checks=False,
        )
    return idata


def sample_posterior_predictive(
    model: pm.Model,
    idata: az.InferenceData,
) -> az.InferenceData:
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


# ─────────────────────────────────────────
# Clinical summaries
# ─────────────────────────────────────────

def current_pressure_samples(
    idata: az.InferenceData,
    icb_idx: int,
) -> np.ndarray:
    """
    Posterior samples for the CURRENT pressure level of one ICB.
    Returns level[icb, T] — the final week — flattened across chains/draws.
    This is the season-adjusted underlying level, not including seasonal effect.
    """
    level = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
    return level[:, :, -1, icb_idx].ravel()


def current_total_pressure_samples(
    idata: az.InferenceData,
    icb_idx: int,
    current_season: int,
) -> np.ndarray:
    """
    Posterior samples for total pressure including the current season's effect.
    Use this for dashboard risk probabilities — a winter reading should
    reflect that winter adds to underlying pressure.

    Parameters
    ----------
    current_season : int
        Current season index: 0=winter, 1=spring, 2=summer, 3=autumn.
    """
    level          = idata.posterior["level"].values[:, :, -1, icb_idx].ravel()
    season_effects = idata.posterior["season_effects"].values.reshape(-1, 4)
    return level + season_effects[:, current_season]


def direction_of_travel(
    idata: az.InferenceData,
    icb_idx: int,
    lookback_weeks: int = 4,
) -> np.ndarray:
    """
    Posterior samples for change in underlying level over the last N weeks.
    Positive = rising pressure, negative = falling.
    Uses the level (seasonal component removed) so direction of travel
    reflects genuine trend, not seasonal fluctuation.
    """
    level    = idata.posterior["level"].values
    current  = level[:, :, -1, icb_idx].ravel()
    previous = level[:, :, -lookback_weeks, icb_idx].ravel()
    return current - previous


def pressure_summary(
    idata: az.InferenceData,
    enc: dict,
    lookback_weeks: int = 4,
    current_season: int = 0,
) -> pd.DataFrame:
    """
    Clinical summary table: one row per ICB.

    Columns
    -------
    icb              — ICB name
    pressure_median  — median current level (latent scale, ex-seasonal)
    pressure_lo      — 10th percentile of level
    pressure_hi      — 90th percentile of level
    bed_occ_median   — implied bed occupancy including current season (%)
    p_above_concern  — P(total pressure > 0.5)
    p_above_high     — P(total pressure > 1.1)
    dot_median       — median direction of travel over lookback_weeks
    p_rising         — P(underlying level is rising)
    """
    rows = []
    for i, icb_name in enumerate(enc["categories"]):
        level_samples = current_pressure_samples(idata, i)
        total_samples = current_total_pressure_samples(idata, i, current_season)
        dot           = direction_of_travel(idata, i, lookback_weeks)

        rows.append(dict(
            icb=icb_name,
            pressure_median=float(np.median(level_samples)),
            pressure_lo=float(np.percentile(level_samples, 10)),
            pressure_hi=float(np.percentile(level_samples, 90)),
            bed_occ_median=float(85 + np.median(total_samples) * 6),
            p_above_concern=float(np.mean(total_samples > 0.5)),
            p_above_high=float(np.mean(total_samples > 1.1)),
            dot_median=float(np.median(dot)),
            p_rising=float(np.mean(dot > 0)),
        ))

    return pd.DataFrame(rows).sort_values("pressure_median", ascending=False)


# ─────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────

def save_posteriors(
    idata: az.InferenceData,
    enc: dict,
    path: str = "posteriors.nc",
) -> None:
    """
    Save posteriors to NetCDF with a Windows-safe atomic swap.

    Writes to a temp file first, then replaces the target.
    On Windows, if the file is locked by the dashboard, stages the
    new file as .posteriors_new.nc for the app to pick up on next load.
    """
    import sys
    import tempfile
    from pathlib import Path

    final_path = Path(path)
    idata.attrs["icbs"] = list(enc["categories"])
    # Store the calendar month of the final training week so the dashboard
    # can determine the current season without a winter_start_week offset.
    last_week_mask = np.where(enc["week_idx"] == enc["n_weeks"] - 1)[0]
    if len(last_week_mask) > 0:
        # season_idx values: 0=Winter,1=Spring,2=Summer,3=Autumn
        # Store the season directly — reconstructing the month isn't needed
        idata.attrs["last_season"] = int(enc["season_idx"][last_week_mask[0]])

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=".posteriors_tmp_",
        suffix=".nc",
    )
    try:
        os.close(tmp_fd)
        idata.to_netcdf(tmp_path, engine="h5netcdf")

        if sys.platform == "win32":
            backup_path = final_path.with_suffix(".nc.bak")
            try:
                if final_path.exists():
                    os.replace(str(final_path), str(backup_path))
                os.replace(tmp_path, str(final_path))
                if backup_path.exists():
                    os.unlink(str(backup_path))
            except PermissionError:
                staged = final_path.with_name(".posteriors_new.nc")
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, str(staged))
                raise RuntimeError(
                    f"posteriors.nc is locked. Staged at {staged} — retrying next cycle."
                )
        else:
            os.replace(tmp_path, str(final_path))

        print(f"Posteriors saved → {final_path}  ({len(enc['categories'])} ICBs)")

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def fit_pressure_model(
    df: pd.DataFrame,
    *,
    fast: bool | None = None,
) -> tuple:
    """
    Full inference pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weekly NHS data including England row.
        Must contain a 'month' column (added by generate_synthetic_data.py).
    fast : bool | None
        If True, use fewer draws for quick iteration.
        Defaults to the PRESSURE_MODEL_FAST environment variable.
    """
    if fast is None:
        fast = _FAST
    draws, tune = (400, 400) if fast else (2000, 2000)

    data = prepare_data(df)
    enc  = encode(data)

    print(f"Rows:  {len(data)}")
    print(f"Weeks: {data['week'].min()}–{data['week'].max()}")
    print(f"ICBs:  {list(enc['categories'])}")

    check_season_alignment(enc, data)

    model = build_model(enc)

    # 1. Prior predictive check
    idata = sample_prior(model)
    #plot_prior_predictive(idata)

    # 2. Fit posterior
    idata = sample_posterior(model, draws, tune)

    # 3. Posterior predictive check
    idata = sample_posterior_predictive(model, idata)
    #plot_posterior_predictive(idata, enc["beds"])
    #plot_residuals(idata, enc["beds"])

    # 4. Convergence
    print("\nConvergence summary:")
    print(az.summary(idata, var_names=[
        "mu_national", "sigma_icb", "rho",
        "sigma_drift", "sigma_season", "sigma_obs",
    ]))

    # 5. Clinical outputs
    #plot_seasonal_effects(idata)
    #plot_pressure_trajectories(idata, enc, data)
    #plot_direction_of_travel(idata, enc)

    # Current season index — find the last week's season reliably
    # (don't use [-1] since rows may not be sorted by week)
    last_week_pos  = np.where(enc["week_idx"] == enc["week_idx"].max())[0][0]
    current_season = int(enc["season_idx"][last_week_pos])
    print(f"\nCurrent season: {SEASON_NAMES[current_season]}")

    summary = pressure_summary(idata, enc, current_season=current_season)
    print("\nClinical summary (ranked by current pressure):")
    print(summary.to_string(index=False, float_format="{:.3f}".format))
    #plot_clinical_summary(summary)

    # 6. Save for dashboard
    save_posteriors(idata, enc)

    return model, idata, enc


def main():
    df = pd.read_csv("synthetic_nhs_pressure.csv")
    fit_pressure_model(df)


if __name__ == "__main__":
    main()