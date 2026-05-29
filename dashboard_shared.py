"""
Shared dashboard constants and posterior helper utilities.

Centralises model-side logic used across dashboard variants.
Updated to read from the AR(1) + seasonal model — level[icb, T] is the
current underlying pressure; total pressure adds the current season's effect.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import arviz as az

# Week 0 anchor — must match generate_synthetic_data.py and synthetic_data_generator.py
_START_DATE = date(2023, 1, 2)

# Month → season index (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
_MONTH_TO_SEASON: dict[int, int] = {
    12: 0, 1: 0, 2: 0,
     3: 1, 4: 1, 5: 1,
     6: 2, 7: 2, 8: 2,
     9: 3, 10: 3, 11: 3,
}


# ─────────────────────────────────────────
# Reference thresholds
# ─────────────────────────────────────────

# Reference levels on the latent index (demo-only communication cut-points).
# These are on the TOTAL pressure scale (level + seasonal effect).
THRESHOLD_BASELINE = 0.0
THRESHOLD_CONCERN  = 0.5
THRESHOLD_ELEVATED = 1.1

# Default UI heuristic gates for qualitative risk summaries.
DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH    = 0.25
DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN = 0.55
DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH      = 0.08
DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN   = 0.30

SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]


# ─────────────────────────────────────────
# Posterior helpers
# ─────────────────────────────────────────

def resolve_icb_index(idata: az.InferenceData, icb_name: str) -> int:
    """
    Resolve an ICB name to its index in the posterior.

    Raises ValueError if ICB metadata is missing or ICB not found.
    """
    icbs = list(idata.attrs.get("icbs", []))
    if not icbs:
        raise ValueError("Posterior metadata does not include an 'icbs' list.")
    try:
        return icbs.index(icb_name)
    except ValueError as exc:
        raise ValueError(
            f"ICB '{icb_name}' was not found in posterior metadata."
        ) from exc


def current_season_index(idata: az.InferenceData) -> int:
    """
    Return the season index (0-3) for the most recent week in the posterior.

    Reads last_month from idata.attrs (the calendar month of the final
    training week, stored during save_posteriors). Falls back to deriving
    from the level shape for backwards compatibility with older posteriors.
    """
    last_season = idata.attrs.get("last_season")
    if last_season is not None:
        return int(last_season)

    # Fallback: derive from n_weeks (not reliable — regenerate posteriors)
    level   = idata.posterior["level"].values
    n_weeks = level.shape[2]
    return int(n_weeks % 4)


def pressure_index_samples(
    idata: az.InferenceData,
    icb_idx: int,
) -> np.ndarray:
    """
    Posterior samples for TOTAL current pressure for one ICB.

    Total pressure = underlying AR(1) level + current season's effect.
    This is what the dashboard displays — it reflects both the structural
    ICB pressure and the seasonal contribution at the current time of year.

    Returns flattened array of shape (n_chains * n_draws,).
    """
    level          = idata.posterior["level"].values
    season_effects = idata.posterior["season_effects"].values

    # Flatten chains × draws
    n_chains, n_draws = level.shape[:2]
    level_flat   = level.reshape(-1, level.shape[2], level.shape[3])
    season_flat  = season_effects.reshape(-1, 4)

    current_level  = level_flat[:, -1, icb_idx]          # (S,)
    season_idx     = current_season_index(idata)
    current_season = season_flat[:, season_idx]           # (S,)

    return (current_level + current_season).astype(float)


def level_only_samples(
    idata: az.InferenceData,
    icb_idx: int,
) -> np.ndarray:
    """
    Posterior samples for the underlying AR(1) level — seasonal component removed.
    Useful for direction-of-travel calculations where you want to distinguish
    genuine trend from seasonal fluctuation.
    """
    level = idata.posterior["level"].values
    return level[:, :, -1, icb_idx].ravel().astype(float)


def seasonal_effect_samples(idata: az.InferenceData) -> np.ndarray:
    """
    Posterior samples for all four seasonal effects.
    Returns shape (n_samples, 4) on the latent scale.
    Multiply by 6 to convert to % bed occupancy.
    """
    return idata.posterior["season_effects"].values.reshape(-1, 4).astype(float)


def credible_triplet(
    samples: np.ndarray,
    mass: float,
) -> tuple[float, float, float]:
    """Equal-tailed interval summary as (lower, median, upper)."""
    alpha = (1.0 - mass) / 2.0
    lo, mid, hi = [
        float(x)
        for x in np.percentile(
            samples, [100.0 * alpha, 50.0, 100.0 * (1.0 - alpha)]
        )
    ]
    return lo, mid, hi


def time_to_threshold(
    idata: az.InferenceData,
    icb_idx: int,
    last_week: int,
    threshold: float,
    horizon_weeks: int = 12,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Posterior distribution of first-passage time over a pressure threshold.

    Starting from the current posterior for level[icb, T], the AR(1) is
    rolled forward week-by-week.  The season effect for each future week is
    determined by the calendar date implied by last_week + t using the same
    START_DATE and MONTH_TO_SEASON mapping used during model fitting.

    Parameters
    ----------
    idata         : InferenceData from fit_pressure_model
    icb_idx       : index of the target ICB in the posterior
    last_week     : the week number of the most recent training observation
    threshold     : latent-scale threshold (e.g. THRESHOLD_CONCERN = 0.5)
    horizon_weeks : how many weeks ahead to simulate (default 12)
    rng           : optional random Generator for reproducibility

    Returns
    -------
    dict with keys:
        crossing_week   : (n_samples,) — first week index (1-based) at which
                          total pressure exceeds the threshold, or np.nan if it
                          never crosses within the horizon.
        p_cross         : float — fraction of samples that cross.
        median_weeks    : float | None — median crossing week among crossing
                          samples; None if p_cross == 0.
        p10_weeks       : float | None — 10th-percentile crossing week (early risk)
        p90_weeks       : float | None — 90th-percentile crossing week (late risk)
        horizon_weeks   : int — the horizon used
        threshold       : float — the threshold used

    Assumptions
    -----------
    - AR(1) dynamics: level[t+1] = rho * level[t] + N(0, sigma_drift)
      The posterior mean of rho and sigma_drift are drawn per-sample.
    - No mean reversion intercept — the model is zero-mean, so unobserved
      structural drift is not captured; the forecast is conditional on the
      current regime continuing.
    - Seasonal effects are drawn from the posterior and held fixed across the
      forecast horizon (seasonal pattern is assumed stable).
    - Future seasons are determined by calendar month derived from
      START_DATE + timedelta(weeks = last_week + t), matching the encoding
      used at training time.
    - Total pressure = AR(1) level + season_effect.  Observation noise
      (sigma_obs) is intentionally excluded — we are forecasting the latent
      level, not a single noisy observation.
    - If the threshold is already exceeded at t=0 (current week), crossing
      week is recorded as 0.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    post = idata.posterior

    # Flatten chains × draws → (n_samples,)
    level_f     = post["level"].values.reshape(-1, post["level"].shape[2], post["level"].shape[3])
    season_f    = post["season_effects"].values.reshape(-1, 4)
    rho_f       = post["rho"].values.ravel()
    sigma_f     = post["sigma_drift"].values.ravel()
    n_samples   = level_f.shape[0]

    current_level = level_f[:, -1, icb_idx]   # (n_samples,)

    # Pre-compute season index for each future step (same for all samples)
    future_season_idx = np.array([
        _MONTH_TO_SEASON[(_START_DATE + timedelta(weeks=int(last_week + t + 1))).month]
        for t in range(horizon_weeks)
    ])  # (horizon_weeks,)

    # Simulate forward — vectorised across samples
    crossing_week = np.full(n_samples, np.nan)
    level_now = current_level.copy()

    # Check t=0 (current state already over threshold)
    season_now_idx = _MONTH_TO_SEASON[(_START_DATE + timedelta(weeks=int(last_week))).month]
    total_now = level_now + season_f[:, season_now_idx]
    already_crossed = total_now > threshold
    crossing_week[already_crossed] = 0

    active = ~already_crossed  # samples still searching

    for t in range(horizon_weeks):
        if not active.any():
            break
        innov      = rng.normal(0.0, sigma_f[active], size=int(active.sum()))
        level_now[active] = rho_f[active] * level_now[active] + innov
        total      = level_now[active] + season_f[active, future_season_idx[t]]
        crossed    = total > threshold
        # Map back to global sample indices
        active_indices = np.where(active)[0]
        newly_crossed  = active_indices[crossed]
        crossing_week[newly_crossed] = t + 1
        active[newly_crossed] = False

    crossings   = crossing_week[~np.isnan(crossing_week)]
    p_cross     = float(len(crossings) / n_samples)
    median_wks  = float(np.median(crossings)) if len(crossings) > 0 else None
    p10_wks     = float(np.percentile(crossings, 10)) if len(crossings) > 0 else None
    p90_wks     = float(np.percentile(crossings, 90)) if len(crossings) > 0 else None

    return dict(
        crossing_week=crossing_week,
        p_cross=p_cross,
        median_weeks=median_wks,
        p10_weeks=p10_wks,
        p90_weeks=p90_wks,
        horizon_weeks=horizon_weeks,
        threshold=threshold,
    )