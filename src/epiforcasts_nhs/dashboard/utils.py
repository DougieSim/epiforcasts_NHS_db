"""
Dashboard utility functions — constants, posterior helpers, and plot primitives.

Replaces dashboard.shared and consolidates private helpers previously scattered
across plots.py. Imported by dashboard.plots, dashboard.app, and dashboard.app_fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import arviz as az


# ─────────────────────────────────────────
# Reference thresholds (demo cut-points only)
# ─────────────────────────────────────────

THRESHOLD_BASELINE = 0.0
THRESHOLD_CONCERN  = 0.5
THRESHOLD_ELEVATED = 1.1

# Default UI heuristic gates for qualitative risk labels
DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH    = 0.25
DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN = 0.55
DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH      = 0.08
DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN   = 0.30

SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]


# ─────────────────────────────────────────
# Posterior helpers
# ─────────────────────────────────────────

def resolve_icb_index(idata: az.InferenceData, icb_name: str) -> int:
    """Resolve an ICB name to its posterior array index."""
    icbs = list(idata.attrs.get("icbs", []))
    if not icbs:
        raise ValueError("Posterior metadata does not include an 'icbs' list.")
    try:
        return icbs.index(icb_name)
    except ValueError as exc:
        raise ValueError(f"ICB '{icb_name}' was not found in posterior metadata.") from exc


def current_season_index(idata: az.InferenceData) -> int:
    """Return the season index (0-3) for the most recent week in the posterior."""
    last_season = idata.attrs.get("last_season")
    if last_season is not None:
        return int(last_season)
    # Fallback for older posteriors — not reliable; regenerate to fix
    n_weeks = idata.posterior["level"].values.shape[2]
    return int(n_weeks % 4)


def pressure_index_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    """
    Posterior samples for TOTAL current pressure for one ICB.

    Total = AR(1) level at final week + current season effect.
    Returns flattened (n_chains * n_draws,) array.
    """
    level         = idata.posterior["level"].values
    season_effects = idata.posterior["season_effects"].values

    level_flat  = level.reshape(-1, level.shape[2], level.shape[3])
    season_flat = season_effects.reshape(-1, 4)

    current_level  = level_flat[:, -1, icb_idx]
    current_season = season_flat[:, current_season_index(idata)]
    return (current_level + current_season).astype(float)


def level_only_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    """Posterior samples for the underlying AR(1) level — seasonal component removed."""
    return idata.posterior["level"].values[:, :, -1, icb_idx].ravel().astype(float)


def seasonal_effect_samples(idata: az.InferenceData) -> np.ndarray:
    """Posterior samples for all four season effects. Shape: (n_samples, 4), latent scale."""
    return idata.posterior["season_effects"].values.reshape(-1, 4).astype(float)


def credible_triplet(samples: np.ndarray, mass: float) -> tuple[float, float, float]:
    """Equal-tailed credible interval as (lower, median, upper)."""
    alpha = (1.0 - mass) / 2.0
    lo, mid, hi = [
        float(x) for x in np.percentile(samples, [100.0 * alpha, 50.0, 100.0 * (1.0 - alpha)])
    ]
    return lo, mid, hi


# ─────────────────────────────────────────
# Plot primitives (used by dashboard.plots)
# ─────────────────────────────────────────

def week_season_map(enc: dict) -> np.ndarray:
    """Return (n_weeks,) array of season indices, one per unique week."""
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    return np.array([
        int(season_idx[np.where(week_idx == w)[0][0]])
        for w in range(enc["n_weeks"])
    ])


def season_per_week_samples(season_f: np.ndarray, week_season: np.ndarray) -> np.ndarray:
    """
    Map posterior season-effect samples to per-week values.

    Parameters
    ----------
    season_f    : (S, 4)      posterior season_effects samples
    week_season : (n_weeks,)  season index per week

    Returns
    -------
    (S, n_weeks)
    """
    return season_f[:, week_season]


def kalman_filter(
    obs_series: np.ndarray,
    sigma_drift: float,
    sigma_obs: float,
    level_init: float = 0.0,
    var_init: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward Kalman filter for the local-level model.

    Returns filtered_mean and filtered_std on the latent scale.
    Uncertainty is honest — wide early, narrows as evidence accumulates.
    """
    n                = len(obs_series)
    filtered_mean    = np.zeros(n)
    filtered_std     = np.zeros(n)
    sigma_obs_latent = sigma_obs / 6.0
    m, v             = level_init, var_init

    for t, bed_occ in enumerate(obs_series):
        y_latent = (bed_occ - 85.0) / 6.0
        v_pred   = v + sigma_drift ** 2
        k        = v_pred / (v_pred + sigma_obs_latent ** 2)
        m        = m + k * (y_latent - m)
        v        = (1 - k) * v_pred
        filtered_mean[t] = m
        filtered_std[t]  = np.sqrt(v)

    return filtered_mean, filtered_std
