"""
Dashboard utility functions — constants, posterior helpers, and plot primitives.

Pressure thresholds and domain constants are imported from core.utils so the
dashboard never owns a separate definition. Visual / UI constants are defined
here as the single source of truth for the dashboard layer.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import arviz as az

from epiforcasts_nhs.core.utils import (
    LATENT_BASELINE,
    LATENT_SCALE,
    LOOKBACK_WEEKS,
    MONTH_TO_SEASON,
    N_SEASONS,
    SEASON_NAMES,
    THRESHOLD_BASELINE,
    THRESHOLD_CONCERN,
    THRESHOLD_ELEVATED,
    current_pressure_samples,
    current_total_pressure_samples,
)

# Re-export so dashboard modules only need one import target.
__all__ = [
    # From core
    "LATENT_BASELINE", "LATENT_SCALE", "LOOKBACK_WEEKS", "MONTH_TO_SEASON",
    "N_SEASONS", "SEASON_NAMES", "THRESHOLD_BASELINE", "THRESHOLD_CONCERN",
    "THRESHOLD_ELEVATED",
    # Risk band
    "DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH", "DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN",
    "DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH",   "DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN",
    "RISK_COLOR_HIGH", "RISK_COLOR_MEDIUM", "RISK_COLOR_LOW", "RISK_COLOR_NEUTRAL",
    "RISK_THRESHOLD_HIGH", "RISK_THRESHOLD_MEDIUM", "DOT_CHANGE_THRESHOLD_PCT",
    # Visual
    "SEASON_COLORS",
    "CI_89_LOWER", "CI_89_UPPER",
    "CI_80_LOWER", "CI_80_UPPER",
    "CI_DEFAULT_MASS",
    "BED_OCC_REF_CONCERN", "BED_OCC_REF_HIGH",
    "KALMAN_CI_Z_90",
    # Helpers
    "credible_triplet", "current_season_index",
    "pressure_index_samples", "level_only_samples", "seasonal_effect_samples",
    "resolve_icb_index", "risk_band", "pressure_colors", "dot_travel_colors",
    "week_season_map", "season_per_week_samples", "kalman_filter",
]


# ─────────────────────────────────────────
# Risk band defaults
# ─────────────────────────────────────────

DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH    = 0.25
DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN = 0.55
DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH      = 0.08
DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN   = 0.30


# ─────────────────────────────────────────
# Risk / direction-of-travel colours
# ─────────────────────────────────────────

RISK_COLOR_HIGH    = "#b91c1c"
RISK_COLOR_MEDIUM  = "#d97706"
RISK_COLOR_LOW     = "#15803d"
RISK_COLOR_NEUTRAL = "#64748b"

RISK_THRESHOLD_HIGH   = 0.25
RISK_THRESHOLD_MEDIUM = 0.08

# ± threshold in % bed-occupancy for direction-of-travel colouring
DOT_CHANGE_THRESHOLD_PCT = 0.5


# ─────────────────────────────────────────
# Visual / chart constants
# ─────────────────────────────────────────

# Colours for Winter / Spring / Summer / Autumn bars
SEASON_COLORS = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]

# Credible interval percentiles
CI_89_LOWER: float = 5.5    # 89% CI lower bound (HDI convention)
CI_89_UPPER: float = 94.5   # 89% CI upper bound
CI_80_LOWER: float = 10.0   # 80% CI lower bound (trajectory plots)
CI_80_UPPER: float = 90.0   # 80% CI upper bound

# Default credible mass shown in the UI
CI_DEFAULT_MASS: float = 0.9

# Bed-occupancy reference lines on charts (not thresholds — operational context only)
BED_OCC_REF_CONCERN: float = 95.0   # amber: approaching strain
BED_OCC_REF_HIGH:    float = 100.0  # red: at or beyond safe capacity

# z-score for 90% normal interval used in the Kalman-filter trajectory CI
KALMAN_CI_Z_90: float = 1.645


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
    # Fallback for older posteriors — regenerate to fix
    n_weeks = idata.posterior["level"].values.shape[2]
    return int(n_weeks % N_SEASONS)


def pressure_index_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    """
    Posterior samples for TOTAL current pressure for one ICB.

    Total = AR(1) level at final week + current season effect.
    Returns flattened (n_chains * n_draws,) array.
    """
    return current_total_pressure_samples(idata, icb_idx, current_season_index(idata))


def level_only_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    """Posterior samples for the underlying AR(1) level — seasonal component removed."""
    return current_pressure_samples(idata, icb_idx)


def seasonal_effect_samples(idata: az.InferenceData) -> np.ndarray:
    """Posterior samples for all four season effects. Shape: (n_samples, 4), latent scale."""
    return idata.posterior["season_effects"].values.reshape(-1, N_SEASONS).astype(float)


def credible_triplet(samples: np.ndarray, mass: float) -> tuple[float, float, float]:
    """Equal-tailed credible interval as (lower, median, upper)."""
    alpha = (1.0 - mass) / 2.0
    lo, mid, hi = [
        float(x) for x in np.percentile(
            samples, [100.0 * alpha, 50.0, 100.0 * (1.0 - alpha)]
        )
    ]
    return lo, mid, hi


# ─────────────────────────────────────────
# Risk classification
# ─────────────────────────────────────────

def risk_band(
    p_elevated: float,
    p_concern: float,
    *,
    pe_hi:  float = DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,
    pc_hi:  float = DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN,
    pe_med: float = DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,
    pc_med: float = DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,
) -> tuple[str, str]:
    """Classify pressure into a qualitative risk band with an explanatory hint."""
    if p_elevated >= pe_hi or p_concern >= pc_hi:
        return "Elevated", "Prioritise review of capacity, flow, and escalation plans (indicative only)."
    if p_elevated >= pe_med or p_concern >= pc_med:
        return "Medium", "Worth closer monitoring; corroborate with local intelligence."
    return "Low", "No strong signal of unusually high modelled pressure; stay vigilant to new data."


# ─────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────

def pressure_colors(p_high_values: Iterable[float]) -> list[str]:
    """Map P(pressure above high reference) to risk colours."""
    return [
        RISK_COLOR_HIGH if p > RISK_THRESHOLD_HIGH
        else RISK_COLOR_MEDIUM if p > RISK_THRESHOLD_MEDIUM
        else RISK_COLOR_LOW
        for p in p_high_values
    ]


def dot_travel_colors(dot_values_pct: Iterable[float]) -> list[str]:
    """Map direction-of-travel values (% bed-occupancy) to colours."""
    return [
        RISK_COLOR_HIGH if d > DOT_CHANGE_THRESHOLD_PCT
        else RISK_COLOR_LOW if d < -DOT_CHANGE_THRESHOLD_PCT
        else RISK_COLOR_NEUTRAL
        for d in dot_values_pct
    ]


# ─────────────────────────────────────────
# Plot primitives
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

    Returns (S, n_weeks).
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
    """
    n                = len(obs_series)
    filtered_mean    = np.zeros(n)
    filtered_std     = np.zeros(n)
    sigma_obs_latent = sigma_obs / LATENT_SCALE
    m, v             = level_init, var_init

    for t, bed_occ in enumerate(obs_series):
        y_latent = (bed_occ - LATENT_BASELINE) / LATENT_SCALE
        v_pred   = v + sigma_drift ** 2
        k        = v_pred / (v_pred + sigma_obs_latent ** 2)
        m        = m + k * (y_latent - m)
        v        = (1 - k) * v_pred
        filtered_mean[t] = m
        filtered_std[t]  = np.sqrt(v)

    return filtered_mean, filtered_std
