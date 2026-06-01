"""
Dashboard utility functions — constants, posterior helpers, and plot primitives.

Imports canonical constants and posterior extractors from core.utils and
adds dashboard-specific helpers (credible intervals, risk classification,
colour mapping, Kalman filter).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import arviz as az

from epiforcasts_nhs.core.utils import (
    LATENT_BASELINE,
    LATENT_SCALE,
    MONTH_TO_SEASON,
    SEASON_NAMES,
    current_pressure_samples,
    current_total_pressure_samples,
)

# Re-export so dashboard modules only need one import target.
__all__ = [
    "LATENT_BASELINE",
    "LATENT_SCALE",
    "MONTH_TO_SEASON",
    "SEASON_NAMES",
    "THRESHOLD_BASELINE",
    "THRESHOLD_CONCERN",
    "THRESHOLD_ELEVATED",
    "DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH",
    "DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN",
    "DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH",
    "DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN",
    "RISK_COLOR_HIGH",
    "RISK_COLOR_MEDIUM",
    "RISK_COLOR_LOW",
    "RISK_COLOR_NEUTRAL",
    "RISK_THRESHOLD_HIGH",
    "RISK_THRESHOLD_MEDIUM",
    "DOT_CHANGE_THRESHOLD_PCT",
    "credible_triplet",
    "current_season_index",
    "pressure_index_samples",
    "level_only_samples",
    "seasonal_effect_samples",
    "resolve_icb_index",
    "risk_band",
    "pressure_colors",
    "dot_travel_colors",
    "week_season_map",
    "season_per_week_samples",
    "kalman_filter",
]


# ─────────────────────────────────────────
# Pressure reference thresholds (demo cut-points only)
# ─────────────────────────────────────────

THRESHOLD_BASELINE = 0.0
THRESHOLD_CONCERN  = 0.5
THRESHOLD_ELEVATED = 1.1


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

RISK_COLOR_HIGH    = "#b91c1c"   # red
RISK_COLOR_MEDIUM  = "#d97706"   # amber
RISK_COLOR_LOW     = "#15803d"   # green
RISK_COLOR_NEUTRAL = "#64748b"   # grey

# P(pressure > RISK_THRESHOLD_HIGH) drives colour in summary charts
RISK_THRESHOLD_HIGH   = 0.25
RISK_THRESHOLD_MEDIUM = 0.08

# ± threshold in % bed-occupancy for direction-of-travel colouring
DOT_CHANGE_THRESHOLD_PCT = 0.5


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
    return int(n_weeks % 4)


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
    return idata.posterior["season_effects"].values.reshape(-1, 4).astype(float)


def credible_triplet(samples: np.ndarray, mass: float) -> tuple[float, float, float]:
    """Equal-tailed credible interval as (lower, median, upper)."""
    alpha = (1.0 - mass) / 2.0
    lo, mid, hi = [
        float(x) for x in np.percentile(samples, [100.0 * alpha, 50.0, 100.0 * (1.0 - alpha)])
    ]
    return lo, mid, hi


# ─────────────────────────────────────────
# Risk classification
# ─────────────────────────────────────────

def risk_band(
    p_elevated: float,
    p_concern: float,
    *,
    pe_hi: float = DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,
    pc_hi: float = DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN,
    pe_med: float = DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,
    pc_med: float = DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,
) -> tuple[str, str]:
    """
    Classify pressure into a qualitative risk band with an explanatory hint.

    Threshold parameters default to the module-level defaults so callers that
    don't need user-adjustable thresholds (e.g. app_fast) can call with two
    positional arguments only.
    """
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
    """
    Map direction-of-travel values (in % bed-occupancy) to colours.

    Values above +DOT_CHANGE_THRESHOLD_PCT are red (rising pressure),
    below -DOT_CHANGE_THRESHOLD_PCT are green (falling), neutral otherwise.
    """
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
    Uncertainty is honest — wide early, narrows as evidence accumulates.
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
