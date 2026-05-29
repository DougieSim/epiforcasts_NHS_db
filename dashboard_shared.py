"""
Shared dashboard constants and posterior helper utilities.

Centralises model-side logic used across dashboard variants.
Updated to read from the AR(1) + seasonal model — level[icb, T] is the
current underlying pressure; total pressure adds the current season's effect.
"""

from __future__ import annotations

import numpy as np
import arviz as az


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