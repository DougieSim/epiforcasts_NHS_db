"""Shared dashboard constants and posterior helper utilities.

Keep UI files focused on presentation while this module centralizes model-side
logic used across dashboard variants.
"""

from __future__ import annotations

import numpy as np
import arviz as az


# Reference levels on the latent index (demo-only communication cut-points).
THRESHOLD_BASELINE = 0.0
THRESHOLD_CONCERN = 0.5
THRESHOLD_ELEVATED = 1.1


# Default UI heuristic gates for qualitative risk summaries.
DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH = 0.25
DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN = 0.55
DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH = 0.08
DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN = 0.30


def pressure_index_samples(idata: az.InferenceData, icb_idx: int) -> np.ndarray:
    post  = idata.posterior
    level = post["level"].values   # (chains, draws, n_weeks, n_icb)
    return level[:, :, -1, icb_idx].astype(float).ravel()


def credible_triplet(samples: np.ndarray, mass: float) -> tuple[float, float, float]:
    """Equal-tailed interval summary as (lower, median, upper)."""
    alpha = (1.0 - mass) / 2.0
    lo, mid, hi = [
        float(x)
        for x in np.percentile(samples, [100.0 * alpha, 50.0, 100.0 * (1.0 - alpha)])
    ]
    return lo, mid, hi


def resolve_icb_index(idata: az.InferenceData, icb_name: str) -> int:
    """Resolve an ICB index from posterior metadata.

    Raises
    ------
    ValueError
        If ICB metadata is missing or the ICB is not present.
    """
    icbs = list(idata.attrs.get("icbs", []))
    if not icbs:
        raise ValueError("Posterior metadata does not include an 'icbs' list.")
    try:
        return icbs.index(icb_name)
    except ValueError as exc:
        raise ValueError(f"ICB '{icb_name}' was not found in posterior metadata.") from exc