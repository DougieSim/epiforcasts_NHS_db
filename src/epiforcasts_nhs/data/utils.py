"""
Helper functions for synthetic data generation.

Used by both data.generator (initial bulk creation) and data.generate (ongoing
interval-based generation).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import epiforcasts_nhs.data.constants as constants

if TYPE_CHECKING:
    from numpy.random import Generator


def rng_for_icb(icb: str) -> np.random.Generator:
    """Deterministic, independent RNG per ICB (stable across runs)."""
    digest = hashlib.sha256(f"{constants.DATASET.dgp_version}|{icb}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") ^ constants.DATASET.master_seed
    return np.random.default_rng(seed)


def latent_series(rng: np.random.Generator, scale: float, n_weeks: int) -> np.ndarray:
    """Random-walk latent pressure shared across indicators for one ICB."""
    return np.cumsum(rng.normal(0, constants.LATENT_PROCESS.random_walk_sd, n_weeks)) + scale


def round_counts(x: np.ndarray) -> np.ndarray:
    """Clip to zero and round to integer counts."""
    return np.maximum(0, np.rint(x)).astype(int)


def apply_measurement_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    count_columns: frozenset[str],
) -> None:
    """Add realistic measurement noise in-place to numeric columns."""
    scale = constants.MEASUREMENT_NOISE.scale
    skip  = {"week", "week_date", "month", "icb", "synthetic_dgp_version"}
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        mask = s.notna()
        if col in count_columns:
            sigma  = scale * (np.abs(s[mask].to_numpy()) ** 0.5 + 2.0)
            jitter = rng.normal(0.0, sigma, size=mask.sum())
            df.loc[mask, col] = round_counts(s[mask].to_numpy().astype(float) + jitter)
        else:
            base  = s[mask].to_numpy(dtype=float)
            sigma = scale * (np.abs(base) + 1.0)
            df.loc[mask, col] = base + rng.normal(0.0, sigma, size=len(base))


def episode_id(icb: str, week: int, seq: int) -> str:
    """Stable synthetic episode identifier."""
    raw = f"{constants.DATASET.dgp_version}|{icb}|{week}|{seq}".encode()
    return "SYN-" + hashlib.sha256(raw).hexdigest()[:16]


def england_aggregate(icb_weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute England aggregate as the mean across all ICBs per week."""
    eng = icb_weekly.groupby("week", as_index=False).mean(numeric_only=True).assign(icb="England")
    eng["synthetic_dgp_version"] = constants.DATASET.dgp_version
    return eng
