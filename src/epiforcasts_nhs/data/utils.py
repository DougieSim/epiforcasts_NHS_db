"""
Shared constants and private helper functions for synthetic data generation.

Used by both data.generator (initial bulk creation) and data.generate (ongoing
interval-based generation).
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.random import Generator

# ─────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────

SYNTHETIC_DGP_VERSION = "2026.04.8"
MASTER_SEED = 2026

# Week 0 = first Monday of 2023; each week advances by 7 days.
START_DATE = date(2023, 1, 2)

AGE_BANDS = ["0-17", "18-64", "65+"]
SEXES = ["F", "M", "Unknown"]
CARE_SETTINGS = ["acute_inpatient", "mental_health_inpatient", "community_crisis", "ed_only"]
PATHWAYS = ["emergency", "elective", "maternity", "other"]
ADMISSION_URGENCY = ["immediate", "urgent", "routine", "not_applicable"]
ICD_CHAPTER_BUCKETS = [
    "I", "II", "IX", "X", "XI", "XIV", "XVIII", "XIX", "XXI", "Other",
]

COUNT_COLS: frozenset[str] = frozenset({
    "resp_111_calls",
    "dtoc_patients",
    "ae_type1_attendances",
    "ed_4hr_breach_count",
    "ambulance_category_red_calls",
    "elective_admissions",
    "elective_cancellations",
    "acute_admissions_nonelective",
    "total_discharges",
    "delayed_transfers_ge_21_days",
    "ooh_primary_care_contacts",
    "community_crisis_team_contacts",
    "nhs_111_online_assessments_completed",
    "social_care_package_delays_new",
    "infection_isolation_beds_occupied",
})

MEASUREMENT_NOISE_SCALE = 0.025

# ─────────────────────────────────────────
# Data-generating process (DGP) parameters
# ─────────────────────────────────────────
# These define the synthetic bed-occupancy formula and latent dynamics.
# Both generator.py (bulk creation) and generate.py (week-by-week extension)
# must use these values so the two generation paths stay consistent.

# Bed-occupancy formula: bed_occ = BED_OCC_INTERCEPT + lp * BED_OCC_LP_SCALE
#                                  + seasonal * BED_OCC_SEA_SCALE + N(0, 3)
# Note: the Bayesian model in core.model uses LATENT_BASELINE=85 / LATENT_SCALE=6,
# which intentionally differ from the DGP — model misspecification is realistic.
BED_OCC_INTERCEPT:  float = 84.0
BED_OCC_LP_SCALE:   float = 7.0
BED_OCC_SEA_SCALE:  float = 4.0

# Seasonal pattern: amplitude * cos(2π(t - peak_week) / weeks_per_year)
SEASONAL_AMPLITUDE: float = 0.15
SEASONAL_PEAK_WEEK: int   = 6       # mid-February peak matches real NHS winter pressure
WEEKS_PER_YEAR:     int   = 52

# Latent random-walk standard deviation (weekly innovation size)
LP_RANDOM_WALK_SD:  float = 0.12


# ─────────────────────────────────────────
# RNG helpers
# ─────────────────────────────────────────

def rng_for_icb(icb: str) -> np.random.Generator:
    """Deterministic, independent RNG per ICB (stable across runs)."""
    digest = hashlib.sha256(f"{SYNTHETIC_DGP_VERSION}|{icb}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") ^ MASTER_SEED
    return np.random.default_rng(seed)


def latent_series(rng: np.random.Generator, scale: float, n_weeks: int) -> np.ndarray:
    """Random-walk latent pressure shared across indicators for one ICB."""
    return np.cumsum(rng.normal(0, LP_RANDOM_WALK_SD, n_weeks)) + scale


def round_counts(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """Clip to zero and round to integer counts."""
    return np.maximum(0, np.rint(x)).astype(int)


def apply_measurement_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    count_columns: frozenset[str],
) -> None:
    """Add realistic measurement noise in-place to numeric columns."""
    skip = {"week", "week_date", "month", "icb", "synthetic_dgp_version"}
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        mask = s.notna()
        if col in count_columns:
            sigma = MEASUREMENT_NOISE_SCALE * (np.abs(s[mask].to_numpy()) ** 0.5 + 2.0)
            jitter = rng.normal(0.0, sigma, size=mask.sum())
            df.loc[mask, col] = round_counts(rng, s[mask].to_numpy().astype(float) + jitter)
        else:
            base = s[mask].to_numpy(dtype=float)
            sigma = MEASUREMENT_NOISE_SCALE * (np.abs(base) + 1.0)
            df.loc[mask, col] = base + rng.normal(0.0, sigma, size=len(base))


def episode_id(icb: str, week: int, seq: int) -> str:
    """Stable synthetic episode identifier."""
    raw = f"{SYNTHETIC_DGP_VERSION}|{icb}|{week}|{seq}".encode()
    return "SYN-" + hashlib.sha256(raw).hexdigest()[:16]



def england_aggregate(icb_weekly: pd.DataFrame) -> pd.DataFrame:
    """Compute England aggregate as the mean across all ICBs per week."""
    eng = icb_weekly.groupby("week", as_index=False).mean(numeric_only=True).assign(icb="England")
    eng["synthetic_dgp_version"] = SYNTHETIC_DGP_VERSION
    return eng
