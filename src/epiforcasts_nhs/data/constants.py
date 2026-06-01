"""
DGP constants for the synthetic NHS dataset generator.

All values are fabricated simulation parameters — not real operational data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


# ─────────────────────────────────────────
# Dataclass definitions
# ─────────────────────────────────────────

@dataclass(frozen=True)
class DatasetConfig:
    n_weeks: int
    max_episodes_per_icb_week: int
    base_episodes_per_week: int
    dgp_version: str
    master_seed: int
    start_date: date


@dataclass(frozen=True)
class SeasonalParams:
    """Cosine seasonal component shared across all indicators."""
    amplitude: float
    peak_week: int      # week index of seasonal peak (mid-February = week 6)
    weeks_per_year: int


@dataclass(frozen=True)
class LatentProcessParams:
    """Parameters for the latent pressure random walk used in ongoing generation."""
    random_walk_sd: float


@dataclass(frozen=True)
class MeasurementNoiseParams:
    """Multiplicative noise added to all generated columns to mimic reporting noise."""
    scale: float


@dataclass(frozen=True)
class EpisodeCategoricals:
    """Categorical domains used when sampling patient episode attributes."""
    age_bands: tuple[str, ...]
    sexes: tuple[str, ...]
    care_settings: tuple[str, ...]
    pathways: tuple[str, ...]
    admission_urgency: tuple[str, ...]
    icd_chapter_buckets: tuple[str, ...]


@dataclass(frozen=True)
class CountParams:
    """Baseline + latent-pressure coefficient + Gaussian noise SD."""
    baseline: float
    lp_coeff: float
    noise_sd: float


@dataclass(frozen=True)
class ClippedParams:
    """CountParams with hard output bounds."""
    baseline: float
    lp_coeff: float
    noise_sd: float
    clip_lo: float
    clip_hi: float


@dataclass(frozen=True)
class BedOccParams:
    """Full bed-occupancy DGP: intercept + lp*lp_scale + seasonal*sea_scale + N(0, noise_sd)."""
    intercept: float
    lp_scale: float
    sea_scale: float
    noise_sd: float
    clip_lo: float
    clip_hi: float


@dataclass(frozen=True)
class Ed4hrParams:
    """ED 4-hour breach count (fraction of A&E, not a simple baseline)."""
    base_rate: float
    lp_coeff: float
    noise_sd: float


@dataclass(frozen=True)
class DischargeParams:
    elec_ratio: float   # proportion of elective admissions contributing to discharges
    noise_sd: float


@dataclass(frozen=True)
class WpiParams:
    """Winter pressure index — has an additional seasonal coefficient."""
    baseline: float
    lp_coeff: float
    sea_coeff: float
    noise_sd: float
    clip_lo: float
    clip_hi: float


@dataclass(frozen=True)
class MissingnessParams:
    resp111_denom: int   # 1/N rows get missing resp_111_calls
    resp111_min: int
    gp_same_denom: int   # 1/N rows get missing gp_same_day_booking_rate_pct
    gp_same_min: int
    elec_can_denom: int  # 1/N rows get missing elective_cancellations
    elec_can_min: int


@dataclass(frozen=True)
class LosParams:
    lp_coeff: float
    exp_scale: float
    noise_base_sd: float
    noise_lp_scale: float
    clip_lo: float
    clip_hi: float
    by_pathway: dict[str, float] = field(default_factory=lambda: {
        "emergency": 4.2,
        "elective":  2.1,
        "maternity": 2.0,
        "other":     3.0,
    })


@dataclass(frozen=True)
class EpisodeParams:
    intensity_lp_coeff: float
    intensity_lp_scale: float
    discharge_alive_rate: float
    readmit_rate_base: float
    readmit_lp_coeff: float
    covid_rate_base: float
    covid_lp_coeff: float


# ─────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────

DATASET = DatasetConfig(
    n_weeks=156,
    max_episodes_per_icb_week=45,
    base_episodes_per_week=4,
    dgp_version="2026.04.8",
    master_seed=2026,
    # Week 0 = first Monday of 2023; each week advances by 7 days.
    start_date=date(2023, 1, 2),
)

# Real NHS England ICB names (public labels).
# `scale` is a unitless simulation parameter — not an official performance score.
ICBS: dict[str, float] = {
    "NHS Birmingham and Solihull ICB":        1.0,
    "NHS Greater Manchester ICB":             1.28,
    "NHS South East London ICB":              0.92,
    "NHS North East and North Cumbria ICB":   1.06,
    "NHS Devon ICB":                          0.98,
    "NHS Nottingham and Nottinghamshire ICB": 1.12,
}

# Column names treated as integer counts (used by apply_measurement_noise).
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


# ─────────────────────────────────────────
# Shared DGP parameters — seasonal & latent process
# ─────────────────────────────────────────

SEASONAL = SeasonalParams(
    amplitude=0.15,
    peak_week=6,        # mid-February peak matches real NHS winter pressure
    weeks_per_year=52,
)

# Used by ongoing generation (generate.py) to extend the latent pressure walk.
LATENT_PROCESS = LatentProcessParams(random_walk_sd=0.12)

MEASUREMENT_NOISE = MeasurementNoiseParams(scale=0.025)


# ─────────────────────────────────────────
# Patient episode categorical domains
# ─────────────────────────────────────────

EPISODE_CATEGORICALS = EpisodeCategoricals(
    age_bands=("0-17", "18-64", "65+"),
    sexes=("F", "M", "Unknown"),
    care_settings=("acute_inpatient", "mental_health_inpatient", "community_crisis", "ed_only"),
    pathways=("emergency", "elective", "maternity", "other"),
    admission_urgency=("immediate", "urgent", "routine", "not_applicable"),
    icd_chapter_buckets=("I", "II", "IX", "X", "XI", "XIV", "XVIII", "XIX", "XXI", "Other"),
)


# ─────────────────────────────────────────
# DGP parameters — bed occupancy
# ─────────────────────────────────────────
# Note: the Bayesian model in core.model uses LATENT_BASELINE=85 / LATENT_SCALE=6,
# which intentionally differ from the DGP — model misspecification is realistic.

BED_OCC = BedOccParams(intercept=84.0, lp_scale=7.0, sea_scale=4.0, noise_sd=3.0, clip_lo=70.0, clip_hi=100.0)


# ─────────────────────────────────────────
# DGP parameters — indicators
# ─────────────────────────────────────────

RESP111_NB_SHAPE = 25  # negative binomial shape — standalone, not a group

DTOC      = CountParams(baseline=2.0,   lp_coeff=2.5,  noise_sd=0.0)  # Poisson; noise_sd unused
AE        = CountParams(baseline=800,   lp_coeff=120,  noise_sd=40)
ED4HR     = Ed4hrParams(base_rate=0.08, lp_coeff=0.04, noise_sd=0.02)
AMB       = CountParams(baseline=40,    lp_coeff=15,   noise_sd=8)
OOH       = CountParams(baseline=1_200, lp_coeff=90,   noise_sd=60)
ELEC_ADM  = CountParams(baseline=350,   lp_coeff=25,   noise_sd=30)
ELEC_CAN  = CountParams(baseline=25,    lp_coeff=18,   noise_sd=10)
ACUTE     = CountParams(baseline=520,   lp_coeff=70,   noise_sd=35)
DISCHARGE = DischargeParams(elec_ratio=0.95, noise_sd=25)
INF_ISO   = CountParams(baseline=15,    lp_coeff=5,    noise_sd=4)
DTOC21    = CountParams(baseline=35,    lp_coeff=22,   noise_sd=12)
SOC_CARE  = CountParams(baseline=28,    lp_coeff=12,   noise_sd=9)
CRISIS    = CountParams(baseline=210,   lp_coeff=35,   noise_sd=25)
NHS111    = CountParams(baseline=1_800, lp_coeff=140,  noise_sd=90)

STAFF_ABS = ClippedParams(baseline=4.5, lp_coeff=1.8,  noise_sd=0.6,  clip_lo=1.0,  clip_hi=22.0)
CC_OCC    = ClippedParams(baseline=72,  lp_coeff=6,    noise_sd=3,    clip_lo=45.0, clip_hi=100.0)
MH_OCC    = ClippedParams(baseline=78,  lp_coeff=4,    noise_sd=2.5,  clip_lo=50.0, clip_hi=100.0)
GP_SAME   = ClippedParams(baseline=42,  lp_coeff=3,    noise_sd=2,    clip_lo=15.0, clip_hi=85.0)   # lp_coeff subtracted
MLOS      = ClippedParams(baseline=5.2, lp_coeff=0.35, noise_sd=0.25, clip_lo=2.0,  clip_hi=18.0)

WPI = WpiParams(
    baseline=3.5, lp_coeff=0.9, sea_coeff=2.0,
    noise_sd=0.35, clip_lo=0.0, clip_hi=10.0,
)


# ─────────────────────────────────────────
# DGP parameters — missingness
# ─────────────────────────────────────────

MISSINGNESS = MissingnessParams(
    resp111_denom=50,  resp111_min=8,
    gp_same_denom=80,  gp_same_min=5,
    elec_can_denom=100, elec_can_min=4,
)


# ─────────────────────────────────────────
# DGP parameters — patient episodes
# ─────────────────────────────────────────

LOS = LosParams(
    lp_coeff=0.55, exp_scale=1.8,
    noise_base_sd=0.35, noise_lp_scale=0.15,
    clip_lo=0.25, clip_hi=45.0,
)

EPISODE = EpisodeParams(
    intensity_lp_coeff=0.25,
    intensity_lp_scale=6.0,
    discharge_alive_rate=0.985,
    readmit_rate_base=0.06,
    readmit_lp_coeff=0.03,
    covid_rate_base=0.02,
    covid_lp_coeff=0.01,
)
