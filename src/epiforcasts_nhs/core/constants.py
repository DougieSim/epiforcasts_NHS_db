"""
Constants for the Bayesian pressure model.

Covers the observation model, pressure thresholds, season configuration,
MCMC sampling strategies, model priors, and cache parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────
# Dataclass definitions
# ─────────────────────────────────────────

@dataclass(frozen=True)
class LatentScaleParams:
    """Maps latent pressure to bed occupancy: obs = baseline + latent * scale.

    Note: the DGP in data.constants uses (84, 7) — intentional misspecification.
    """
    baseline: float
    scale: float


@dataclass(frozen=True)
class PressureThresholds:
    """Latent-scale cut-points for clinical pressure classification.

    Demo values only — not NHS operational thresholds.
    Used by model, cache, and dashboard so core modules never import from dashboard.
    """
    baseline: float
    concern: float
    elevated: float


@dataclass(frozen=True)
class SummaryParams:
    """Parameters controlling clinical pressure summary output."""
    ci_lower_pct: float             # lower bound of credible interval (%)
    ci_upper_pct: float             # upper bound of credible interval (%)
    lookback_weeks: int             # window for direction-of-travel calculations
    season_alignment_display_weeks: int  # weeks shown in season-alignment console check


@dataclass(frozen=True)
class ForecastParams:
    """Parameters for the forward AR(1) weeks-to-threshold projection."""
    horizon_weeks: int   # how far ahead to simulate first-passage crossings
    seed: int            # RNG seed for reproducible forward simulation
    weeks_per_season: int  # equal-season block length (fallback season stepping)


@dataclass(frozen=True)
class SeasonConfig:
    """Season names, count, and calendar month → season index mapping."""
    names: tuple[str, ...]
    n_seasons: int
    month_to_season: dict[int, int] = field(default_factory=lambda: {
        12: 0,  1: 0,  2: 0,   # Winter
         3: 1,  4: 1,  5: 1,   # Spring
         6: 2,  7: 2,  8: 2,   # Summer
         9: 3, 10: 3, 11: 3,   # Autumn
    })


@dataclass(frozen=True)
class ModelSamplingConfig:
    """MCMC sampling parameters for the core.model entry point."""
    draws_fast: int
    tune_fast: int
    draws_full: int
    tune_full: int
    chains: int
    cores: int
    target_accept: float


@dataclass(frozen=True)
class ModelPriors:
    """Prior distribution parameters for the AR(1) Bayesian model."""
    mu_national_sd: float
    sigma_icb_rate: float
    rho_alpha: float        # Beta(alpha, beta) — symmetric prior, data-driven persistence
    rho_beta: float
    sigma_drift_lam: float
    sigma_season_lam: float
    sigma_obs_lam: float


@dataclass(frozen=True)
class InferenceSamplingConfig:
    """Sampling strategy parameters for the core.inference ADVI/NUTS runner."""
    draws_fast: int
    draws_full: int
    tune_full: int
    advi_steps_min_fallback: int  # minimum steps when NUTS fails and we fall back to ADVI
    nuts_target_accept_1: float   # first NUTS attempt
    nuts_target_accept_2: float   # second NUTS attempt (more conservative)
    nuts_init_method: str
    save_retry_attempts: int      # lock-retry count before falling back to artifact path


@dataclass(frozen=True)
class CacheConfig:
    """Cache management parameters."""
    ttl_hours_default: int
    version: int
    summary_quantile_keys: tuple[str, ...]
    summary_quantile_pcts: tuple[int, ...]


# ─────────────────────────────────────────
# Instances
# ─────────────────────────────────────────

LATENT_SCALE = LatentScaleParams(baseline=85.0, scale=6.0)

PRESSURE_THRESHOLDS = PressureThresholds(baseline=0.0, concern=0.5, elevated=1.1)

SUMMARY = SummaryParams(
    ci_lower_pct=10.0,
    ci_upper_pct=90.0,
    lookback_weeks=4,
    season_alignment_display_weeks=8,
)

SEASONS = SeasonConfig(
    names=("Winter", "Spring", "Summer", "Autumn"),
    n_seasons=4,
)

FORECAST = ForecastParams(
    horizon_weeks=52,
    seed=42,
    weeks_per_season=13,
)

MODEL_SAMPLING = ModelSamplingConfig(
    draws_fast=400, tune_fast=400,
    draws_full=2000, tune_full=2000,
    chains=4, cores=4,
    target_accept=0.95,
)

MODEL_PRIORS = ModelPriors(
    mu_national_sd=1.0,
    sigma_icb_rate=1.0,
    rho_alpha=2.0, rho_beta=2.0,
    sigma_drift_lam=1.0,
    sigma_season_lam=5.0,
    sigma_obs_lam=5.0,
)

INFERENCE_SAMPLING = InferenceSamplingConfig(
    draws_fast=400,
    draws_full=1000, tune_full=1000,
    advi_steps_min_fallback=30_000,
    nuts_target_accept_1=0.90,
    nuts_target_accept_2=0.95,
    nuts_init_method="adapt_diag",
    save_retry_attempts=3,
)

CACHE = CacheConfig(
    ttl_hours_default=24,
    version=1,
    summary_quantile_keys=("q05", "q25", "q50", "q75", "q95"),
    summary_quantile_pcts=(5, 25, 50, 75, 95),
)
