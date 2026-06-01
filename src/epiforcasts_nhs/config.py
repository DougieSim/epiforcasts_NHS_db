"""
Central configuration — single source of truth for all cross-cutting constants.

Import from here rather than hardcoding paths, ports, seeds, or thresholds
in individual modules.
"""

from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────
# Runtime file paths (relative to working directory)
# ─────────────────────────────────────────

WEEKLY_CSV         = Path("synthetic_nhs_pressure.csv")
PATIENT_CSV        = Path("synthetic_patient_episodes.csv")
POSTERIORS_PATH    = Path("posteriors.nc")
STAGED_POSTERIORS  = Path(".posteriors_new.nc")   # atomic-swap staging path
CI_POSTERIORS_PATH = Path("ci_posteriors.nc")
CI_METADATA_PATH   = Path("ci_posteriors_metadata.nc")
CACHE_DIR          = Path(".cache")
ARTIFACTS_DIR      = Path(".artifacts")            # fallback when posteriors.nc is locked

# ─────────────────────────────────────────
# Inference defaults
# ─────────────────────────────────────────

DEFAULT_RANDOM_SEED = 42
DEFAULT_ADVI_STEPS  = 20_000

# ─────────────────────────────────────────
# Dashboard serving
# ─────────────────────────────────────────

STREAMLIT_PREFERRED_PORT  = 8501
STREAMLIT_MAX_PORT        = 8510
DASHBOARD_POLL_INTERVAL_S = 10

# ─────────────────────────────────────────
# Governance / CI acceptance checks
# ─────────────────────────────────────────

CI_MIN_DRAWS  = 200   # posterior must have at least this many draws
CI_MIN_OBS    = 100   # metadata n_obs floor
CI_MIN_ICBS   = 1     # metadata n_icbs floor
CI_ADVI_STEPS = 500   # fast step count used in evidence-cycle runs

VALID_INFERENCE_METHODS = frozenset({"advi-fast", "advi-fallback", "nuts-1", "nuts-2"})

GITIGNORE_ARTIFACT_ENTRIES = (".cache/", "posteriors.nc", "posteriors_metadata.nc")

# ─────────────────────────────────────────
# Ops
# ─────────────────────────────────────────

COVARIATE_CHECK_HOLDOUT_WEEKS = 12   # weeks held out for covariate check test set
COVARIATE_CHECK_LAGS          = (1, 2, 4)   # lag windows (weeks) for lagged-correlation check
