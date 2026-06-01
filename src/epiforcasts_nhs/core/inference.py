"""
Configurable AR(1) inference runner and posterior persistence.

Uses the same AR(1) model as core.model but exposes finer-grained control
over the sampling strategy (ADVI vs NUTS, seed, step count) for CLI use.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from epiforcasts_nhs.config import (
    ARTIFACTS_DIR,
    DEFAULT_ADVI_STEPS,
    DEFAULT_RANDOM_SEED,
)
from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.core.model import build_model
import epiforcasts_nhs.core.constants as constants
from epiforcasts_nhs.core.utils import (
    check_season_alignment,
    current_season_from_enc,
    encode,
    prepare_data,
)


# ─────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────

def fit_pressure_model(
    df: pd.DataFrame,
    *,
    fast: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    advi_steps: int = DEFAULT_ADVI_STEPS,
) -> tuple[pm.Model, az.InferenceData, dict]:
    """
    Fit the AR(1) pressure model with a configurable sampling strategy.

    Parameters
    ----------
    df          : Weekly panel — must include 'icb', 'bed_occupancy', 'month'.
    fast        : If True, use ADVI (400 draws). If False, try NUTS with
                  progressive retries and ADVI fallback.
    random_seed : Seed for reproducibility.
    advi_steps  : ADVI optimisation steps used in fast mode and NUTS fallback.

    Returns
    -------
    model : Fitted PyMC model.
    idata : ArviZ InferenceData with posterior samples and metadata attrs.
    enc   : Encoding dict (icb_codes, week_idx, categories, …).
    """
    data = prepare_data(df)
    enc  = encode(data)

    print(f"Rows:  {len(data)}")
    print(f"Weeks: {data['week'].min()}–{data['week'].max()}")
    print(f"ICBs:  {list(enc['categories'])}")
    check_season_alignment(enc, data)

    model = build_model(enc)
    print(f"Inference config: fast={fast}, seed={random_seed}, advi_steps={advi_steps}")

    s = constants.INFERENCE_SAMPLING
    if fast:
        idata, inference_method = _sample_advi(
            model, advi_steps=advi_steps, draws=s.draws_fast, random_seed=random_seed
        )
    else:
        idata, inference_method = _sample_nuts_with_fallback(
            model, draws=s.draws_full, tune=s.tune_full,
            advi_steps=advi_steps, random_seed=random_seed,
        )

    idata.attrs = dict(idata.attrs or {})
    idata.attrs.update({
        "icbs":                 list(enc["categories"]),
        "last_season":          current_season_from_enc(enc),
        "inference_method":     inference_method,
        "inference_fast_mode":  int(bool(fast)),
        "inference_seed":       int(random_seed),
        "inference_advi_steps": int(advi_steps),
    })

    return model, idata, enc


def _sample_advi(
    model: pm.Model,
    *,
    advi_steps: int,
    draws: int,
    random_seed: int,
) -> tuple[az.InferenceData, str]:
    print("Fast mode: ADVI approximation.")
    with model:
        approx = pm.fit(n=advi_steps, method="advi", progressbar=True, random_seed=random_seed)
        idata  = approx.sample(draws=draws, return_inferencedata=True, random_seed=random_seed)
    return idata, "advi-fast"


def _sample_nuts_with_fallback(
    model: pm.Model,
    *,
    draws: int,
    tune: int,
    advi_steps: int,
    random_seed: int,
) -> tuple[az.InferenceData, str]:
    """Try NUTS with two target-accept configurations; fall back to ADVI."""
    s = constants.INFERENCE_SAMPLING
    attempts = [
        {"target_accept": s.nuts_target_accept_1, "init": s.nuts_init_method},
        {"target_accept": s.nuts_target_accept_2, "init": s.nuts_init_method},
    ]
    last_error: Exception | None = None

    for idx, cfg in enumerate(attempts, start=1):
        try:
            print(f"NUTS attempt {idx}/{len(attempts)} (target_accept={cfg['target_accept']})")
            with model:
                idata = pm.sample(
                    draws=draws, tune=tune, chains=1, cores=1,
                    target_accept=cfg["target_accept"], init=cfg["init"],
                    progressbar=True, compute_convergence_checks=True,
                    random_seed=random_seed,
                )
            return idata, f"nuts-{idx}"
        except ValueError as exc:
            last_error = exc
            if "Not enough samples to build a trace" not in str(exc):
                raise
            print("[WARN] NUTS attempt failed; trying next configuration...")

    print("[WARN] All NUTS attempts failed; falling back to ADVI.")
    if last_error is not None:
        print(f"Last NUTS error: {last_error}")
    idata, _ = _sample_advi(
        model, advi_steps=max(advi_steps, constants.INFERENCE_SAMPLING.advi_steps_min_fallback),
        draws=draws, random_seed=random_seed,
    )
    return idata, "advi-fallback"


# ─────────────────────────────────────────
# Posterior persistence
# ─────────────────────────────────────────

def save_posterior_summaries(
    idata: az.InferenceData,
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Persist InferenceData to NetCDF with lock-safe retry, write a JSON
    metadata sidecar, and warm the downstream cache.

    Returns the path the posterior was actually written to.
    """
    # n_obs counts all rows (including England aggregate).
    # icbs / last_season already set correctly by fit_pressure_model — do not overwrite.
    idata.attrs = dict(idata.attrs or {})
    idata.attrs["n_obs"] = len(df)

    actual_path = _save_netcdf_with_retry(idata, output_path)
    _write_metadata_sidecar(df, actual_path)
    _warm_cache(actual_path)
    return actual_path


def _save_netcdf_with_retry(idata: az.InferenceData, output_path: Path) -> Path:
    """Write NetCDF; on Windows file-lock error fall back to a timestamped artifact."""
    lock_error: OSError | None = None

    for attempt in range(1, constants.INFERENCE_SAMPLING.save_retry_attempts + 1):
        try:
            idata.to_netcdf(str(output_path))
            print(f"[OK] Posterior saved to {output_path}")
            return output_path
        except OSError as exc:
            if "unable to lock file" not in str(exc).lower():
                raise
            lock_error = exc
            print(f"[WARN] File lock on {output_path} (attempt {attempt}/{constants.INFERENCE_SAMPLING.save_retry_attempts}). Retrying...")
            time.sleep(attempt)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fallback_path = ARTIFACTS_DIR / f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.nc"
    idata.to_netcdf(str(fallback_path))
    print(f"[WARN] Locked file; saved to fallback: {fallback_path}")
    return fallback_path


def _write_metadata_sidecar(df: pd.DataFrame, posterior_path: Path) -> None:
    # Exclude England — the model was only fitted on individual ICBs.
    icb_names     = sorted(df.loc[df["icb"] != "England", "icb"].unique().tolist())
    metadata      = {"icbs": icb_names, "n_icbs": len(icb_names), "n_obs": len(df)}
    metadata_path = posterior_path.with_stem(posterior_path.stem + "_metadata")
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[OK] Metadata saved to {metadata_path}")


def _warm_cache(posterior_path: Path) -> None:
    print("Warming cache for instant UI access…")
    cache = CacheManager(posteriors_path=posterior_path)
    if cache.warm_cache():
        print(cache.get_status_report())
    else:
        print("[WARN] Cache warming failed; UI will be slower on first access")
