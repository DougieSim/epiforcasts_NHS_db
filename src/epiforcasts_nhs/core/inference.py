"""
Offline Bayesian inference — model fitting and posterior persistence.

This module contains the library-level inference logic. It is called by
cli.inference (entry point) and core.pipeline (daemon orchestration).
The model here is a fast ICB-offset model suited to the CLI workflow;
the full AR(1) model lives in core.model.
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

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.core.utils import LATENT_BASELINE, LATENT_SCALE


# ─────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────

def fit_pressure_model(
    df: pd.DataFrame,
    *,
    fast: bool = True,
    random_seed: int = 42,
    advi_steps: int = 20_000,
) -> tuple[pm.Model, az.InferenceData]:
    """
    Fit Bayesian latent pressure model on bed occupancy.

    Parameters
    ----------
    df          : Data with columns 'icb', 'bed_occupancy', etc.
    fast        : If True, use ADVI (400 draws). If False, try NUTS with fallback.
    random_seed : Seed for reproducibility.
    advi_steps  : ADVI optimisation steps in fast / fallback paths.

    Returns
    -------
    model : Fitted PyMC model.
    idata : ArviZ InferenceData (posterior + diagnostics).
    """
    icb_codes = df["icb"].astype("category").cat.codes.values
    beds      = df["bed_occupancy"].values
    draws     = 400 if fast else 1000
    tune      = 400 if fast else 1000

    print(
        f"Inference config: draws={draws}, tune={tune}, fast={fast}, "
        f"seed={random_seed}, advi_steps={advi_steps}"
    )

    with pm.Model() as model:
        mu_national = pm.Normal("mu_national", 0, 1)
        sigma_icb   = pm.HalfNormal("sigma_icb", sigma=1)
        icb_offset  = pm.Normal("icb_offset", 0, 1, shape=len(np.unique(icb_codes)))
        icb_effect  = pm.Deterministic("icb_effect", icb_offset * sigma_icb)

        latent_pressure = mu_national + icb_effect[icb_codes]
        sigma_obs       = pm.HalfNormal("sigma_obs", sigma=5)
        pm.Normal(
            "bed_obs",
            mu=LATENT_BASELINE + latent_pressure * LATENT_SCALE,
            sigma=sigma_obs,
            observed=beds,
        )

        if fast:
            print("Fast mode: using deterministic ADVI approximation.")
            approx = pm.fit(n=advi_steps, method="advi", progressbar=True, random_seed=random_seed)
            idata  = approx.sample(draws=draws, return_inferencedata=True, random_seed=random_seed)
            inference_method = "advi-fast"
        else:
            idata, inference_method = _nuts_with_fallback(
                draws=draws, tune=tune, advi_steps=advi_steps, random_seed=random_seed
            )

    idata.attrs = dict(idata.attrs or {})
    idata.attrs.update({
        "inference_method":    str(inference_method),
        "inference_fast_mode": int(bool(fast)),
        "inference_seed":      int(random_seed),
        "inference_advi_steps": int(advi_steps),
    })
    return model, idata


def _nuts_with_fallback(
    *,
    draws: int,
    tune: int,
    advi_steps: int,
    random_seed: int,
) -> tuple[az.InferenceData, str]:
    """Try NUTS sampling with progressive fallback to ADVI."""
    attempts = [
        {"target_accept": 0.90, "init": "adapt_diag"},
        {"target_accept": 0.95, "init": "adapt_diag"},
    ]
    last_error: Exception | None = None
    idata: az.InferenceData | None = None

    for idx, cfg in enumerate(attempts, start=1):
        try:
            print(f"NUTS attempt {idx}/{len(attempts)} (target_accept={cfg['target_accept']})")
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
            print("[WARN] NUTS attempt failed; trying next fallback path...")

    print("[WARN] NUTS retries exhausted; falling back to deterministic ADVI.")
    if last_error is not None:
        print(f"Last NUTS error: {last_error}")
    approx = pm.fit(n=max(advi_steps, 30_000), method="advi", progressbar=True, random_seed=random_seed)
    idata  = approx.sample(draws=draws, return_inferencedata=True, random_seed=random_seed)
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
    Persist InferenceData to NetCDF with lock-safe retry, write a JSON metadata
    sidecar, and warm the downstream cache.

    Returns the path the posterior was actually written to (may differ from
    output_path if a locked-file fallback was triggered).
    """
    idata.attrs = dict(idata.attrs or {})
    idata.attrs.update({
        "n_obs":  len(df),
        "n_icbs": int(df["icb"].nunique()),
        "icbs":   list(df["icb"].unique()),
    })

    actual_path = _save_netcdf_with_retry(idata, output_path)
    _write_metadata_sidecar(df, actual_path)
    _warm_cache(actual_path)

    return actual_path


def _save_netcdf_with_retry(idata: az.InferenceData, output_path: Path) -> Path:
    """Write NetCDF; on file-lock error fall back to a timestamped artifact."""
    lock_error: OSError | None = None

    for attempt in range(1, 4):
        try:
            idata.to_netcdf(str(output_path))
            print(f"[OK] Posterior saved to {output_path}")
            return output_path
        except OSError as exc:
            if "unable to lock file" not in str(exc).lower():
                raise
            lock_error = exc
            print(f"[WARN] File lock on {output_path} (attempt {attempt}/3). Retrying...")
            time.sleep(attempt)

    artifacts_dir = Path(".artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = artifacts_dir / f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.nc"
    idata.to_netcdf(str(fallback_path))
    print(f"[WARN] Locked file; saved to fallback: {fallback_path}")
    return fallback_path


def _write_metadata_sidecar(df: pd.DataFrame, posterior_path: Path) -> None:
    icb_names     = df["icb"].astype("category").cat.categories.tolist()
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
