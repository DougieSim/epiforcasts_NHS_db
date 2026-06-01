"""
Offline Bayesian inference script.

Runs the pressure model on synthetic data and saves posterior summaries to disk.
Run this once (or on a schedule) to update posterior estimates; the Streamlit app
consumes the stored results.

Usage:
    epiforcasts-inference [--data-path synthetic_nhs_pressure.csv] [--output-path posteriors.nc] [--fast]

Environment:
    PRESSURE_MODEL_FAST=0  to use fuller sampling (1000 draws, 1000 tune) instead of defaults.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.core.utils import LATENT_BASELINE, LATENT_SCALE


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
    df : pd.DataFrame
        Data with columns: 'icb', 'bed_occupancy', etc.
    fast : bool
        If True, use reduced sampling (400 draws, 400 tune).
        If False, use fuller sampling (1000 draws, 1000 tune).

    Returns
    -------
    model : pm.Model
        Fitted PyMC model.
    idata : az.InferenceData
        ArviZ InferenceData object (posterior, diagnostics, etc.).
    """
    icb_codes = df["icb"].astype("category").cat.codes.values
    beds = df["bed_occupancy"].values

    draws = 400 if fast else 1000
    tune = 400 if fast else 1000

    print(
        f"Inference config: draws={draws}, tune={tune}, fast={fast}, "
        f"seed={random_seed}, advi_steps={advi_steps}"
    )

    with pm.Model() as model:
        mu_national = pm.Normal("mu_national", 0, 1)
        sigma_icb = pm.HalfNormal("sigma_icb", sigma=1)
        icb_offset = pm.Normal(
            "icb_offset",
            0,
            1,
            shape=len(np.unique(icb_codes)),
        )
        icb_effect = pm.Deterministic("icb_effect", icb_offset * sigma_icb)

        latent_pressure = mu_national + icb_effect[icb_codes]
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=5)

        pm.Normal(
            "bed_obs",
            mu=LATENT_BASELINE + latent_pressure * LATENT_SCALE,
            sigma=sigma_obs,
            observed=beds,
        )

        if fast:
            print("Fast mode: using deterministic ADVI approximation.")
            approx = pm.fit(
                n=advi_steps,
                method="advi",
                progressbar=True,
                random_seed=random_seed,
            )
            idata = approx.sample(
                draws=draws,
                return_inferencedata=True,
                random_seed=random_seed,
            )
            inference_method = "advi-fast"
        else:
            nuts_attempts = [
                {"target_accept": 0.90, "init": "adapt_diag"},
                {"target_accept": 0.95, "init": "adapt_diag"},
            ]
            last_error: Exception | None = None
            idata = None

            for idx, cfg in enumerate(nuts_attempts, start=1):
                try:
                    print(
                        f"NUTS attempt {idx}/{len(nuts_attempts)} "
                        f"(target_accept={cfg['target_accept']}, init={cfg['init']})"
                    )
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=1,
                        cores=1,
                        target_accept=cfg["target_accept"],
                        init=cfg["init"],
                        progressbar=True,
                        compute_convergence_checks=True,
                        random_seed=random_seed,
                    )
                    inference_method = f"nuts-{idx}"
                    break
                except ValueError as exc:
                    last_error = exc
                    if "Not enough samples to build a trace" not in str(exc):
                        raise
                    print(
                        "[WARN] NUTS attempt failed with trace-build error; "
                        "trying next fallback path..."
                    )

            if idata is None:
                print("[WARN] NUTS retries exhausted; falling back to deterministic ADVI.")
                approx = pm.fit(
                    n=max(advi_steps, 30_000),
                    method="advi",
                    progressbar=True,
                    random_seed=random_seed,
                )
                idata = approx.sample(
                    draws=draws,
                    return_inferencedata=True,
                    random_seed=random_seed,
                )
                inference_method = "advi-fallback"
                if last_error is not None:
                    print(f"Last NUTS error: {last_error}")

    idata.attrs = dict(idata.attrs or {})
    idata.attrs["inference_method"] = str(inference_method)
    idata.attrs["inference_fast_mode"] = int(bool(fast))
    idata.attrs["inference_seed"] = int(random_seed)
    idata.attrs["inference_advi_steps"] = int(advi_steps)

    return model, idata


def save_posterior_summaries(
    idata: az.InferenceData,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save posterior InferenceData and metadata to NetCDF.

    Parameters
    ----------
    idata : az.InferenceData
        Posterior samples from ArviZ.
    df : pd.DataFrame
        Original data (for metadata).
    output_path : Path
        Output NetCDF file path.
    """
    # Add metadata to InferenceData
    existing_attrs = dict(idata.attrs or {})
    existing_attrs.update({
        "n_obs": len(df),
        "n_icbs": int(df["icb"].nunique()),
        "icbs": list(df["icb"].unique()),
    })
    idata.attrs = existing_attrs

    # Save to NetCDF with lock recovery for Windows/readers holding the file.
    actual_output_path = output_path
    lock_error: OSError | None = None
    for attempt in range(1, 4):
        try:
            idata.to_netcdf(str(output_path))
            lock_error = None
            break
        except OSError as exc:
            if "unable to lock file" not in str(exc).lower():
                raise
            lock_error = exc
            print(
                f"[WARN] Output file lock detected on {output_path} "
                f"(attempt {attempt}/3). Retrying..."
            )
            time.sleep(attempt)

    if lock_error is not None:
        artifacts_dir = Path(".artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fallback_name = f"{output_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.nc"
        actual_output_path = artifacts_dir / fallback_name
        idata.to_netcdf(str(actual_output_path))
        print(
            "[WARN] Could not overwrite locked output file. "
            f"Saved to fallback path: {actual_output_path}"
        )
    else:
        print(f"[OK] Posterior saved to {actual_output_path}")

    # Also save ICB index mapping as JSON for UI reference
    icb_names = df["icb"].astype("category").cat.categories.tolist()
    metadata = {
        "icbs": icb_names,
        "n_icbs": len(icb_names),
        "n_obs": len(df),
    }
    metadata_path = actual_output_path.with_stem(actual_output_path.stem + "_metadata")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to {metadata_path}")

    # Warm cache for instant UI access
    print("Warming cache for instant UI access…")
    cache = CacheManager(posteriors_path=actual_output_path)
    if cache.warm_cache():
        print(cache.get_status_report())
    else:
        print("[WARN] Cache warming failed; UI will be slower on first access")


def main():
    parser = argparse.ArgumentParser(
        description="Run offline Bayesian inference on pressure model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("synthetic_nhs_pressure.csv"),
        help="Path to input CSV data.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("posteriors.nc"),
        help="Path to output NetCDF file.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fast",
        action="store_true",
        help="Use fast deterministic ADVI path (default).",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Use NUTS with retries/fallback for deeper inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling/fallback behavior.",
    )
    parser.add_argument(
        "--advi-steps",
        type=int,
        default=int(os.environ.get("PRESSURE_MODEL_ADVI_STEPS", "20000")),
        help="Number of ADVI optimization steps in fast/fallback paths.",
    )

    args = parser.parse_args()

    # Resolve fast flag
    fast = True if not args.full else False

    # Load data
    if not args.data_path.exists():
        print(f"[FAIL] Data file not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {args.data_path}…")
    df = pd.read_csv(args.data_path)
    print(f"[OK] Loaded {len(df)} rows, {df['icb'].nunique()} ICBs")

    # Fit model
    print("Fitting Bayesian model…")
    model, idata = fit_pressure_model(
        df,
        fast=fast,
        random_seed=args.seed,
        advi_steps=args.advi_steps,
    )
    print(f"[OK] Model fitted; posterior has {idata.posterior.dims['draw']} draws")

    # Save results
    print(f"Saving to {args.output_path}…")
    save_posterior_summaries(idata, df, Path(args.output_path))

    print("[OK] Done!")


if __name__ == "__main__":
    main()
