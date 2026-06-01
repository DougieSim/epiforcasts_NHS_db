"""
CLI entry point — offline Bayesian inference.

Parses arguments and delegates entirely to core.inference.
All model fitting and persistence logic lives there.

Usage:
    epiforcasts-inference [--data-path FILE] [--output-path FILE] [--fast | --full]
    epiforcasts-inference --seed 123 --advi-steps 5000

Environment:
    PRESSURE_MODEL_ADVI_STEPS  default ADVI step count (overridden by --advi-steps)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from epiforcasts_nhs.core.inference import fit_pressure_model, save_posterior_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline Bayesian inference on the pressure model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("synthetic_nhs_pressure.csv"),
        help="Path to input CSV (default: synthetic_nhs_pressure.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("posteriors.nc"),
        help="Path for output NetCDF file (default: posteriors.nc).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="ADVI fast path (default).")
    mode.add_argument("--full", action="store_true", help="NUTS with retries / ADVI fallback.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--advi-steps",
        type=int,
        default=int(os.environ.get("PRESSURE_MODEL_ADVI_STEPS", "20000")),
        help="ADVI optimisation steps.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        print(f"[FAIL] Data file not found: {args.data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {args.data_path}…")
    df = pd.read_csv(args.data_path)
    print(f"[OK] Loaded {len(df)} rows, {df['icb'].nunique()} ICBs")

    print("Fitting Bayesian model…")
    _, idata, _ = fit_pressure_model(
        df,
        fast=not args.full,
        random_seed=args.seed,
        advi_steps=args.advi_steps,
    )
    print(f"[OK] Posterior has {idata.posterior.dims['draw']} draws")

    print(f"Saving to {args.output_path}…")
    save_posterior_summaries(idata, df, args.output_path)
    print("[OK] Done!")


if __name__ == "__main__":
    main()
