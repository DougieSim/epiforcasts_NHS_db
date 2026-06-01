"""
CLI entry point — offline Bayesian inference.

Parses arguments and delegates to core.inference.

Usage:
    epiforcasts-inference [--data-path FILE] [--output-path FILE] [--fast | --full]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from epiforcasts_nhs.config import (
    DEFAULT_ADVI_STEPS,
    DEFAULT_RANDOM_SEED,
    POSTERIORS_PATH,
    WEEKLY_CSV,
)
from epiforcasts_nhs.core.inference import fit_pressure_model, save_posterior_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline Bayesian inference on the pressure model."
    )
    parser.add_argument("--data-path",   type=Path, default=WEEKLY_CSV)
    parser.add_argument("--output-path", type=Path, default=POSTERIORS_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="ADVI fast path (default).")
    mode.add_argument("--full", action="store_true", help="NUTS with retries / ADVI fallback.")
    parser.add_argument("--seed",       type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--advi-steps",
        type=int,
        default=int(os.environ.get("PRESSURE_MODEL_ADVI_STEPS", str(DEFAULT_ADVI_STEPS))),
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
