"""
CLI entry point — offline Bayesian inference.

Parses arguments and delegates to core.inference.

Usage:
    epiforcasts-inference [--data-path FILE] [--output-path FILE] [--fast | --full]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import pandas as pd

from epiforcasts_nhs.config import (
    DEFAULT_ADVI_STEPS,
    DEFAULT_RANDOM_SEED,
    POSTERIORS_PATH,
    WEEKLY_CSV,
)
from epiforcasts_nhs.core.inference import fit_pressure_model, save_posterior_summaries


@click.command(name="inference")
@click.option("--data-path", type=click.Path(path_type=Path), default=WEEKLY_CSV,
              show_default=True, help="Input weekly panel CSV.")
@click.option("--output-path", type=click.Path(path_type=Path), default=POSTERIORS_PATH,
              show_default=True, help="Destination for posterior summaries.")
@click.option("--fast/--full", "fast", default=True, show_default=True,
              help="--fast: ADVI fast path.  --full: NUTS with retries / ADVI fallback.")
@click.option("--seed", type=int, default=DEFAULT_RANDOM_SEED, show_default=True)
@click.option("--advi-steps", type=int,
              default=lambda: int(os.environ.get("PRESSURE_MODEL_ADVI_STEPS", str(DEFAULT_ADVI_STEPS))),
              help=f"ADVI iterations (default: ${{PRESSURE_MODEL_ADVI_STEPS}} or {DEFAULT_ADVI_STEPS}).")
def main(data_path: Path, output_path: Path, fast: bool, seed: int, advi_steps: int) -> None:
    """Run offline Bayesian inference on the pressure model."""
    if not data_path.exists():
        click.echo(f"[FAIL] Data file not found: {data_path}", err=True)
        sys.exit(1)

    click.echo(f"Loading data from {data_path}…")
    df = pd.read_csv(data_path)
    click.echo(f"[OK] Loaded {len(df)} rows, {df['icb'].nunique()} ICBs")

    click.echo("Fitting Bayesian model…")
    _, idata, _ = fit_pressure_model(
        df,
        fast=fast,
        random_seed=seed,
        advi_steps=advi_steps,
    )
    click.echo(f"[OK] Posterior has {idata.posterior.dims['draw']} draws")

    click.echo(f"Saving to {output_path}…")
    save_posterior_summaries(idata, df, output_path)
    click.echo("[OK] Done!")


if __name__ == "__main__":
    main()
