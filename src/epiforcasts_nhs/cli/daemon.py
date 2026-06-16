"""
CLI entry point — inference daemon.

Parses arguments and delegates to core.pipeline.

Usage:
    epiforcasts-daemon --once
    epiforcasts-daemon --interval-hours 1
    epiforcasts-daemon --generate-only
    epiforcasts-daemon --infer-only
    epiforcasts-daemon --fast
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import click
import pandas as pd

from epiforcasts_nhs.config import CACHE_DIR, POSTERIORS_PATH, WEEKLY_CSV
from epiforcasts_nhs.core.pipeline import run_cycle, step_generate, step_inference, step_warm_cache
from epiforcasts_nhs.data.generate import SyntheticGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_SECONDS_PER_HOUR = 3600


@click.command(name="daemon")
@click.option("--once",          is_flag=True, help="Run a single cycle then exit.")
@click.option("--generate-only", is_flag=True, help="Only append a new data week, then exit.")
@click.option("--infer-only",    is_flag=True, help="Only run inference + warm cache, then exit.")
@click.option("--fast",          is_flag=True, help="Use the ADVI fast inference path.")
@click.option("--interval-hours", type=float, default=1.0, show_default=True,
              help="Continuous-mode interval between cycles.")
@click.option("--data",           "data_path",      type=click.Path(path_type=Path), default=WEEKLY_CSV, show_default=True)
@click.option("--posterior-path", type=click.Path(path_type=Path), default=POSTERIORS_PATH, show_default=True)
@click.option("--cache-dir",      type=click.Path(path_type=Path), default=CACHE_DIR, show_default=True)
def main(
    once: bool,
    generate_only: bool,
    infer_only: bool,
    fast: bool,
    interval_hours: float,
    data_path: Path,
    posterior_path: Path,
    cache_dir: Path,
) -> None:
    """NHS pressure model inference daemon.

    With no flags, runs continuously every --interval-hours. Use --once for a
    single cycle, or --generate-only / --infer-only for individual steps.
    """
    generator = SyntheticGenerator(data_path)

    if generate_only:
        df      = pd.read_csv(data_path)
        updated = step_generate(generator, df)
        logger.info(f"Generated week {updated['week'].max()}. Done.")
        return

    if infer_only:
        df = pd.read_csv(data_path)
        ok = step_inference(df, fast=fast)
        if ok:
            step_warm_cache(posterior_path=posterior_path, cache_dir=cache_dir)
        raise SystemExit(0 if ok else 1)

    if once:
        ok = run_cycle(generator, data_path=data_path, posterior_path=posterior_path,
                       cache_dir=cache_dir, fast=fast)
        raise SystemExit(0 if ok else 1)

    interval_s = interval_hours * _SECONDS_PER_HOUR
    logger.info(f"Running continuously every {interval_hours}h.  Ctrl+C to stop.")

    while True:
        ok = run_cycle(generator, data_path=data_path, posterior_path=posterior_path,
                       cache_dir=cache_dir, fast=fast)
        if not ok:
            logger.warning("Cycle failed — will retry next interval.")
        next_str = datetime.fromtimestamp(time.time() + interval_s).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Next cycle at {next_str}")
        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break


if __name__ == "__main__":
    main()
