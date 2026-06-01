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

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NHS pressure model inference daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--once",           action="store_true")
    parser.add_argument("--generate-only",  action="store_true")
    parser.add_argument("--infer-only",     action="store_true")
    parser.add_argument("--fast",           action="store_true")
    parser.add_argument("--interval-hours", type=float, default=1.0)
    parser.add_argument("--data",           type=Path, default=WEEKLY_CSV)
    parser.add_argument("--posterior-path", type=Path, default=POSTERIORS_PATH)
    parser.add_argument("--cache-dir",      type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    data_path      = args.data
    posterior_path = args.posterior_path
    cache_dir      = args.cache_dir
    generator      = SyntheticGenerator(data_path)

    if args.generate_only:
        df      = pd.read_csv(data_path)
        updated = step_generate(generator, df)
        logger.info(f"Generated week {updated['week'].max()}. Done.")
        return 0

    if args.infer_only:
        df = pd.read_csv(data_path)
        ok = step_inference(df, fast=args.fast)
        if ok:
            step_warm_cache(posterior_path=posterior_path, cache_dir=cache_dir)
        return 0 if ok else 1

    if args.once:
        ok = run_cycle(generator, data_path=data_path, posterior_path=posterior_path,
                       cache_dir=cache_dir, fast=args.fast)
        return 0 if ok else 1

    interval_s = args.interval_hours * _SECONDS_PER_HOUR
    logger.info(f"Running continuously every {args.interval_hours}h.  Ctrl+C to stop.")

    while True:
        ok = run_cycle(generator, data_path=data_path, posterior_path=posterior_path,
                       cache_dir=cache_dir, fast=args.fast)
        if not ok:
            logger.warning("Cycle failed — will retry next interval.")
        next_str = datetime.fromtimestamp(time.time() + interval_s).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Next cycle at {next_str}")
        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
