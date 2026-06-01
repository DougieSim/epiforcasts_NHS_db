"""
CLI entry point — inference daemon.

Parses arguments and delegates to core.pipeline for all orchestration logic.

Usage:
    epiforcasts-daemon --once              # one generate → infer → cache cycle then exit
    epiforcasts-daemon --interval-hours 1  # run continuously
    epiforcasts-daemon --generate-only     # generate a new week, no inference
    epiforcasts-daemon --infer-only        # run inference on current data only
    epiforcasts-daemon --fast              # use fewer MCMC draws
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from epiforcasts_nhs.core.pipeline import run_cycle, step_generate, step_inference, step_warm_cache
from epiforcasts_nhs.data.generate import SyntheticGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_DATA      = Path("synthetic_nhs_pressure.csv")
_DEFAULT_POSTERIOR = Path("posteriors.nc")
_DEFAULT_CACHE     = Path(".cache")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NHS pressure model inference daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--once",          action="store_true", help="One cycle then exit.")
    parser.add_argument("--generate-only", action="store_true", help="Generate one week and exit.")
    parser.add_argument("--infer-only",    action="store_true", help="Infer on current data only.")
    parser.add_argument("--fast",          action="store_true", help="Fewer MCMC draws.")
    parser.add_argument("--interval-hours", type=float, default=1.0,
                        help="Hours between cycles when running continuously (default: 1).")
    parser.add_argument("--data",           type=Path, default=_DEFAULT_DATA,
                        help=f"Weekly panel CSV (default: {_DEFAULT_DATA}).")
    parser.add_argument("--posterior-path", type=Path, default=_DEFAULT_POSTERIOR,
                        help=f"Posterior output path (default: {_DEFAULT_POSTERIOR}).")
    parser.add_argument("--cache-dir",      type=Path, default=_DEFAULT_CACHE,
                        help=f"Cache directory (default: {_DEFAULT_CACHE}).")
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
        ok = run_cycle(
            generator,
            data_path=data_path,
            posterior_path=posterior_path,
            cache_dir=cache_dir,
            fast=args.fast,
        )
        return 0 if ok else 1

    # Continuous loop
    interval_s = args.interval_hours * 3600
    logger.info(f"Running continuously every {args.interval_hours}h.  Ctrl+C to stop.")

    while True:
        ok = run_cycle(
            generator,
            data_path=data_path,
            posterior_path=posterior_path,
            cache_dir=cache_dir,
            fast=args.fast,
        )
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
