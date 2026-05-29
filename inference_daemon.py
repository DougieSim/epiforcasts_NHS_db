"""
Offline inference daemon with automatic synthetic data generation.

Every interval:
  1. Generate a new synthetic week and append to the CSV
  2. Re-fit the model on the full updated dataset
  3. Save posteriors and warm the cache
  4. Dashboard reads fresh results automatically

Usage
-----
Run once (useful for initial setup or testing):
    python inference_daemon.py --once

Run on a schedule (production):
    python inference_daemon.py --interval-hours 1

Generate a week without running inference (debugging):
    python inference_daemon.py --generate-only

Fast sampling mode (development):
    python inference_daemon.py --interval-hours 1 --fast
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_pressure_model import fit_pressure_model
from cache_manager import CacheManager
from synthetic_data_generator import SyntheticGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PATH       = "synthetic_nhs_pressure.csv"
POSTERIORS_PATH = "posteriors.nc"
CACHE_DIR       = ".cache"


# ─────────────────────────────────────────
# Core pipeline steps
# ─────────────────────────────────────────

def step_generate(generator: SyntheticGenerator, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one new synthetic week and append it to the CSV.
    Returns the updated DataFrame.
    """
    logger.info("Generating new synthetic week...")
    generator.fit(df)
    new_week_df = generator.generate_next_week(df)
    new_week    = int(new_week_df["week"].iloc[0])
    logger.info(f"  Generated week {new_week} for "
                f"{len(new_week_df[new_week_df['icb'] != 'England'])} ICBs")

    updated = generator.append_and_save(new_week_df, df=df)
    logger.info(f"  Dataset now covers weeks "
                f"{updated['week'].min()}–{updated['week'].max()}")
    return updated


def step_inference(df: pd.DataFrame, fast: bool) -> bool:
    """
    Run MCMC inference on the full dataset and save posteriors.
    Returns True on success.
    """
    logger.info("Running inference...")
    try:
        model, idata, enc = fit_pressure_model(df, fast=fast)
        logger.info("Inference complete.")
        return True
    except Exception:
        logger.exception("Inference failed")
        return False


def step_warm_cache() -> bool:
    """Warm the summary stats cache from the saved posteriors."""
    logger.info("Warming cache...")
    cache = CacheManager(posteriors_path=POSTERIORS_PATH, cache_dir=CACHE_DIR)
    success = cache.warm_cache()
    if success:
        logger.info("Cache ready — dashboard will show updated pressures.")
    else:
        logger.error("Cache warming failed.")
    return success


# ─────────────────────────────────────────
# Full run
# ─────────────────────────────────────────

def run_cycle(
    generator: SyntheticGenerator,
    *,
    fast: bool,
    generate: bool = True,
) -> bool:
    """
    Execute one full generate → infer → cache cycle.

    Parameters
    ----------
    generator   : SyntheticGenerator bound to the data CSV
    fast        : use fewer MCMC draws (development mode)
    generate    : if False, skip data generation (inference only)

    Returns True if all steps succeeded.
    """
    cycle_start = time.time()
    logger.info("─" * 60)
    logger.info(f"Starting cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} rows "
                f"(weeks {df['week'].min()}–{df['week'].max()})")

    if generate:
        df = step_generate(generator, df)

    ok = step_inference(df, fast=fast)
    if not ok:
        logger.error("Cycle aborted — inference failed.")
        return False

    ok = step_warm_cache()
    elapsed = time.time() - cycle_start
    logger.info(f"Cycle complete in {elapsed:.1f}s")
    logger.info("─" * 60)
    return ok


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> int:
    global DATA_PATH

    parser = argparse.ArgumentParser(
        description="NHS pressure model inference daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Generate one week, run inference, warm cache, then exit.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate one new synthetic week and exit (no inference).",
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Run inference on current data without generating a new week.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=1.0,
        help="Hours between cycles when running continuously (default: 1).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer MCMC draws (development/testing only).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help=f"Path to weekly CSV (default: {DATA_PATH}).",
    )
    args = parser.parse_args()

    DATA_PATH = args.data

    generator = SyntheticGenerator(DATA_PATH)

    # ── Single-shot modes ─────────────────────────────────────────────

    if args.generate_only:
        df      = pd.read_csv(DATA_PATH)
        updated = step_generate(generator, df)
        logger.info(f"Generated week {updated['week'].max()}. Done.")
        return 0

    if args.infer_only:
        df = pd.read_csv(DATA_PATH)
        ok = step_inference(df, fast=args.fast)
        if ok:
            step_warm_cache()
        return 0 if ok else 1

    if args.once:
        ok = run_cycle(generator, fast=args.fast, generate=True)
        return 0 if ok else 1

    # ── Continuous loop ───────────────────────────────────────────────

    interval_s = args.interval_hours * 3600
    logger.info(
        f"Running continuously every {args.interval_hours}h. "
        "Ctrl+C to stop."
    )

    while True:
        ok = run_cycle(generator, fast=args.fast, generate=True)

        if not ok:
            logger.warning("Cycle failed — will retry next interval.")

        next_run = time.time() + interval_s
        next_str = datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Next cycle at {next_str}")

        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())