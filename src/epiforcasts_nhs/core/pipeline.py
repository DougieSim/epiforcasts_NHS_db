"""
Inference daemon pipeline — generate → infer → cache orchestration.

Each public function is one named step in the pipeline. They are composed by
run_cycle() for scheduled / continuous operation and called individually by
cli.daemon for single-shot debugging modes.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.core.model import fit_pressure_model
from epiforcasts_nhs.data.generate import SyntheticGenerator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────

def step_generate(generator: SyntheticGenerator, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate one new synthetic week and append it to the CSV.

    Returns the updated DataFrame (does not re-read from disk).
    """
    logger.info("Generating new synthetic week...")
    generator.fit(df)
    new_week_df = generator.generate_next_week(df)
    new_week    = int(new_week_df["week"].iloc[0])
    n_icbs      = len(new_week_df[new_week_df["icb"] != "England"])
    logger.info(f"  Generated week {new_week} for {n_icbs} ICBs")

    updated = generator.append_and_save(new_week_df, df=df)
    logger.info(f"  Dataset now covers weeks {updated['week'].min()}–{updated['week'].max()}")
    return updated


def step_inference(df: pd.DataFrame, *, fast: bool) -> bool:
    """
    Run full AR(1) MCMC inference on df and save posteriors.

    Returns True on success, False on any exception.
    """
    logger.info("Running inference...")
    try:
        fit_pressure_model(df, fast=fast)
        logger.info("Inference complete.")
        return True
    except Exception:
        logger.exception("Inference failed")
        return False


def step_warm_cache(*, posterior_path: Path, cache_dir: Path) -> bool:
    """Warm the pre-computed summary-stats cache from saved posteriors."""
    logger.info("Warming cache...")
    cache   = CacheManager(posteriors_path=posterior_path, cache_dir=cache_dir)
    success = cache.warm_cache()
    if success:
        logger.info("Cache ready — dashboard will show updated pressures.")
    else:
        logger.error("Cache warming failed.")
    return success


# ─────────────────────────────────────────
# Full cycle
# ─────────────────────────────────────────

def run_cycle(
    generator: SyntheticGenerator,
    *,
    data_path: Path,
    posterior_path: Path,
    cache_dir: Path,
    fast: bool,
    generate: bool = True,
) -> bool:
    """
    Execute one full generate → infer → cache cycle.

    Parameters
    ----------
    generator      : SyntheticGenerator bound to data_path.
    data_path      : Path to the weekly panel CSV.
    posterior_path : Destination for the saved posteriors.
    cache_dir      : Pre-computed statistics cache directory.
    fast           : Use fewer MCMC draws (development mode).
    generate       : If False, skip data generation (inference-only mode).

    Returns True if all steps succeeded.
    """
    start = time.time()
    logger.info("─" * 60)
    logger.info(f"Starting cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows (weeks {df['week'].min()}–{df['week'].max()})")

    if generate:
        df = step_generate(generator, df)

    if not step_inference(df, fast=fast):
        logger.error("Cycle aborted — inference failed.")
        return False

    ok = step_warm_cache(posterior_path=posterior_path, cache_dir=cache_dir)
    logger.info(f"Cycle complete in {time.time() - start:.1f}s")
    logger.info("─" * 60)
    return ok
