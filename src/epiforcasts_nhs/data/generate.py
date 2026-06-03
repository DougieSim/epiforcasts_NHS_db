"""
Ongoing synthetic data generation — appends new weeks at a configurable interval.

Extends the existing panel by estimating the latent pressure state from the
last observed bed-occupancy row, advancing it one random-walk step, then
delegating to build_weekly_icb_frame for the full set of correlated indicators.

This keeps the correlation structure between all variables consistent with
the initial dataset — every metric is driven by the same shared latent lp.

Usage (CLI):
    epiforcasts-generate              # append one new week
    epiforcasts-generate --weeks 4    # append four new weeks

Usage (library):
    from epiforcasts_nhs.data.generate import SyntheticGenerator

    gen = SyntheticGenerator("synthetic_nhs_pressure.csv")
    new_df = gen.generate_next_week(df)
    gen.append_and_save(new_df, df=df)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import epiforcasts_nhs.data.constants as constants
from epiforcasts_nhs.config import WEEKLY_CSV_ARCHIVE
from epiforcasts_nhs.data.generator import build_weekly_icb_frame
from epiforcasts_nhs.data.utils import england_aggregate

logger = logging.getLogger(__name__)


class SyntheticGenerator:
    """
    Appends new synthetic weeks by extending the existing latent pressure path.

    For each ICB, the latent pressure of the last observed week is estimated
    by inverting the bed-occupancy formula, one random-walk step is added,
    and build_weekly_icb_frame generates the full row of correlated indicators
    from that single lp value — preserving the same distributional structure
    as the initial dataset.
    """

    def __init__(
        self,
        csv_path: str | Path,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rng      = rng or np.random.default_rng()

    def generate_next_week(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Generate one new week for all ICBs and the England aggregate.

        Parameters
        ----------
        df : Full panel DataFrame.  If None, reads from self.csv_path.

        Returns
        -------
        pd.DataFrame with one row per ICB + England for the new week.
        """
        if df is None:
            df = pd.read_csv(self.csv_path)

        icb_rows = df[df["icb"] != "England"]
        new_week = int(icb_rows["week"].max()) + 1

        parts: list[pd.DataFrame] = []
        for icb, scale in constants.ICBS.items():
            last_row = icb_rows[icb_rows["icb"] == icb].sort_values("week").iloc[-1]
            lp_next  = self._next_lp(last_row, new_week)
            new_icb_df = build_weekly_icb_frame(
                icb, scale, self.rng,
                n_weeks=1,
                lp=np.array([lp_next]),
                t_offset=new_week,
            )
            parts.append(new_icb_df)

        new_icb_df  = pd.concat(parts, ignore_index=True)
        england_row = england_aggregate(new_icb_df)

        result = pd.concat([new_icb_df, england_row], ignore_index=True)
        logger.info(f"Generated week {new_week} for {len(constants.ICBS)} ICBs + England")
        return result

    def append_and_save(
        self,
        new_week_df: pd.DataFrame,
        df: pd.DataFrame | None = None,
        archive_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Append a new week, drop the earliest week from the rolling CSV, and
        append the new week to the full historical archive.

        The rolling CSV (self.csv_path) always stays the same length — one week
        is added and one is removed — so the model's n_weeks remains stable run
        to run.  The archive (archive_path or WEEKLY_CSV_ARCHIVE) accumulates
        every week ever generated and is never trimmed.

        Returns the updated rolling DataFrame (with the earliest week dropped).
        """
        existing = pd.read_csv(self.csv_path) if df is None else df.copy()
        max_week = int(existing["week"].max())
        new_week = int(new_week_df["week"].iloc[0])

        if new_week != max_week + 1:
            raise ValueError(
                f"Expected week {max_week + 1}, got {new_week}. "
                "Weeks must be strictly sequential."
            )

        # ── Archive: append new week to the full history ──────────────────────
        arc_path = Path(archive_path) if archive_path is not None else WEEKLY_CSV_ARCHIVE
        if arc_path.exists():
            archive = pd.read_csv(arc_path)
            archive = pd.concat([archive, new_week_df], ignore_index=True)
        else:
            # First run: seed the archive with everything we have
            archive = pd.concat([existing, new_week_df], ignore_index=True)
        archive.to_csv(arc_path, index=False)
        logger.info(f"Archive updated -> {arc_path} (weeks {archive['week'].min()}-{archive['week'].max()})")

        # ── Rolling window: append new week then drop the oldest ──────────────
        earliest_week = int(existing["week"].min())
        updated = pd.concat([existing, new_week_df], ignore_index=True)
        updated = updated[updated["week"] != earliest_week].reset_index(drop=True)
        updated.to_csv(self.csv_path, index=False)
        logger.info(
            f"Saved week {new_week} -> {self.csv_path} "
            f"(dropped week {earliest_week}, rolling window: "
            f"{updated['week'].min()}-{updated['week'].max()})"
        )
        return updated

    # ─────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────

    def _next_lp(self, last_row: pd.Series, new_week: int) -> float:
        """
        Estimate the latent pressure for the new week.

        Back-calculates lp from the last observed bed occupancy:
            lp ≈ (bed_occ − 84 − seasonal × 4) / 7

        then advances the random walk by one step:
            lp_new = lp_estimated + N(0, 0.12)
        """
        s            = constants.SEASONAL
        last_week    = int(last_row["week"])
        seasonal     = s.amplitude * np.cos(2 * np.pi * (last_week - s.peak_week) / s.weeks_per_year)
        lp_estimated = (last_row["bed_occupancy"] - constants.BED_OCC.intercept - seasonal * constants.BED_OCC.sea_scale) / constants.BED_OCC.lp_scale
        return float(lp_estimated + self.rng.normal(0, constants.LATENT_PROCESS.random_walk_sd))


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Append new synthetic week(s) to the panel CSV.")
    parser.add_argument("--data",  type=str, default="synthetic_nhs_pressure.csv")
    parser.add_argument("--weeks", type=int, default=1, help="Number of new weeks to generate.")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"[FAIL] Data file not found: {args.data}", file=sys.stderr)
        print("Run epiforcasts-seed first to create the initial dataset.", file=sys.stderr)
        return 1

    gen = SyntheticGenerator(args.data)
    df  = pd.read_csv(args.data)

    for i in range(args.weeks):
        new_week_df = gen.generate_next_week(df)
        df = gen.append_and_save(new_week_df, df=df)
        print(f"[OK] Generated week {int(df['week'].max())} ({i + 1}/{args.weeks})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
