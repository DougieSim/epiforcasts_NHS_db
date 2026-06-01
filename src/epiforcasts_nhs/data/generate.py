"""
Ongoing synthetic data generation — appends new weeks at a configurable interval.

Uses per-ICB AR(1) dynamics fitted on the existing dataset so each new week is
a plausible continuation of the observed time series.

Usage (CLI):
    epiforcasts-generate              # generate one new week
    epiforcasts-generate --weeks 4    # generate four new weeks

Usage (library):
    from epiforcasts_nhs.data.generate import SyntheticGenerator

    gen = SyntheticGenerator("synthetic_nhs_pressure.csv")
    new_df = gen.generate_next_week()
    gen.append_and_save(new_df)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from epiforcasts_nhs.data.utils import START_DATE

logger = logging.getLogger(__name__)

# Columns produced by the generator for every new week
NUMERIC_COLS = [
    "bed_occupancy",
    "dtoc_patients",
    "ae_type1_attendances",
    "ed_4hr_breach_count",
    "ambulance_category_red_calls",
    "elective_admissions",
    "elective_cancellations",
    "acute_admissions_nonelective",
    "total_discharges",
    "staff_absence_rate_pct",
    "critical_care_occupancy_pct",
    "mental_health_inpatient_beds_occ_pct",
    "delayed_transfers_ge_21_days",
    "mean_los_acute_days",
    "resp_111_calls",
    "ooh_primary_care_contacts",
    "community_crisis_team_contacts",
    "gp_same_day_booking_rate_pct",
    "nhs_111_online_assessments_completed",
    "social_care_package_delays_new",
    "infection_isolation_beds_occupied",
]

# Values clipped to [historical_min / BOUND_FACTOR, historical_max * BOUND_FACTOR]
# to prevent drift outside plausible NHS ranges over many generated weeks.
BOUND_FACTOR = 1.25


class SyntheticGenerator:
    """
    Appends new synthetic weeks as AR(1) continuations of existing panel data.

    For each ICB and variable, fits:
        x[t] = mu + rho * (x[t-1] - mu) + N(0, sigma)

    where mu is the per-ICB historical mean and rho/sigma are estimated from
    the per-ICB time series, producing mean-reverting dynamics that stay within
    plausible ranges over long runs.
    """

    def __init__(
        self,
        csv_path: str | Path,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rng = rng or np.random.default_rng()
        self._params: dict | None = None
        self._bounds: dict | None = None

    # ─────────────────────────────────────────
    # Fitting
    # ─────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """Fit per-ICB AR(1) parameters on the full dataset."""
        icbs = df[df["icb"] != "England"]["icb"].unique()
        self._params = {}
        self._bounds = {}

        for icb in icbs:
            icb_df = df[df["icb"] == icb].sort_values("week")
            self._params[icb] = {}
            self._bounds[icb] = {}

            for col in NUMERIC_COLS:
                if col not in icb_df.columns:
                    continue

                series = icb_df[col].values.astype(float)
                mu = float(series.mean())
                diffs = series[1:] - series[:-1]
                sigma = float(np.std(diffs)) if len(diffs) > 1 else 1.0

                if len(series) > 2:
                    x_lag = series[:-1] - mu
                    x_next = series[1:] - mu
                    denom = float(np.dot(x_lag, x_lag))
                    rho = float(np.clip(np.dot(x_lag, x_next) / denom if denom > 0 else 0.0, -0.98, 0.98))
                else:
                    rho = 0.5

                self._params[icb][col] = dict(mu=mu, rho=rho, sigma=sigma, last=float(series[-1]))
                self._bounds[icb][col] = dict(
                    lo=float(series.min()) / BOUND_FACTOR,
                    hi=float(series.max()) * BOUND_FACTOR,
                )

        logger.info(f"Fitted AR(1) params for {len(icbs)} ICBs x {len(NUMERIC_COLS)} variables")

    # ─────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────

    def generate_next_week(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Generate one new week for all ICBs.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Full dataset. If None, reads from self.csv_path.

        Returns
        -------
        pd.DataFrame with one row per ICB + England aggregate for the new week.
        """
        if df is None:
            df = pd.read_csv(self.csv_path)
        if self._params is None:
            self.fit(df)

        icb_rows = df[df["icb"] != "England"]
        new_week = int(icb_rows["week"].max()) + 1
        new_week_date = START_DATE + timedelta(weeks=new_week)
        new_month = new_week_date.month

        last_rows = icb_rows.sort_values("week").groupby("icb").last().reset_index()

        rows = []
        for _, last_row in last_rows.iterrows():
            icb = last_row["icb"]
            icb_p = self._params[icb]
            icb_b = self._bounds[icb]
            new_row: dict = {
                "week": new_week,
                "week_date": new_week_date.isoformat(),
                "month": new_month,
                "icb": icb,
            }

            for col in NUMERIC_COLS:
                if col not in icb_p:
                    new_row[col] = float(last_row.get(col, 0))
                    continue
                p = icb_p[col]
                b = icb_b[col]
                raw = p["mu"] + p["rho"] * (float(last_row[col]) - p["mu"]) + self.rng.normal(0, p["sigma"])
                new_row[col] = float(np.clip(raw, b["lo"], b["hi"]))
                p["last"] = new_row[col]

            for col in ["synthetic_dgp_version", "winter_pressure_index_demo"]:
                if col in last_row.index:
                    new_row[col] = last_row[col]

            rows.append(new_row)

        new_df = pd.DataFrame(rows)
        england: dict = {"week": new_week, "week_date": new_week_date.isoformat(), "month": new_month, "icb": "England"}
        for col in NUMERIC_COLS:
            if col in new_df.columns:
                england[col] = float(new_df[col].mean())
        rows.append(england)

        result = pd.DataFrame(rows)
        logger.info(f"Generated week {new_week} for {len(rows) - 1} ICBs + England")
        return result

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def append_and_save(
        self,
        new_week_df: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Append a new week to the CSV and return the updated DataFrame."""
        existing = pd.read_csv(self.csv_path) if df is None else df.copy()
        max_week = int(existing["week"].max())
        new_week = int(new_week_df["week"].iloc[0])

        if new_week != max_week + 1:
            raise ValueError(
                f"Expected week {max_week + 1}, got {new_week}. Weeks must be strictly sequential."
            )

        updated = pd.concat([existing, new_week_df], ignore_index=True)
        updated.to_csv(self.csv_path, index=False)
        logger.info(
            f"Saved week {new_week} -> {self.csv_path} "
            f"(total weeks: {updated['week'].min()}-{updated['week'].max()})"
        )
        return updated

    def refit(self) -> None:
        """Re-fit AR(1) parameters from the latest CSV."""
        self.fit(pd.read_csv(self.csv_path))
        logger.info("Re-fitted parameters on updated dataset")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Append new synthetic week(s) to the panel CSV.")
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic_nhs_pressure.csv",
        help="Path to weekly panel CSV.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=1,
        help="Number of new weeks to generate (default: 1).",
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"[FAIL] Data file not found: {args.data}", file=sys.stderr)
        print("Run epiforcasts-seed first to create the initial dataset.", file=sys.stderr)
        return 1

    gen = SyntheticGenerator(args.data)
    df = pd.read_csv(args.data)
    gen.fit(df)

    for i in range(args.weeks):
        new_week_df = gen.generate_next_week(df)
        df = gen.append_and_save(new_week_df, df=df)
        print(f"[OK] Generated week {int(df['week'].max())} ({i + 1}/{args.weeks})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
