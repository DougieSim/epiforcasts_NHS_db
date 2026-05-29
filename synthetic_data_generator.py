"""
Synthetic NHS weekly data generator.

Produces realistic week-on-week synthetic data for all ICBs using
per-ICB AR(1) dynamics fitted on the existing dataset. Each new week
is a plausible continuation of the observed time series.

Usage
-----
    from synthetic_data_generator import SyntheticGenerator

    gen = SyntheticGenerator("synthetic_nhs_pressure.csv")
    new_week_df = gen.generate_next_week()
    gen.append_and_save(new_week_df)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Must match generate_synthetic_data.py so week→date mapping is consistent.
_START_DATE = date(2023, 1, 2)

logger = logging.getLogger(__name__)

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

# Soft bounds: values are clipped to [min * BOUND_FACTOR, max * BOUND_FACTOR]
# to prevent drift outside plausible NHS ranges over many generated weeks.
BOUND_FACTOR = 1.25


class SyntheticGenerator:
    """
    Generates new synthetic weeks as AR(1) continuations of existing data.

    For each ICB and each variable, fits:
        x[t] = mu + rho * (x[t-1] - mu) + N(0, sigma)

    where mu is the per-ICB historical mean and rho/sigma are estimated
    from the per-ICB time series. This produces mean-reverting dynamics
    that stay within plausible ranges over long runs.
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
        """
        Fit per-ICB AR(1) parameters for every numeric column.
        Call this on the full dataset before generating new weeks.
        """
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
                mu     = float(series.mean())
                diffs  = series[1:] - series[:-1]
                sigma  = float(np.std(diffs)) if len(diffs) > 1 else 1.0

                # Fit rho via lagged regression
                if len(series) > 2:
                    x_lag  = (series[:-1] - mu)
                    x_next = (series[1:]  - mu)
                    denom  = float(np.dot(x_lag, x_lag))
                    rho    = float(np.dot(x_lag, x_next) / denom) if denom > 0 else 0.0
                    rho    = float(np.clip(rho, -0.98, 0.98))
                else:
                    rho = 0.5

                self._params[icb][col] = dict(
                    mu=mu, rho=rho, sigma=sigma, last=float(series[-1])
                )

                # Bounds from historical range with headroom
                self._bounds[icb][col] = dict(
                    lo=float(series.min()) / BOUND_FACTOR,
                    hi=float(series.max()) * BOUND_FACTOR,
                )

        logger.info(f"Fitted AR(1) params for {len(icbs)} ICBs × {len(NUMERIC_COLS)} variables")

    # ─────────────────────────────────────────
    # Generation
    # ─────────────────────────────────────────

    def generate_next_week(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Generate one new week for all ICBs.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Full dataset to fit on before generating. If None, uses
            the CSV at self.csv_path. Pass explicitly after appending
            to avoid re-reading from disk.

        Returns
        -------
        pd.DataFrame with one row per ICB for the new week.
        """
        if df is None:
            df = pd.read_csv(self.csv_path)

        if self._params is None:
            self.fit(df)

        icb_rows  = df[df["icb"] != "England"]
        last_week = int(icb_rows["week"].max())
        new_week  = last_week + 1

        # Get last observed row per ICB (re-seed from actual data)
        last_rows = (
            icb_rows.sort_values("week")
            .groupby("icb")
            .last()
            .reset_index()
        )

        new_week_date = _START_DATE + timedelta(weeks=new_week)
        new_month     = new_week_date.month

        rows = []
        for _, last_row in last_rows.iterrows():
            icb      = last_row["icb"]
            icb_p    = self._params[icb]
            icb_b    = self._bounds[icb]
            new_row  = {
                "week": new_week,
                "week_date": new_week_date.isoformat(),
                "month": new_month,
                "icb": icb,
            }

            for col in NUMERIC_COLS:
                if col not in icb_p:
                    new_row[col] = float(last_row.get(col, 0))
                    continue

                p     = icb_p[col]
                b     = icb_b[col]
                innov = self.rng.normal(0, p["sigma"])
                raw   = p["mu"] + p["rho"] * (float(last_row[col]) - p["mu"]) + innov

                # Soft clip to historical bounds
                new_row[col] = float(np.clip(raw, b["lo"], b["hi"]))

                # Update last value for next call
                p["last"] = new_row[col]

            # Carry forward non-numeric metadata columns
            for col in ["synthetic_dgp_version", "winter_pressure_index_demo"]:
                if col in last_row.index:
                    new_row[col] = last_row[col]

            rows.append(new_row)

        # Also update England aggregate (simple mean across ICBs)
        new_df  = pd.DataFrame(rows)
        england = {
            "week": new_week,
            "week_date": new_week_date.isoformat(),
            "month": new_month,
            "icb": "England",
        }
        for col in NUMERIC_COLS:
            if col in new_df.columns:
                england[col] = float(new_df[col].mean())
        rows.append(england)

        result = pd.DataFrame(rows)
        logger.info(f"Generated week {new_week} for {len(rows)-1} ICBs + England")
        return result

    # ─────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────

    def append_and_save(
        self,
        new_week_df: pd.DataFrame,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Append a new week to the CSV and return the updated DataFrame.

        Validates that the new week is exactly current_max + 1.
        """
        existing  = pd.read_csv(self.csv_path) if df is None else df.copy()
        max_week  = int(existing["week"].max())
        new_week  = int(new_week_df["week"].iloc[0])

        if new_week != max_week + 1:
            raise ValueError(
                f"Expected week {max_week + 1}, got {new_week}. "
                "Weeks must be strictly sequential."
            )

        updated = pd.concat([existing, new_week_df], ignore_index=True)
        updated.to_csv(self.csv_path, index=False)
        logger.info(
            f"Saved week {new_week} → {self.csv_path}  "
            f"(total weeks: {updated['week'].min()}–{updated['week'].max()})"
        )
        return updated

    def refit(self) -> None:
        """Re-fit AR(1) parameters from the latest CSV (call after appending)."""
        df = pd.read_csv(self.csv_path)
        self.fit(df)
        logger.info("Re-fitted parameters on updated dataset")
