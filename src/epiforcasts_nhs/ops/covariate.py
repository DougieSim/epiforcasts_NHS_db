"""
Covariate alignment checks — within-ICB correlations with bed occupancy.

Usage:
    epiforcasts-covariate
"""

from __future__ import annotations

import sys

import click
import pandas as pd

from epiforcasts_nhs.config import COVARIATE_CHECK_HOLDOUT_WEEKS, COVARIATE_CHECK_LAGS, WEEKLY_CSV

COLS = [
    "ae_type1_attendances",
    "staff_absence_rate_pct",
    "delayed_transfers_ge_21_days",
    "acute_admissions_nonelective",
    "total_discharges",
    "ed_4hr_breach_count",
    "mean_los_acute_days",
    "dtoc_patients",
]


@click.command(name="covariate")
def main() -> None:
    """Covariate alignment checks — within-ICB correlations with bed occupancy."""
    if not WEEKLY_CSV.exists():
        print(f"[FAIL] Missing data file: {WEEKLY_CSV}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(WEEKLY_CSV)
    df = df[df["icb"] != "England"].copy()
    cutoff = df["week"].max() - COVARIATE_CHECK_HOLDOUT_WEEKS + 1
    train  = df[df["week"] < cutoff].copy()

    cols = [c for c in COLS if c in train.columns]

    within = train.copy()
    for col in cols + ["bed_occupancy"]:
        within[col] = within[col] - within.groupby("icb")[col].transform("mean")

    print("Within-ICB correlations with bed_occupancy:")
    print(within[cols + ["bed_occupancy"]].corr()["bed_occupancy"].drop("bed_occupancy")
          .sort_values(ascending=False).round(3))

    print("\nCovariate inter-correlations:")
    print(within[cols].corr().round(2))

    print("\nLagged correlations (covariate at t predicting occupancy at t+lag):")
    for lag in COVARIATE_CHECK_LAGS:
        lagged = train.copy()
        lagged["bed_occ_future"] = lagged.groupby("icb")["bed_occupancy"].shift(-lag)
        lagged = lagged.dropna(subset=["bed_occ_future"])
        for col in cols:
            lagged[col] = lagged[col] - lagged.groupby("icb")[col].transform("mean")
        lagged["bed_occ_future"] = (
            lagged["bed_occ_future"] - lagged.groupby("icb")["bed_occ_future"].transform("mean")
        )
        corrs = lagged[cols + ["bed_occ_future"]].corr()["bed_occ_future"].drop("bed_occ_future")
        print(f"\n  lag={lag} weeks:")
        print(corrs.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
