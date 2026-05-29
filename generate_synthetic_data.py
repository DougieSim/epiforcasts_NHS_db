"""
Synthetic NHS-style weekly aggregates + patient-episode rows for **demo / Discovery only**.

All values are **fabricated** — not real individuals or operational returns. Do not use for
clinical decisions. If you ever move toward real data: DPIA, IG sign-off, and statistical
disclosure control apply.

**Design:** One standalone synthetic world — **no Monte Carlo replication**. ICB **names**
match real **NHS England Integrated Care Board** labels (for recognisability); **all numeric
values remain simulated** and are not official statistics. Each listed ICB has its own RNG
stream and latent path (mutually independent across ICBs).

Improvement suggestions (next iterations)
----------------------------------------
- **Definitions**: Map columns to real NHS / Opendata metrics (units, numerators, denominators)
  so models trained here port to evaluation against official series.
- **Correlation structure**: Replace independent noise with a copula or fitted covariance from
  published aggregate tables so multivariate models are stress-tested realistically.
- **Calibration**: Fit the latent DGP to public winter pressure indices (or de-identified
  historical aggregates) instead of hand-tuned scales.
- **Episode grain**: If blending with real events, apply k-anonymity / small-number suppression
  and avoid free-text; keep episode IDs non-linkable.
- **Versioning**: Record `SYNTHETIC_DGP_VERSION` in CSV metadata or a sidecar JSON for audit.
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
import numpy as np
import pandas as pd

# --- Reproducibility & scale -------------------------------------------------
SYNTHETIC_DGP_VERSION = "2026.04.8"

MASTER_SEED = 2026

# Week 0 maps to the first Monday of 2023. Each subsequent week advances
# by 7 days so dates are stable across regenerations of the dataset.
START_DATE = date(2023, 1, 2)

WEEKLY_CSV = "synthetic_nhs_pressure.csv"
PATIENT_CSV = "synthetic_patient_episodes.csv"

# Weekly history per ICB (raise for backtests; lower keeps Streamlit + CSV I/O snappier).
N_WEEKS = 156  # ~3 years

# Real NHS England ICB **names** (public organisational labels). `scale` is a **unitless**
# simulation parameter only — not an official performance score or funding weight.
ICBS: dict[str, float] = {
    "NHS Birmingham and Solihull ICB": 1.0,
    "NHS Greater Manchester ICB": 1.28,
    "NHS South East London ICB": 0.92,
    "NHS North East and North Cumbria ICB": 1.06,
    "NHS Devon ICB": 0.98,
    "NHS Nottingham and Nottinghamshire ICB": 1.12,
}

MEASUREMENT_NOISE_SCALE = 0.025

MAX_EPISODES_PER_ICB_WEEK = 45
BASE_EPISODES_PER_WEEK = 4

AGE_BANDS = ["0-17", "18-64", "65+"]
SEXES = ["F", "M", "Unknown"]
CARE_SETTINGS = ["acute_inpatient", "mental_health_inpatient", "community_crisis", "ed_only"]
PATHWAYS = ["emergency", "elective", "maternity", "other"]
ADMISSION_URGENCY = ["immediate", "urgent", "routine", "not_applicable"]
ICD_CHAPTER_BUCKETS = [
    "I", "II", "IX", "X", "XI", "XIV", "XVIII", "XIX", "XXI", "Other",
]

COUNT_COLS = frozenset(
    {
        "resp_111_calls",
        "dtoc_patients",
        "ae_type1_attendances",
        "ed_4hr_breach_count",
        "ambulance_category_red_calls",
        "elective_admissions",
        "elective_cancellations",
        "acute_admissions_nonelective",
        "total_discharges",
        "delayed_transfers_ge_21_days",
        "ooh_primary_care_contacts",
        "community_crisis_team_contacts",
        "nhs_111_online_assessments_completed",
        "social_care_package_delays_new",
        "infection_isolation_beds_occupied",
    }
)


def _rng_for_icb(icb: str) -> np.random.Generator:
    """Independent bit generator per synthetic area (stable across runs)."""
    digest = hashlib.sha256(f"{SYNTHETIC_DGP_VERSION}|{icb}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") ^ MASTER_SEED
    return np.random.default_rng(seed)


def _latent_series(rng: np.random.Generator, scale: float, n_weeks: int) -> np.ndarray:
    """Random-walk latent "pressure" shared across indicators for one ICB."""
    return np.cumsum(rng.normal(0, 0.12, n_weeks)) + scale


def _n(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.rint(x)).astype(int)


def _apply_measurement_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    count_columns: frozenset[str],
) -> None:
    skip = {"week", "week_date", "month", "icb", "synthetic_dgp_version"}
    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        mask = s.notna()
        if col in count_columns:
            sigma = MEASUREMENT_NOISE_SCALE * (np.abs(s[mask].to_numpy()) ** 0.5 + 2.0)
            jitter = rng.normal(0.0, sigma, size=mask.sum())
            df.loc[mask, col] = _n(rng, s[mask].to_numpy().astype(float) + jitter)
        else:
            base = s[mask].to_numpy(dtype=float)
            sigma = MEASUREMENT_NOISE_SCALE * (np.abs(base) + 1.0)
            df.loc[mask, col] = base + rng.normal(0.0, sigma, size=len(base))


def build_weekly_icb_frame(
    icb: str,
    scale: float,
    rng: np.random.Generator,
    n_weeks: int,
    lp: np.ndarray | None = None,
) -> pd.DataFrame:
    if lp is None:
        lp = _latent_series(rng, scale, n_weeks)
    t = np.arange(n_weeks, dtype=float)
    # Seasonal pattern: peaks in Jan, troughs in Jul — matches real NHS winter pressure.
    # Amplitude 0.15 on the latent scale → ~2.5% occupancy swing peak-to-trough,
    # which is realistic for a synthetic demo while remaining visible in the data.
    # Phase offset of 6 weeks (peak mid-Feb) gives all four seasons
    # a clearly distinct signal: Winter > Spring > Autumn > Summer.
    # This matches the real NHS pattern where peak pressure runs through Feb.
    seasonal = 0.15 * np.cos(2 * np.pi * (t - 6) / 52.0)

    resp_111 = rng.negative_binomial(25, 1 / (1 + np.exp(lp + seasonal)))
    # Seasonal contribution to bed occupancy: positive in winter (cos peaks Jan),
    # negative in summer (cos troughs Jul). Scale of 4 means ~0.6% occupancy
    # swing per 0.15 unit of seasonal → ~1.8% peak-to-trough, realistic for demo.
    bed_occ = np.clip(
        84 + lp * 7 + seasonal * 4 + rng.normal(0, 3, n_weeks),
        70,
        100,
    )
    dtoc = rng.poisson(np.maximum(2 + lp * 2.5, 0)).astype(float)

    ae_type1_attendances = _n(rng, 800 + lp * 120 + rng.normal(0, 40, n_weeks))
    ed_4hr_breach_count = _n(
        rng,
        np.maximum(
            0,
            ae_type1_attendances * (0.08 + 0.04 * np.tanh(lp) + rng.normal(0, 0.02, n_weeks)),
        ),
    )
    ambulance_red_calls = _n(rng, 40 + lp * 15 + rng.normal(0, 8, n_weeks))
    ooh_primary_care_contacts = _n(rng, 1200 + lp * 90 + rng.normal(0, 60, n_weeks))

    elective_admissions = _n(rng, 350 + lp * 25 + rng.normal(0, 30, n_weeks))
    elective_cancellations = _n(rng, np.maximum(0, 25 + lp * 18 + rng.normal(0, 10, n_weeks)))
    acute_admissions_nonelective = _n(rng, 520 + lp * 70 + rng.normal(0, 35, n_weeks))
    total_discharges = _n(
        rng,
        acute_admissions_nonelective + elective_admissions * 0.95 + rng.normal(0, 25, n_weeks),
    )

    staff_absence_rate_pct = np.clip(
        4.5 + lp * 1.8 + rng.normal(0, 0.6, n_weeks),
        1.0,
        22.0,
    )
    critical_care_occupancy_pct = np.clip(
        72 + lp * 6 + rng.normal(0, 3, n_weeks),
        45.0,
        100.0,
    )
    mental_health_inpatient_beds_occ_pct = np.clip(
        78 + lp * 4 + rng.normal(0, 2.5, n_weeks),
        50.0,
        100.0,
    )
    infection_isolation_beds_occupied = _n(
        rng,
        15 + np.maximum(0, lp) * 5 + rng.normal(0, 4, n_weeks),
    )

    delayed_transfers_ge_21_days = _n(rng, 35 + lp * 22 + rng.normal(0, 12, n_weeks))
    social_care_package_delays_new = _n(rng, 28 + lp * 12 + rng.normal(0, 9, n_weeks))
    community_crisis_team_contacts = _n(rng, 210 + lp * 35 + rng.normal(0, 25, n_weeks))

    gp_same_day_booking_rate_pct = np.clip(
        42 - lp * 3 + rng.normal(0, 2, n_weeks),
        15.0,
        85.0,
    )
    nhs_111_online_assessments_completed = _n(rng, 1800 + lp * 140 + rng.normal(0, 90, n_weeks))

    mean_los_acute_days = np.clip(
        5.2 + lp * 0.35 + rng.normal(0, 0.25, n_weeks),
        2.0,
        18.0,
    )

    winter_pressure_index_demo = np.clip(
        3.5 + lp * 0.9 + seasonal * 2 + rng.normal(0, 0.35, n_weeks),
        0.0,
        10.0,
    )

    week_dates = [START_DATE + timedelta(weeks=int(w)) for w in range(n_weeks)]
    week_months = [d.month for d in week_dates]

    df = pd.DataFrame(
        {
            "week": np.arange(n_weeks, dtype=int),
            "week_date": [d.isoformat() for d in week_dates],
            "month": week_months,
            "icb": icb,
            "synthetic_dgp_version": SYNTHETIC_DGP_VERSION,
            "resp_111_calls": resp_111.astype(float),
            "bed_occupancy": bed_occ,
            "dtoc_patients": dtoc,
            "ae_type1_attendances": ae_type1_attendances.astype(float),
            "ed_4hr_breach_count": ed_4hr_breach_count.astype(float),
            "ambulance_category_red_calls": ambulance_red_calls.astype(float),
            "elective_admissions": elective_admissions.astype(float),
            "elective_cancellations": elective_cancellations.astype(float),
            "acute_admissions_nonelective": acute_admissions_nonelective.astype(float),
            "total_discharges": total_discharges.astype(float),
            "staff_absence_rate_pct": staff_absence_rate_pct,
            "critical_care_occupancy_pct": critical_care_occupancy_pct,
            "mental_health_inpatient_beds_occ_pct": mental_health_inpatient_beds_occ_pct,
            "delayed_transfers_ge_21_days": delayed_transfers_ge_21_days.astype(float),
            "ooh_primary_care_contacts": ooh_primary_care_contacts.astype(float),
            "community_crisis_team_contacts": community_crisis_team_contacts.astype(float),
            "gp_same_day_booking_rate_pct": gp_same_day_booking_rate_pct,
            "nhs_111_online_assessments_completed": nhs_111_online_assessments_completed.astype(float),
            "social_care_package_delays_new": social_care_package_delays_new.astype(float),
            "infection_isolation_beds_occupied": infection_isolation_beds_occupied.astype(float),
            "mean_los_acute_days": mean_los_acute_days,
            "winter_pressure_index_demo": winter_pressure_index_demo,
        }
    )

    n_miss = min(len(df), max(8, len(df) // 50))
    miss_idx = rng.choice(df.index, size=n_miss, replace=False)
    df.loc[miss_idx, "resp_111_calls"] = np.nan
    miss2 = rng.choice(df.index, size=min(len(df), max(5, len(df) // 80)), replace=False)
    df.loc[miss2, "gp_same_day_booking_rate_pct"] = np.nan
    miss3 = rng.choice(df.index, size=min(len(df), max(4, len(df) // 100)), replace=False)
    df.loc[miss3, "elective_cancellations"] = np.nan

    _apply_measurement_noise(df, rng, count_columns=COUNT_COLS)

    return df


def _episode_id(icb: str, week: int, seq: int) -> str:
    raw = f"{SYNTHETIC_DGP_VERSION}|{icb}|{week}|{seq}".encode()
    return "SYN-" + hashlib.sha256(raw).hexdigest()[:16]


def build_patient_episodes(
    icb: str,
    rng: np.random.Generator,
    lp: np.ndarray,
    n_weeks: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for w in range(n_weeks):
        intensity = float(np.exp(0.25 * lp[w]))
        n_eps = int(
            min(
                MAX_EPISODES_PER_ICB_WEEK,
                rng.poisson(BASE_EPISODES_PER_WEEK + 6 * intensity),
            )
        )
        if n_eps < 1:
            n_eps = 1

        w_age = np.array([0.17, 0.60, 0.23], dtype=float) + np.array(
            [0.0, -0.06, 0.06], dtype=float
        ) * float(np.tanh(lp[w]))
        w_age = w_age / w_age.sum()

        for s in range(n_eps):
            age_band = rng.choice(AGE_BANDS, p=w_age)
            sex = rng.choice(SEXES, p=[0.52, 0.46, 0.02])
            pathway = rng.choice(PATHWAYS, p=[0.55, 0.25, 0.12, 0.08])
            care_setting = rng.choice(CARE_SETTINGS, p=[0.62, 0.12, 0.18, 0.08])
            if pathway == "elective":
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.05, 0.25, 0.65, 0.05])
            elif pathway == "emergency":
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.45, 0.35, 0.05, 0.15])
            else:
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.1, 0.3, 0.4, 0.2])

            los_base = {"emergency": 4.2, "elective": 2.1, "maternity": 2.0, "other": 3.0}[pathway]
            length_of_stay_days = float(
                np.clip(
                    los_base
                    + lp[w] * 0.55
                    + rng.exponential(1.8)
                    + rng.normal(0, 0.35 + 0.15 * abs(lp[w])),
                    0.25,
                    45.0,
                )
            )
            acuity_score = int(
                np.clip(
                    rng.integers(2, 6)
                    + (1 if pathway == "emergency" and lp[w] > 0.5 else 0),
                    1,
                    5,
                )
            )
            icd_bucket = rng.choice(ICD_CHAPTER_BUCKETS)
            discharged_alive = bool(rng.random() < 0.985)
            readmit_28d = bool(
                discharged_alive and rng.random() < (0.06 + 0.03 * max(0, lp[w]))
            )
            covid_suspected_flag = int(rng.random() < (0.02 + 0.01 * max(0, lp[w])))

            rows.append(
                {
                    "synthetic_episode_id": _episode_id(icb, w, s),
                    "week_index": w,
                    "icb": icb,
                    "age_band": age_band,
                    "sex": sex,
                    "care_setting": care_setting,
                    "pathway": pathway,
                    "admission_urgency": admission_urgency,
                    "length_of_stay_days": round(length_of_stay_days, 2),
                    "acuity_score": acuity_score,
                    "icd10_chapter_bucket": icd_bucket,
                    "discharged_alive": discharged_alive,
                    "readmission_within_28d": readmit_28d,
                    "covid_suspected_flag": covid_suspected_flag,
                }
            )

    return pd.DataFrame(rows)


def _england_aggregate(icb_weekly: pd.DataFrame) -> pd.DataFrame:
    eng = icb_weekly.groupby("week", as_index=False).mean(numeric_only=True).assign(icb="England")
    eng["synthetic_dgp_version"] = SYNTHETIC_DGP_VERSION
    return eng


def main() -> None:
    weekly_parts: list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []

    for icb, scale in ICBS.items():
        rng_shared = _rng_for_icb(icb)
        lp = _latent_series(rng_shared, float(scale), N_WEEKS)
        weekly_parts.append(build_weekly_icb_frame(icb, float(scale), rng_shared, N_WEEKS, lp=lp))
        episode_parts.append(build_patient_episodes(icb, rng_shared, lp, N_WEEKS))

    icb_weekly = pd.concat(weekly_parts, ignore_index=True)
    england = _england_aggregate(icb_weekly)
    weekly_all = pd.concat([icb_weekly, england], ignore_index=True)
    weekly_all.to_csv(WEEKLY_CSV, index=False)

    episodes_all = pd.concat(episode_parts, ignore_index=True)
    episodes_all.to_csv(PATIENT_CSV, index=False)

    n_icb = len(ICBS)
    print(f"Wrote {WEEKLY_CSV} ({len(weekly_all)} rows, {len(weekly_all.columns)} columns)")
    print(f"Wrote {PATIENT_CSV} ({len(episodes_all)} episode rows)")
    print(
        f"Panel: {n_icb} named NHS ICBs x {N_WEEKS} weeks + England aggregate "
        f"(values synthetic; names for grounding only)"
    )


if __name__ == "__main__":
    main()