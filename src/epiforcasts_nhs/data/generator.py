"""
Initial synthetic NHS dataset generator.

Builds the full historical weekly panel (ICB × weeks) and patient-episode
table used to seed the model. Run once to create the starting CSV files.

All values are **fabricated** — not real individuals or operational returns.
Do not use for clinical decisions. If you ever move toward real data: DPIA,
IG sign-off, and statistical disclosure control apply.

Usage:
    epiforcasts-seed
    python -m epiforcasts_nhs.data.generator
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from epiforcasts_nhs.data.utils import (
    AGE_BANDS,
    ADMISSION_URGENCY,
    CARE_SETTINGS,
    COUNT_COLS,
    ICD_CHAPTER_BUCKETS,
    PATHWAYS,
    SEXES,
    START_DATE,
    SYNTHETIC_DGP_VERSION,
    apply_measurement_noise,
    england_aggregate,
    episode_id,
    latent_series,
    round_counts,
    rng_for_icb,
)

# ─────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────

WEEKLY_CSV = "synthetic_nhs_pressure.csv"
PATIENT_CSV = "synthetic_patient_episodes.csv"

# ~3 years of weekly history per ICB
N_WEEKS = 156

MAX_EPISODES_PER_ICB_WEEK = 45
BASE_EPISODES_PER_WEEK = 4

# Real NHS England ICB names (public labels). `scale` is a unitless simulation
# parameter only — not an official performance score or funding weight.
ICBS: dict[str, float] = {
    "NHS Birmingham and Solihull ICB": 1.0,
    "NHS Greater Manchester ICB": 1.28,
    "NHS South East London ICB": 0.92,
    "NHS North East and North Cumbria ICB": 1.06,
    "NHS Devon ICB": 0.98,
    "NHS Nottingham and Nottinghamshire ICB": 1.12,
}


# ─────────────────────────────────────────
# ICB weekly frame
# ─────────────────────────────────────────

def build_weekly_icb_frame(
    icb: str,
    scale: float,
    rng: np.random.Generator,
    n_weeks: int,
    lp: np.ndarray | None = None,
) -> pd.DataFrame:
    if lp is None:
        lp = latent_series(rng, scale, n_weeks)
    t = np.arange(n_weeks, dtype=float)

    # Seasonal pattern: peaks in Jan, troughs in Jul.
    # Amplitude 0.15 on the latent scale → ~2.5% occupancy swing peak-to-trough.
    seasonal = 0.15 * np.cos(2 * np.pi * (t - 6) / 52.0)

    resp_111 = rng.negative_binomial(25, 1 / (1 + np.exp(lp + seasonal)))
    bed_occ = np.clip(84 + lp * 7 + seasonal * 4 + rng.normal(0, 3, n_weeks), 70, 100)
    dtoc = rng.poisson(np.maximum(2 + lp * 2.5, 0)).astype(float)

    ae_type1_attendances = round_counts(rng, 800 + lp * 120 + rng.normal(0, 40, n_weeks))
    ed_4hr_breach_count = round_counts(
        rng,
        np.maximum(0, ae_type1_attendances * (0.08 + 0.04 * np.tanh(lp) + rng.normal(0, 0.02, n_weeks))),
    )
    ambulance_red_calls = round_counts(rng, 40 + lp * 15 + rng.normal(0, 8, n_weeks))
    ooh_primary_care_contacts = round_counts(rng, 1200 + lp * 90 + rng.normal(0, 60, n_weeks))
    elective_admissions = round_counts(rng, 350 + lp * 25 + rng.normal(0, 30, n_weeks))
    elective_cancellations = round_counts(rng, np.maximum(0, 25 + lp * 18 + rng.normal(0, 10, n_weeks)))
    acute_admissions_nonelective = round_counts(rng, 520 + lp * 70 + rng.normal(0, 35, n_weeks))
    total_discharges = round_counts(
        rng, acute_admissions_nonelective + elective_admissions * 0.95 + rng.normal(0, 25, n_weeks),
    )
    staff_absence_rate_pct = np.clip(4.5 + lp * 1.8 + rng.normal(0, 0.6, n_weeks), 1.0, 22.0)
    critical_care_occupancy_pct = np.clip(72 + lp * 6 + rng.normal(0, 3, n_weeks), 45.0, 100.0)
    mental_health_inpatient_beds_occ_pct = np.clip(78 + lp * 4 + rng.normal(0, 2.5, n_weeks), 50.0, 100.0)
    infection_isolation_beds_occupied = round_counts(
        rng, 15 + np.maximum(0, lp) * 5 + rng.normal(0, 4, n_weeks),
    )
    delayed_transfers_ge_21_days = round_counts(rng, 35 + lp * 22 + rng.normal(0, 12, n_weeks))
    social_care_package_delays_new = round_counts(rng, 28 + lp * 12 + rng.normal(0, 9, n_weeks))
    community_crisis_team_contacts = round_counts(rng, 210 + lp * 35 + rng.normal(0, 25, n_weeks))
    gp_same_day_booking_rate_pct = np.clip(42 - lp * 3 + rng.normal(0, 2, n_weeks), 15.0, 85.0)
    nhs_111_online_assessments_completed = round_counts(rng, 1800 + lp * 140 + rng.normal(0, 90, n_weeks))
    mean_los_acute_days = np.clip(5.2 + lp * 0.35 + rng.normal(0, 0.25, n_weeks), 2.0, 18.0)
    winter_pressure_index_demo = np.clip(
        3.5 + lp * 0.9 + seasonal * 2 + rng.normal(0, 0.35, n_weeks), 0.0, 10.0,
    )

    week_dates = [START_DATE + timedelta(weeks=int(w)) for w in range(n_weeks)]
    week_months = [d.month for d in week_dates]

    df = pd.DataFrame({
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
    })

    n_miss = min(len(df), max(8, len(df) // 50))
    miss_idx = rng.choice(df.index, size=n_miss, replace=False)
    df.loc[miss_idx, "resp_111_calls"] = np.nan
    miss2 = rng.choice(df.index, size=min(len(df), max(5, len(df) // 80)), replace=False)
    df.loc[miss2, "gp_same_day_booking_rate_pct"] = np.nan
    miss3 = rng.choice(df.index, size=min(len(df), max(4, len(df) // 100)), replace=False)
    df.loc[miss3, "elective_cancellations"] = np.nan

    apply_measurement_noise(df, rng, count_columns=COUNT_COLS)
    return df


# ─────────────────────────────────────────
# Patient episodes
# ─────────────────────────────────────────

def build_patient_episodes(
    icb: str,
    rng: np.random.Generator,
    lp: np.ndarray,
    n_weeks: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for w in range(n_weeks):
        intensity = float(np.exp(0.25 * lp[w]))
        n_eps = int(min(MAX_EPISODES_PER_ICB_WEEK, rng.poisson(BASE_EPISODES_PER_WEEK + 6 * intensity)))
        if n_eps < 1:
            n_eps = 1

        w_age = np.array([0.17, 0.60, 0.23]) + np.array([0.0, -0.06, 0.06]) * float(np.tanh(lp[w]))
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
            length_of_stay_days = float(np.clip(
                los_base + lp[w] * 0.55 + rng.exponential(1.8) + rng.normal(0, 0.35 + 0.15 * abs(lp[w])),
                0.25, 45.0,
            ))
            acuity_score = int(np.clip(
                rng.integers(2, 6) + (1 if pathway == "emergency" and lp[w] > 0.5 else 0), 1, 5,
            ))
            icd_bucket = rng.choice(ICD_CHAPTER_BUCKETS)
            discharged_alive = bool(rng.random() < 0.985)
            readmit_28d = bool(discharged_alive and rng.random() < (0.06 + 0.03 * max(0, lp[w])))
            covid_suspected_flag = int(rng.random() < (0.02 + 0.01 * max(0, lp[w])))

            rows.append({
                "synthetic_episode_id": episode_id(icb, w, s),
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
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> None:
    weekly_parts: list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []

    for icb, scale in ICBS.items():
        rng = rng_for_icb(icb)
        lp = latent_series(rng, float(scale), N_WEEKS)
        weekly_parts.append(build_weekly_icb_frame(icb, float(scale), rng, N_WEEKS, lp=lp))
        episode_parts.append(build_patient_episodes(icb, rng, lp, N_WEEKS))

    icb_weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly_all = pd.concat([icb_weekly, england_aggregate(icb_weekly)], ignore_index=True)
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
