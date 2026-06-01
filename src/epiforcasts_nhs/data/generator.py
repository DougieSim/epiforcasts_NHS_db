"""
Initial synthetic NHS dataset generator.

Builds the full historical weekly panel (ICB × weeks) and patient-episode
table used to seed the model. Run once to create the starting CSV files.

All values are **fabricated** — not real individuals or operational returns.
Do not use for clinical decisions.

Usage:
    epiforcasts-seed
    python -m epiforcasts_nhs.data.generator
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from epiforcasts_nhs.config import PATIENT_CSV, WEEKLY_CSV
from epiforcasts_nhs.data.utils import (
    AGE_BANDS,
    ADMISSION_URGENCY,
    BED_OCC_INTERCEPT,
    BED_OCC_LP_SCALE,
    BED_OCC_SEA_SCALE,
    CARE_SETTINGS,
    COUNT_COLS,
    ICD_CHAPTER_BUCKETS,
    PATHWAYS,
    SEASONAL_AMPLITUDE,
    SEASONAL_PEAK_WEEK,
    SEXES,
    START_DATE,
    SYNTHETIC_DGP_VERSION,
    WEEKS_PER_YEAR,
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

N_WEEKS = 156   # ~3 years of weekly history per ICB

MAX_EPISODES_PER_ICB_WEEK = 45
BASE_EPISODES_PER_WEEK    = 4

# Real NHS England ICB names (public labels).
# `scale` is a unitless simulation parameter — not an official performance score.
ICBS: dict[str, float] = {
    "NHS Birmingham and Solihull ICB":        1.0,
    "NHS Greater Manchester ICB":             1.28,
    "NHS South East London ICB":              0.92,
    "NHS North East and North Cumbria ICB":   1.06,
    "NHS Devon ICB":                          0.98,
    "NHS Nottingham and Nottinghamshire ICB": 1.12,
}


# ─────────────────────────────────────────
# DGP parameters — bed occupancy
# ─────────────────────────────────────────
# All from data.utils: BED_OCC_INTERCEPT, BED_OCC_LP_SCALE, BED_OCC_SEA_SCALE

_BED_OCC_NOISE_SD = 3.0
_BED_OCC_CLIP_LO  = 70.0
_BED_OCC_CLIP_HI  = 100.0


# ─────────────────────────────────────────
# DGP parameters — indicators
# ─────────────────────────────────────────

# Resp 111 calls (negative binomial)
_RESP111_NB_SHAPE = 25

# DTOC patients (Poisson)
_DTOC_BASELINE    = 2.0
_DTOC_LP_COEFF    = 2.5

# A&E type 1 attendances
_AE_BASELINE      = 800
_AE_LP_COEFF      = 120
_AE_NOISE_SD      = 40

# ED 4-hour breach count (fraction of A&E)
_ED4HR_BASE_RATE  = 0.08
_ED4HR_LP_COEFF   = 0.04
_ED4HR_NOISE_SD   = 0.02

# Ambulance red calls
_AMB_BASELINE     = 40
_AMB_LP_COEFF     = 15
_AMB_NOISE_SD     = 8

# OOH primary care contacts
_OOH_BASELINE     = 1_200
_OOH_LP_COEFF     = 90
_OOH_NOISE_SD     = 60

# Elective admissions
_ELEC_ADM_BASELINE  = 350
_ELEC_ADM_LP_COEFF  = 25
_ELEC_ADM_NOISE_SD  = 30

# Elective cancellations
_ELEC_CAN_BASELINE  = 25
_ELEC_CAN_LP_COEFF  = 18
_ELEC_CAN_NOISE_SD  = 10

# Acute non-elective admissions
_ACUTE_BASELINE     = 520
_ACUTE_LP_COEFF     = 70
_ACUTE_NOISE_SD     = 35

# Total discharges (derived from admissions)
_DISCHARGE_ELEC_RATIO = 0.95   # proportion of elective admissions contributing to discharges
_DISCHARGE_NOISE_SD   = 25

# Staff absence rate (%)
_STAFF_ABS_BASELINE = 4.5
_STAFF_ABS_LP_COEFF = 1.8
_STAFF_ABS_NOISE_SD = 0.6
_STAFF_ABS_CLIP_LO  = 1.0
_STAFF_ABS_CLIP_HI  = 22.0

# Critical care occupancy (%)
_CC_OCC_BASELINE    = 72
_CC_OCC_LP_COEFF    = 6
_CC_OCC_NOISE_SD    = 3
_CC_OCC_CLIP_LO     = 45.0
_CC_OCC_CLIP_HI     = 100.0

# Mental health inpatient beds occupancy (%)
_MH_OCC_BASELINE    = 78
_MH_OCC_LP_COEFF    = 4
_MH_OCC_NOISE_SD    = 2.5
_MH_OCC_CLIP_LO     = 50.0
_MH_OCC_CLIP_HI     = 100.0

# Infection isolation beds occupied
_INF_ISO_BASELINE   = 15
_INF_ISO_LP_COEFF   = 5
_INF_ISO_NOISE_SD   = 4

# Delayed transfers >= 21 days
_DTOC21_BASELINE    = 35
_DTOC21_LP_COEFF    = 22
_DTOC21_NOISE_SD    = 12

# Social care package delays (new)
_SOC_CARE_BASELINE  = 28
_SOC_CARE_LP_COEFF  = 12
_SOC_CARE_NOISE_SD  = 9

# Community crisis team contacts
_CRISIS_BASELINE    = 210
_CRISIS_LP_COEFF    = 35
_CRISIS_NOISE_SD    = 25

# GP same-day booking rate (%) — inversely related to pressure
_GP_SAME_BASELINE   = 42
_GP_SAME_LP_COEFF   = 3      # subtracted (pressure reduces same-day availability)
_GP_SAME_NOISE_SD   = 2
_GP_SAME_CLIP_LO    = 15.0
_GP_SAME_CLIP_HI    = 85.0

# NHS 111 online assessments completed
_NHS111_BASELINE    = 1_800
_NHS111_LP_COEFF    = 140
_NHS111_NOISE_SD    = 90

# Mean LOS acute (days)
_MLOS_BASELINE      = 5.2
_MLOS_LP_COEFF      = 0.35
_MLOS_NOISE_SD      = 0.25
_MLOS_CLIP_LO       = 2.0
_MLOS_CLIP_HI       = 18.0

# Winter pressure index (demo composite)
_WPI_BASELINE       = 3.5
_WPI_LP_COEFF       = 0.9
_WPI_SEA_COEFF      = 2.0
_WPI_NOISE_SD       = 0.35
_WPI_CLIP_LO        = 0.0
_WPI_CLIP_HI        = 10.0


# ─────────────────────────────────────────
# DGP parameters — missingness rates
# ─────────────────────────────────────────

_MISS_RESP111_DENOM  = 50    # 1/50 of rows have missing resp_111_calls
_MISS_RESP111_MIN    = 8
_MISS_GPSAME_DENOM   = 80    # 1/80 of rows have missing gp_same_day_booking_rate_pct
_MISS_GPSAME_MIN     = 5
_MISS_ELECCAN_DENOM  = 100   # 1/100 of rows have missing elective_cancellations
_MISS_ELECCAN_MIN    = 4


# ─────────────────────────────────────────
# DGP parameters — patient episodes
# ─────────────────────────────────────────

_LOS_BY_PATHWAY: dict[str, float] = {
    "emergency": 4.2,
    "elective":  2.1,
    "maternity": 2.0,
    "other":     3.0,
}
_LOS_LP_COEFF        = 0.55
_LOS_EXP_SCALE       = 1.8
_LOS_NOISE_BASE_SD   = 0.35
_LOS_NOISE_LP_SCALE  = 0.15
_LOS_CLIP_LO         = 0.25
_LOS_CLIP_HI         = 45.0

_EPISODE_INTENSITY_LP_COEFF = 0.25
_EPISODE_INTENSITY_LP_SCALE = 6.0

_DISCHARGE_ALIVE_RATE  = 0.985
_READMIT_RATE_BASE     = 0.06
_READMIT_LP_COEFF      = 0.03
_COVID_RATE_BASE       = 0.02
_COVID_LP_COEFF        = 0.01


# ─────────────────────────────────────────
# ICB weekly frame
# ─────────────────────────────────────────

def build_weekly_icb_frame(
    icb: str,
    scale: float,
    rng: np.random.Generator,
    n_weeks: int,
    lp: np.ndarray | None = None,
    t_offset: int = 0,
) -> pd.DataFrame:
    """
    Generate n_weeks rows of synthetic weekly data for one ICB.

    Parameters
    ----------
    icb      : ICB name written into the 'icb' column.
    scale    : Baseline latent pressure level.
    rng      : NumPy Generator — caller controls state for reproducibility.
    n_weeks  : Number of rows to generate.
    lp       : Pre-computed latent pressure series of length n_weeks.
               If None a fresh random walk is drawn.
    t_offset : Absolute week index of the first generated row (default 0).
               Pass the actual week number when extending an existing panel
               so the seasonal calculation uses the correct calendar position.
    """
    if lp is None:
        lp = latent_series(rng, scale, n_weeks)

    t        = np.arange(t_offset, t_offset + n_weeks, dtype=float)
    seasonal = SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (t - SEASONAL_PEAK_WEEK) / WEEKS_PER_YEAR)

    resp_111 = rng.negative_binomial(_RESP111_NB_SHAPE, 1 / (1 + np.exp(lp + seasonal)))
    bed_occ  = np.clip(
        BED_OCC_INTERCEPT + lp * BED_OCC_LP_SCALE + seasonal * BED_OCC_SEA_SCALE
        + rng.normal(0, _BED_OCC_NOISE_SD, n_weeks),
        _BED_OCC_CLIP_LO, _BED_OCC_CLIP_HI,
    )
    dtoc = rng.poisson(np.maximum(_DTOC_BASELINE + lp * _DTOC_LP_COEFF, 0)).astype(float)

    ae_type1_attendances = round_counts(
        rng, _AE_BASELINE + lp * _AE_LP_COEFF + rng.normal(0, _AE_NOISE_SD, n_weeks)
    )
    ed_4hr_breach_count = round_counts(
        rng, np.maximum(0, ae_type1_attendances * (
            _ED4HR_BASE_RATE + _ED4HR_LP_COEFF * np.tanh(lp) + rng.normal(0, _ED4HR_NOISE_SD, n_weeks)
        ))
    )
    ambulance_red_calls      = round_counts(rng, _AMB_BASELINE  + lp * _AMB_LP_COEFF  + rng.normal(0, _AMB_NOISE_SD,  n_weeks))
    ooh_primary_care_contacts = round_counts(rng, _OOH_BASELINE + lp * _OOH_LP_COEFF  + rng.normal(0, _OOH_NOISE_SD,  n_weeks))
    elective_admissions      = round_counts(rng, _ELEC_ADM_BASELINE + lp * _ELEC_ADM_LP_COEFF + rng.normal(0, _ELEC_ADM_NOISE_SD, n_weeks))
    elective_cancellations   = round_counts(rng, np.maximum(0, _ELEC_CAN_BASELINE + lp * _ELEC_CAN_LP_COEFF + rng.normal(0, _ELEC_CAN_NOISE_SD, n_weeks)))
    acute_admissions_nonelective = round_counts(rng, _ACUTE_BASELINE + lp * _ACUTE_LP_COEFF + rng.normal(0, _ACUTE_NOISE_SD, n_weeks))
    total_discharges         = round_counts(rng,
        acute_admissions_nonelective + elective_admissions * _DISCHARGE_ELEC_RATIO
        + rng.normal(0, _DISCHARGE_NOISE_SD, n_weeks)
    )

    staff_absence_rate_pct               = np.clip(_STAFF_ABS_BASELINE + lp * _STAFF_ABS_LP_COEFF + rng.normal(0, _STAFF_ABS_NOISE_SD, n_weeks), _STAFF_ABS_CLIP_LO, _STAFF_ABS_CLIP_HI)
    critical_care_occupancy_pct          = np.clip(_CC_OCC_BASELINE    + lp * _CC_OCC_LP_COEFF    + rng.normal(0, _CC_OCC_NOISE_SD,    n_weeks), _CC_OCC_CLIP_LO,    _CC_OCC_CLIP_HI)
    mental_health_inpatient_beds_occ_pct = np.clip(_MH_OCC_BASELINE    + lp * _MH_OCC_LP_COEFF    + rng.normal(0, _MH_OCC_NOISE_SD,    n_weeks), _MH_OCC_CLIP_LO,    _MH_OCC_CLIP_HI)
    infection_isolation_beds_occupied    = round_counts(rng, _INF_ISO_BASELINE + np.maximum(0, lp) * _INF_ISO_LP_COEFF + rng.normal(0, _INF_ISO_NOISE_SD, n_weeks))
    delayed_transfers_ge_21_days         = round_counts(rng, _DTOC21_BASELINE   + lp * _DTOC21_LP_COEFF   + rng.normal(0, _DTOC21_NOISE_SD,   n_weeks))
    social_care_package_delays_new       = round_counts(rng, _SOC_CARE_BASELINE + lp * _SOC_CARE_LP_COEFF + rng.normal(0, _SOC_CARE_NOISE_SD, n_weeks))
    community_crisis_team_contacts       = round_counts(rng, _CRISIS_BASELINE   + lp * _CRISIS_LP_COEFF   + rng.normal(0, _CRISIS_NOISE_SD,   n_weeks))
    gp_same_day_booking_rate_pct         = np.clip(_GP_SAME_BASELINE - lp * _GP_SAME_LP_COEFF + rng.normal(0, _GP_SAME_NOISE_SD, n_weeks), _GP_SAME_CLIP_LO, _GP_SAME_CLIP_HI)
    nhs_111_online_assessments_completed = round_counts(rng, _NHS111_BASELINE + lp * _NHS111_LP_COEFF + rng.normal(0, _NHS111_NOISE_SD, n_weeks))
    mean_los_acute_days                  = np.clip(_MLOS_BASELINE + lp * _MLOS_LP_COEFF + rng.normal(0, _MLOS_NOISE_SD, n_weeks), _MLOS_CLIP_LO, _MLOS_CLIP_HI)
    winter_pressure_index_demo           = np.clip(_WPI_BASELINE  + lp * _WPI_LP_COEFF + seasonal * _WPI_SEA_COEFF + rng.normal(0, _WPI_NOISE_SD, n_weeks), _WPI_CLIP_LO, _WPI_CLIP_HI)

    week_dates  = [START_DATE + timedelta(weeks=int(t_offset + w)) for w in range(n_weeks)]
    week_months = [d.month for d in week_dates]

    df = pd.DataFrame({
        "week":                                 np.arange(t_offset, t_offset + n_weeks, dtype=int),
        "week_date":                            [d.isoformat() for d in week_dates],
        "month":                                week_months,
        "icb":                                  icb,
        "synthetic_dgp_version":                SYNTHETIC_DGP_VERSION,
        "resp_111_calls":                       resp_111.astype(float),
        "bed_occupancy":                        bed_occ,
        "dtoc_patients":                        dtoc,
        "ae_type1_attendances":                 ae_type1_attendances.astype(float),
        "ed_4hr_breach_count":                  ed_4hr_breach_count.astype(float),
        "ambulance_category_red_calls":         ambulance_red_calls.astype(float),
        "elective_admissions":                  elective_admissions.astype(float),
        "elective_cancellations":               elective_cancellations.astype(float),
        "acute_admissions_nonelective":         acute_admissions_nonelective.astype(float),
        "total_discharges":                     total_discharges.astype(float),
        "staff_absence_rate_pct":               staff_absence_rate_pct,
        "critical_care_occupancy_pct":          critical_care_occupancy_pct,
        "mental_health_inpatient_beds_occ_pct": mental_health_inpatient_beds_occ_pct,
        "delayed_transfers_ge_21_days":         delayed_transfers_ge_21_days.astype(float),
        "ooh_primary_care_contacts":            ooh_primary_care_contacts.astype(float),
        "community_crisis_team_contacts":       community_crisis_team_contacts.astype(float),
        "gp_same_day_booking_rate_pct":         gp_same_day_booking_rate_pct,
        "nhs_111_online_assessments_completed": nhs_111_online_assessments_completed.astype(float),
        "social_care_package_delays_new":       social_care_package_delays_new.astype(float),
        "infection_isolation_beds_occupied":    infection_isolation_beds_occupied.astype(float),
        "mean_los_acute_days":                  mean_los_acute_days,
        "winter_pressure_index_demo":           winter_pressure_index_demo,
    })

    # Introduce realistic sporadic missingness — only for multi-week frames.
    # A single-row frame (used by SyntheticGenerator) would otherwise always be NaN.
    if n_weeks > 1:
        n_resp  = min(len(df), max(_MISS_RESP111_MIN,  len(df) // _MISS_RESP111_DENOM))
        n_gp    = min(len(df), max(_MISS_GPSAME_MIN,   len(df) // _MISS_GPSAME_DENOM))
        n_elec  = min(len(df), max(_MISS_ELECCAN_MIN,  len(df) // _MISS_ELECCAN_DENOM))
        df.loc[rng.choice(df.index, size=n_resp,  replace=False), "resp_111_calls"]              = np.nan
        df.loc[rng.choice(df.index, size=n_gp,    replace=False), "gp_same_day_booking_rate_pct"] = np.nan
        df.loc[rng.choice(df.index, size=n_elec,  replace=False), "elective_cancellations"]       = np.nan

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
        intensity = float(np.exp(_EPISODE_INTENSITY_LP_COEFF * lp[w]))
        n_eps = int(min(MAX_EPISODES_PER_ICB_WEEK,
                        rng.poisson(BASE_EPISODES_PER_WEEK + _EPISODE_INTENSITY_LP_SCALE * intensity)))
        if n_eps < 1:
            n_eps = 1

        w_age = np.array([0.17, 0.60, 0.23]) + np.array([0.0, -0.06, 0.06]) * float(np.tanh(lp[w]))
        w_age = w_age / w_age.sum()

        for s in range(n_eps):
            age_band     = rng.choice(AGE_BANDS, p=w_age)
            sex          = rng.choice(SEXES, p=[0.52, 0.46, 0.02])
            pathway      = rng.choice(PATHWAYS, p=[0.55, 0.25, 0.12, 0.08])
            care_setting = rng.choice(CARE_SETTINGS, p=[0.62, 0.12, 0.18, 0.08])

            if pathway == "elective":
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.05, 0.25, 0.65, 0.05])
            elif pathway == "emergency":
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.45, 0.35, 0.05, 0.15])
            else:
                admission_urgency = rng.choice(ADMISSION_URGENCY, p=[0.1, 0.3, 0.4, 0.2])

            los = float(np.clip(
                _LOS_BY_PATHWAY[pathway]
                + lp[w] * _LOS_LP_COEFF
                + rng.exponential(_LOS_EXP_SCALE)
                + rng.normal(0, _LOS_NOISE_BASE_SD + _LOS_NOISE_LP_SCALE * abs(lp[w])),
                _LOS_CLIP_LO, _LOS_CLIP_HI,
            ))
            acuity = int(np.clip(
                rng.integers(2, 6) + (1 if pathway == "emergency" and lp[w] > 0.5 else 0), 1, 5,
            ))
            discharged_alive = bool(rng.random() < _DISCHARGE_ALIVE_RATE)

            rows.append({
                "synthetic_episode_id":   episode_id(icb, w, s),
                "week_index":             w,
                "icb":                    icb,
                "age_band":               age_band,
                "sex":                    sex,
                "care_setting":           care_setting,
                "pathway":                pathway,
                "admission_urgency":      admission_urgency,
                "length_of_stay_days":    round(los, 2),
                "acuity_score":           acuity,
                "icd10_chapter_bucket":   rng.choice(ICD_CHAPTER_BUCKETS),
                "discharged_alive":       discharged_alive,
                "readmission_within_28d": bool(
                    discharged_alive
                    and rng.random() < (_READMIT_RATE_BASE + _READMIT_LP_COEFF * max(0, lp[w]))
                ),
                "covid_suspected_flag": int(
                    rng.random() < (_COVID_RATE_BASE + _COVID_LP_COEFF * max(0, lp[w]))
                ),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main() -> None:
    weekly_parts:  list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []

    for icb, scale in ICBS.items():
        rng = rng_for_icb(icb)
        lp  = latent_series(rng, float(scale), N_WEEKS)
        weekly_parts.append(build_weekly_icb_frame(icb, float(scale), rng, N_WEEKS, lp=lp))
        episode_parts.append(build_patient_episodes(icb, rng, lp, N_WEEKS))

    icb_weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly_all = pd.concat([icb_weekly, england_aggregate(icb_weekly)], ignore_index=True)
    weekly_all.to_csv(WEEKLY_CSV, index=False)

    episodes_all = pd.concat(episode_parts, ignore_index=True)
    episodes_all.to_csv(PATIENT_CSV, index=False)

    print(f"Wrote {WEEKLY_CSV} ({len(weekly_all)} rows, {len(weekly_all.columns)} columns)")
    print(f"Wrote {PATIENT_CSV} ({len(episodes_all)} episode rows)")
    print(f"Panel: {len(ICBS)} ICBs x {N_WEEKS} weeks + England aggregate")


if __name__ == "__main__":
    main()
