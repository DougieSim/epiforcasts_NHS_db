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

import click
import numpy as np
import pandas as pd

from epiforcasts_nhs.config import PATIENT_CSV, WEEKLY_CSV
import epiforcasts_nhs.data.constants as constants
from epiforcasts_nhs.data.constants import ClippedParams, CountParams
from epiforcasts_nhs.data.utils import (
    apply_measurement_noise,
    england_aggregate,
    episode_id,
    latent_series,
    round_counts,
    rng_for_icb,
)


# ─────────────────────────────────────────
# DGP helpers
# ─────────────────────────────────────────

def _count(rng: np.random.Generator, p: CountParams, lp: np.ndarray, n: int) -> np.ndarray:
    return round_counts(p.baseline + lp * p.lp_coeff + rng.normal(0, p.noise_sd, n))


def _clipped(rng: np.random.Generator, p: ClippedParams, lp: np.ndarray, n: int) -> np.ndarray:
    return np.clip(p.baseline + lp * p.lp_coeff + rng.normal(0, p.noise_sd, n), p.clip_lo, p.clip_hi)


# ─────────────────────────────────────────
# ICB weekly frame — private helpers
# ─────────────────────────────────────────

def _compute_indicators(
    rng: np.random.Generator,
    lp: np.ndarray,
    seasonal: np.ndarray,
    n: int,
) -> dict[str, np.ndarray]:
    c = constants

    ae       = _count(rng, c.AE, lp, n)
    elec_adm = _count(rng, c.ELEC_ADM, lp, n)
    acute    = _count(rng, c.ACUTE, lp, n)
    return {
        "resp_111_calls":                       rng.negative_binomial(c.RESP111_NB_SHAPE, 1 / (1 + np.exp(lp + seasonal))).astype(float),
        "bed_occupancy":                        c.BED_OCC.intercept + lp * c.BED_OCC.lp_scale + seasonal * c.BED_OCC.sea_scale + rng.normal(0, c.BED_OCC.noise_sd, n),
        "dtoc_patients":                        rng.poisson(np.maximum(c.DTOC.baseline + lp * c.DTOC.lp_coeff, 0)).astype(float),
        "ae_type1_attendances":                 ae.astype(float),
        "ed_4hr_breach_count":                  round_counts(np.maximum(0, ae * (c.ED4HR.base_rate + c.ED4HR.lp_coeff * np.tanh(lp) + rng.normal(0, c.ED4HR.noise_sd, n)))).astype(float),
        "ambulance_category_red_calls":         _count(rng, c.AMB, lp, n).astype(float),
        "elective_admissions":                  elec_adm.astype(float),
        "elective_cancellations":               _count(rng, c.ELEC_CAN, lp, n).astype(float),
        "acute_admissions_nonelective":         acute.astype(float),
        "total_discharges":                     round_counts(acute + elec_adm * c.DISCHARGE.elec_ratio + rng.normal(0, c.DISCHARGE.noise_sd, n)).astype(float),
        "staff_absence_rate_pct":               _clipped(rng, c.STAFF_ABS, lp, n),
        "critical_care_occupancy_pct":          _clipped(rng, c.CC_OCC, lp, n),
        "mental_health_inpatient_beds_occ_pct": _clipped(rng, c.MH_OCC, lp, n),
        "delayed_transfers_ge_21_days":         _count(rng, c.DTOC21, lp, n).astype(float),
        "ooh_primary_care_contacts":            _count(rng, c.OOH, lp, n).astype(float),
        "community_crisis_team_contacts":       _count(rng, c.CRISIS, lp, n).astype(float),
        "gp_same_day_booking_rate_pct":         np.clip(c.GP_SAME.baseline - lp * c.GP_SAME.lp_coeff + rng.normal(0, c.GP_SAME.noise_sd, n), c.GP_SAME.clip_lo, c.GP_SAME.clip_hi),
        "nhs_111_online_assessments_completed": _count(rng, c.NHS111, lp, n).astype(float),
        "social_care_package_delays_new":       _count(rng, c.SOC_CARE, lp, n).astype(float),
        "infection_isolation_beds_occupied":    round_counts(c.INF_ISO.baseline + np.maximum(0, lp) * c.INF_ISO.lp_coeff + rng.normal(0, c.INF_ISO.noise_sd, n)).astype(float),
        "mean_los_acute_days":                  _clipped(rng, c.MLOS, lp, n),
        "winter_pressure_index_demo":           np.clip(c.WPI.baseline + lp * c.WPI.lp_coeff + seasonal * c.WPI.sea_coeff + rng.normal(0, c.WPI.noise_sd, n), c.WPI.clip_lo, c.WPI.clip_hi),
    }


def _inject_missingness(df: pd.DataFrame, rng: np.random.Generator) -> None:
    m      = constants.MISSINGNESS
    n_resp = min(len(df), max(m.resp111_min,  len(df) // m.resp111_denom))
    n_gp   = min(len(df), max(m.gp_same_min,  len(df) // m.gp_same_denom))
    n_elec = min(len(df), max(m.elec_can_min, len(df) // m.elec_can_denom))
    df.loc[rng.choice(df.index, size=n_resp,  replace=False), "resp_111_calls"]               = np.nan
    df.loc[rng.choice(df.index, size=n_gp,    replace=False), "gp_same_day_booking_rate_pct"] = np.nan
    df.loc[rng.choice(df.index, size=n_elec,  replace=False), "elective_cancellations"]       = np.nan


# ─────────────────────────────────────────
# ICB weekly frame — public API
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

    s        = constants.SEASONAL
    t        = np.arange(t_offset, t_offset + n_weeks, dtype=float)
    seasonal = s.amplitude * np.cos(2 * np.pi * (t - s.peak_week) / s.weeks_per_year)

    week_dates = [constants.DATASET.start_date + timedelta(weeks=int(t_offset + w)) for w in range(n_weeks)]

    df = pd.DataFrame({
        "week":                  np.arange(t_offset, t_offset + n_weeks, dtype=int),
        "week_date":             [d.isoformat() for d in week_dates],
        "month":                 [d.month for d in week_dates],
        "icb":                   icb,
        "synthetic_dgp_version": constants.DATASET.dgp_version,
        **_compute_indicators(rng, lp, seasonal, n_weeks),
    })

    # Introduce realistic sporadic missingness — only for multi-week frames.
    # A single-row frame (used by SyntheticGenerator) would otherwise always be NaN.
    if n_weeks > 1:
        _inject_missingness(df, rng)

    apply_measurement_noise(df, rng, count_columns=constants.COUNT_COLS)
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
    ep   = constants.EPISODE
    los  = constants.LOS
    cats = constants.EPISODE_CATEGORICALS

    for w in range(n_weeks):
        intensity = float(np.exp(ep.intensity_lp_coeff * lp[w]))
        n_eps = int(min(constants.DATASET.max_episodes_per_icb_week,
                        rng.poisson(constants.DATASET.base_episodes_per_week + ep.intensity_lp_scale * intensity)))
        if n_eps < 1:
            n_eps = 1

        w_age = np.array([0.17, 0.60, 0.23]) + np.array([0.0, -0.06, 0.06]) * float(np.tanh(lp[w]))
        w_age = w_age / w_age.sum()

        for s in range(n_eps):
            age_band     = rng.choice(cats.age_bands, p=w_age)
            sex          = rng.choice(cats.sexes, p=[0.52, 0.46, 0.02])
            pathway      = rng.choice(cats.pathways, p=[0.55, 0.25, 0.12, 0.08])
            care_setting = rng.choice(cats.care_settings, p=[0.62, 0.12, 0.18, 0.08])

            if pathway == "elective":
                admission_urgency = rng.choice(cats.admission_urgency, p=[0.05, 0.25, 0.65, 0.05])
            elif pathway == "emergency":
                admission_urgency = rng.choice(cats.admission_urgency, p=[0.45, 0.35, 0.05, 0.15])
            else:
                admission_urgency = rng.choice(cats.admission_urgency, p=[0.1, 0.3, 0.4, 0.2])

            los_val = float(np.clip(
                los.by_pathway[pathway]
                + lp[w] * los.lp_coeff
                + rng.exponential(los.exp_scale)
                + rng.normal(0, los.noise_base_sd + los.noise_lp_scale * abs(lp[w])),
                los.clip_lo, los.clip_hi,
            ))
            acuity = int(np.clip(
                rng.integers(2, 6) + (1 if pathway == "emergency" and lp[w] > 0.5 else 0), 1, 5,
            ))
            discharged_alive = bool(rng.random() < ep.discharge_alive_rate)

            rows.append({
                "synthetic_episode_id":   episode_id(icb, w, s),
                "week_index":             w,
                "icb":                    icb,
                "age_band":               age_band,
                "sex":                    sex,
                "care_setting":           care_setting,
                "pathway":                pathway,
                "admission_urgency":      admission_urgency,
                "length_of_stay_days":    round(los_val, 2),
                "acuity_score":           acuity,
                "icd10_chapter_bucket":   rng.choice(cats.icd_chapter_buckets),
                "discharged_alive":       discharged_alive,
                "readmission_within_28d": bool(
                    discharged_alive
                    and rng.random() < (ep.readmit_rate_base + ep.readmit_lp_coeff * max(0, lp[w]))
                ),
                "covid_suspected_flag": int(
                    rng.random() < (ep.covid_rate_base + ep.covid_lp_coeff * max(0, lp[w]))
                ),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

@click.command(name="seed")
def main() -> None:
    """Build the initial synthetic weekly panel and patient-episode CSVs."""
    weekly_parts:  list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []

    for icb, scale in constants.ICBS.items():
        rng = rng_for_icb(icb)
        lp  = latent_series(rng, float(scale), constants.DATASET.n_weeks)
        weekly_parts.append(build_weekly_icb_frame(icb, float(scale), rng, constants.DATASET.n_weeks, lp=lp))
        episode_parts.append(build_patient_episodes(icb, rng, lp, constants.DATASET.n_weeks))

    icb_weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly_all = pd.concat([icb_weekly, england_aggregate(icb_weekly)], ignore_index=True)
    weekly_all.to_csv(WEEKLY_CSV, index=False)

    episodes_all = pd.concat(episode_parts, ignore_index=True)
    episodes_all.to_csv(PATIENT_CSV, index=False)

    print(f"Wrote {WEEKLY_CSV} ({len(weekly_all)} rows, {len(weekly_all.columns)} columns)")
    print(f"Wrote {PATIENT_CSV} ({len(episodes_all)} episode rows)")
    print(f"Panel: {len(constants.ICBS)} ICBs x {constants.DATASET.n_weeks} weeks + England aggregate")


if __name__ == "__main__":
    main()
