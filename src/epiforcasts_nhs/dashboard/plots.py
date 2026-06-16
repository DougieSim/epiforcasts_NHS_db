"""
Plotting functions for the NHS system pressure model.

Offline functions (call plt.show()):
    plot_prior_predictive, plot_posterior_predictive, plot_residuals,
    plot_seasonal_effects, plot_pressure_trajectories, plot_direction_of_travel,
    plot_clinical_summary

Dashboard functions (return plt.Figure):
    fig_pressure_question, fig_pressure_trajectory,
    fig_seasonal_effects, fig_clinical_summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az

from epiforcasts_nhs.dashboard.utils import (
    LATENT_BASELINE,
    LATENT_SCALE,
    LOOKBACK_WEEKS,
    MONTH_TO_SEASON,
    THRESHOLD_BASELINE,
    THRESHOLD_CONCERN,
    THRESHOLD_ELEVATED,
    direction_of_travel,
)
from epiforcasts_nhs.dashboard.utils import (
    BED_OCC_REF_CONCERN,
    BED_OCC_REF_HIGH,
    CI_80_LOWER,
    CI_80_UPPER,
    CI_89_LOWER,
    CI_89_UPPER,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    SEASON_COLORS,
    SEASON_NAMES,
    credible_triplet,
    current_season_index,
    dot_travel_colors,
    pressure_colors,
    seasonal_effect_samples,
    season_per_week_samples,
    week_season_map,
)

# Max prior/posterior samples to overlay in diagnostic plots
_MAX_OVERLAY_SAMPLES = 100


# ─────────────────────────────────────────
# Offline plots  (call plt.show())
# ─────────────────────────────────────────

def plot_prior_predictive(idata: az.InferenceData) -> None:
    prior_beds = idata.prior_predictive["bed_obs"].values.squeeze()
    flat       = prior_beds.flatten()
    print(f"Mean: {flat.mean():.1f}  Std: {flat.std():.1f}  "
          f"Range: [{flat.min():.1f}, {flat.max():.1f}]")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(_MAX_OVERLAY_SAMPLES, prior_beds.shape[0])):
        ax.hist(prior_beds[i], bins=30, alpha=0.05, color="steelblue", density=True)
    ax.axvline(prior_beds.mean(), color="red",   linewidth=2,    label="Mean")
    ax.axvline(LATENT_BASELINE,  color="black",  linestyle="--", label=f"Baseline ({LATENT_BASELINE:.0f}%)")
    ax.axvline(BED_OCC_REF_HIGH, color="orange", linestyle="--", label=f"Max ({BED_OCC_REF_HIGH:.0f}%)")
    ax.set_xlabel("Bed occupancy (%)")
    ax.set_title("Prior predictive check")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_posterior_predictive(idata: az.InferenceData, observed: np.ndarray) -> None:
    post_beds = idata.posterior_predictive["bed_obs"].values.squeeze().reshape(-1, len(observed))
    fig, ax   = plt.subplots(figsize=(8, 4))
    for i in range(min(_MAX_OVERLAY_SAMPLES, post_beds.shape[0])):
        ax.hist(post_beds[i], bins=40, alpha=0.05, color="steelblue", density=True)
    ax.hist(observed, bins=40, alpha=0.8, color="red", density=True,
            histtype="step", linewidth=2, label="Observed")
    ax.set_xlabel("Bed occupancy (%)")
    ax.set_title("Posterior predictive check")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(idata: az.InferenceData, observed: np.ndarray) -> None:
    predicted = idata.posterior_predictive["bed_obs"].mean(("chain", "draw")).values
    residuals = observed - predicted
    plt.figure(figsize=(8, 3))
    plt.plot(residuals, alpha=0.7, linewidth=0.8)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residuals")
    plt.xlabel("Observation index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


def plot_seasonal_effects(idata: az.InferenceData) -> None:
    effects = seasonal_effect_samples(idata) * LATENT_SCALE
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, color) in enumerate(zip(SEASON_NAMES, SEASON_COLORS)):
        samples     = effects[:, i]
        lo, mid, hi = np.percentile(samples, [CI_89_LOWER, 50.0, CI_89_UPPER])
        ax.barh(i, mid, xerr=[[mid - lo], [hi - mid]], color=color, alpha=0.75, height=0.5, capsize=4)
        ax.text(mid + (0.05 if mid >= 0 else -0.05), i, f"  {mid:+.1f}%", va="center",
                fontsize=10, ha="left" if mid >= 0 else "right")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(SEASON_NAMES)
    ax.set_xlabel("Effect on bed occupancy (%) relative to annual mean")
    ax.set_title("Seasonal effects — shared across all ICBs (89% CI)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_pressure_trajectories(idata: az.InferenceData, enc: dict, df: pd.DataFrame) -> None:
    level_f  = idata.posterior["level"].values.reshape(
        -1, idata.posterior["level"].shape[2], idata.posterior["level"].shape[3]
    )
    season_f        = seasonal_effect_samples(idata)
    weeks           = np.arange(enc["n_weeks"]) + df["week"].min()
    w_season        = week_season_map(enc)
    season_per_week = season_per_week_samples(season_f, w_season)
    n_icb           = enc["n_icb"]

    fig, axes = plt.subplots(n_icb, 1, figsize=(12, 3 * n_icb), sharex=True)
    if n_icb == 1:
        axes = [axes]

    for i, (ax, icb_name) in enumerate(zip(axes, enc["categories"])):
        total = level_f[:, :, i] + season_per_week
        lo    = np.percentile(total, CI_80_LOWER, axis=0)
        mid   = np.percentile(total, 50.0,        axis=0)
        hi    = np.percentile(total, CI_80_UPPER,  axis=0)
        ax.fill_between(weeks,
                        LATENT_BASELINE + lo * LATENT_SCALE,
                        LATENT_BASELINE + hi * LATENT_SCALE,
                        alpha=0.3, color="steelblue", label="80% CI")
        ax.plot(weeks, LATENT_BASELINE + mid * LATENT_SCALE,
                color="steelblue", linewidth=1.5, label="Fitted median")
        obs = df[df["icb"] == icb_name]
        ax.scatter(obs["week"], obs["bed_occupancy"], s=6, color="black", alpha=0.4, label="Observed", zorder=3)
        ax.axhline(BED_OCC_REF_CONCERN, color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(BED_OCC_REF_HIGH,    color="red",    linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("Bed Demand (%)"); ax.set_title(icb_name)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Week")
    plt.suptitle("Fitted bed occupancy per ICB", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_direction_of_travel(idata: az.InferenceData, enc: dict, lookback_weeks: int = LOOKBACK_WEEKS) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, icb_name in enumerate(enc["categories"]):
        dot = direction_of_travel(idata, i, lookback_weeks)
        ax.hist(dot * LATENT_SCALE, bins=40, density=True, alpha=0.5,
                label=f"{icb_name}  (P(rising)={np.mean(dot > 0):.0%})")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Change in underlying level (% occ) over last {lookback_weeks} weeks")
    ax.set_title("Direction of travel (seasonal component removed)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_clinical_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, len(summary) * 0.6 + 1.5)))
    icbs   = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y      = np.arange(len(icbs))
    colors = pressure_colors(summary["p_above_high"])

    ax = axes[0]
    xerr = np.array([
        summary["bed_occ_median"] - (LATENT_BASELINE + summary["pressure_lo"] * LATENT_SCALE),
        (LATENT_BASELINE + summary["pressure_hi"] * LATENT_SCALE) - summary["bed_occ_median"],
    ])
    ax.barh(y, summary["bed_occ_median"], xerr=xerr, color=colors, alpha=0.75, height=0.6, capsize=3)
    ax.axvline(BED_OCC_REF_CONCERN, color="orange", linestyle="--", linewidth=1)
    ax.axvline(BED_OCC_REF_HIGH,    color="red",    linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(icbs, fontsize=9)
    ax.set_xlabel("Bed occupancy % (median + 80% CI)"); ax.set_title("Current pressure (incl. season)")

    ax = axes[1]
    ax.barh(y, summary["p_above_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(RISK_THRESHOLD_HIGH,   color="red",    linestyle="--", linewidth=1, label=f"Elevated ({RISK_THRESHOLD_HIGH})")
    ax.axvline(RISK_THRESHOLD_MEDIUM, color="orange", linestyle="--", linewidth=1, label=f"Medium ({RISK_THRESHOLD_MEDIUM})")
    ax.set_yticks(y); ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("P(pressure above high reference)"); ax.set_title("Risk probability")
    ax.legend(fontsize=7)

    ax = axes[2]
    dot_pct = [d * LATENT_SCALE for d in summary["dot_median"]]
    ax.barh(y, dot_pct, color=dot_travel_colors(dot_pct), alpha=0.75, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("Underlying level change (% occ, last 4 weeks)"); ax.set_title("Direction of travel")

    plt.suptitle("ICB System Pressure — Clinical Summary", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Dashboard figures  (return plt.Figure)
# ─────────────────────────────────────────

def fig_pressure_question(
    samples: np.ndarray,
    *,
    credible_mass: float,
    show_median_line: bool,
) -> plt.Figure:
    lo, mid, hi = credible_triplet(samples, credible_mass)
    pct_label   = f"{int(credible_mass * 100)}% plausible range"

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    ax.hist(samples, bins=40, density=True, alpha=0.78, color="#1d4ed8", edgecolor="white", linewidth=0.5)
    ax.axvspan(lo, hi, alpha=0.15, color="#1e3a8a", label=pct_label)
    ax.axvline(THRESHOLD_BASELINE, color="#64748b", linestyle="--", linewidth=1.5, label="Baseline reference (demo)")
    ax.axvline(THRESHOLD_CONCERN,  color="#d97706", linestyle="--", linewidth=1.5, label="Concern reference (demo)")
    ax.axvline(THRESHOLD_ELEVATED, color="#b91c1c", linestyle="--", linewidth=1.5, label="High pressure reference (demo)")
    if show_median_line:
        ax.axvline(mid, color="#0f172a", linestyle="-", linewidth=1.0, alpha=0.85, label="Median")

    ax.set_xlabel("System pressure index (modelled, unitless — not an NHS operational metric)")
    ax.set_ylabel("Relative plausibility")
    ax.set_title("Where does the evidence put system pressure for this area?", fontsize=13, pad=10)
    ax.set_xlim(min(samples.min(), THRESHOLD_BASELINE) - 0.35,
                max(samples.max(), THRESHOLD_ELEVATED) + 0.35)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return fig


def fig_pressure_trajectory(idata: az.InferenceData, df: pd.DataFrame, icb: str) -> plt.Figure | None:
    """Posterior pressure trajectory including seasonal component."""
    icbs = list(idata.attrs.get("icbs", []))
    if icb not in icbs:
        return None

    icb_idx  = icbs.index(icb)
    icb_df   = df[df["icb"] == icb].sort_values("week")
    obs_beds = icb_df["bed_occupancy"].values
    weeks    = icb_df["week"].values
    date = pd.to_datetime(icb_df["week_date"].values)
    n_weeks  = len(weeks)

    # level: (chain, draw, n_weeks, n_icb) → (n_samples, n_weeks) for this ICB
    level_samples = idata.posterior["level"].values[:, :, :n_weeks, icb_idx].reshape(-1, n_weeks)

    # season effects: (n_samples, 4) → map to per-week (n_samples, n_weeks)
    season_f = seasonal_effect_samples(idata)
    if "month" in icb_df.columns:
        season_indices = np.array([
            MONTH_TO_SEASON[int(m)] if pd.notna(m) else int(idx % 4)
            for idx, m in enumerate(icb_df["month"].values)
        ])
    else:
        season_indices = np.array([int(i % 4) for i in range(n_weeks)])
    season_week = season_f[:, season_indices]  # (n_samples, n_weeks)

    occ_samples = LATENT_BASELINE + (level_samples + season_week) * LATENT_SCALE
    occ_lo   = np.percentile(occ_samples, 5.0,  axis=0)
    occ_mean = np.percentile(occ_samples, 50.0, axis=0)
    occ_hi   = np.percentile(occ_samples, 95.0, axis=0)

    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.fill_between(date, occ_lo, occ_hi, alpha=0.25, color="#1d4ed8", label="90% posterior CI")
    ax.plot(date, occ_mean, color="#1d4ed8", linewidth=1.8, label="Posterior median (incl. season)")
    ax.scatter(date, obs_beds, s=8, color="#0f172a", alpha=0.45, label="Observed", zorder=3)
    ax.axhline(BED_OCC_REF_CONCERN, color="#d97706", linestyle="--", linewidth=1, alpha=0.8,
               label=f"{BED_OCC_REF_CONCERN:.0f}% reference")
    ax.axhline(BED_OCC_REF_HIGH, color="#b91c1c", linestyle="--", linewidth=1, alpha=0.8,
               label=f"{BED_OCC_REF_HIGH:.0f}% reference")
    ax.axvline(date[-1], color="#64748b", linestyle=":", linewidth=1.2, alpha=0.8,
               label=f"Latest ({date[-1]})")
    ax.set_ylabel("Bed occupancy (%)"); ax.set_xlabel("Date")

    ax.set_title(f"Posterior pressure trajectory — {icb}", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate()
    return fig


def fig_seasonal_effects(idata: az.InferenceData) -> plt.Figure:
    effects = seasonal_effect_samples(idata) * LATENT_SCALE
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for i, (name, color) in enumerate(zip(SEASON_NAMES, SEASON_COLORS)):
        samples     = effects[:, i]
        lo, mid, hi = np.percentile(samples, [CI_89_LOWER, 50.0, CI_89_UPPER])
        ax.barh(i, mid, xerr=[[mid - lo], [hi - mid]], color=color, alpha=0.75, height=0.5, capsize=4)
        ax.text(mid + (0.05 if mid >= 0 else -0.05), i, f"  {mid:+.1f}%", va="center",
                fontsize=10, ha="left" if mid >= 0 else "right")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(SEASON_NAMES)
    ax.set_xlabel("Effect on bed occupancy (%) relative to annual mean")
    ax.set_title("Seasonal effects — shared across all ICBs (89% CI)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return fig


def fig_clinical_summary(idata: az.InferenceData, icb_filter: str | None = None) -> plt.Figure:
    icbs       = list(idata.attrs.get("icbs", []))
    level_f    = idata.posterior["level"].values.reshape(
        -1, idata.posterior["level"].shape[2], idata.posterior["level"].shape[3]
    )
    season_f   = seasonal_effect_samples(idata)
    season_now = current_season_index(idata)
    icb_filter_str = str(icb_filter) if icb_filter is not None else None

    rows = []
    for i, icb_name in enumerate(icbs):
        if icb_filter_str is not None and str(icb_name) != icb_filter_str:
            continue
        lev   = level_f[:, -1, i]
        total = lev + season_f[:, season_now]
        dot   = (level_f[:, -1, i] - level_f[:, -LOOKBACK_WEEKS, i]) * LATENT_SCALE
        rows.append(dict(
            icb=icb_name,
            median=float(LATENT_BASELINE + np.median(total) * LATENT_SCALE),
            lo=float(LATENT_BASELINE + np.percentile(lev, CI_80_LOWER) * LATENT_SCALE),
            hi=float(LATENT_BASELINE + np.percentile(lev, CI_80_UPPER) * LATENT_SCALE),
            p_high=float(np.mean(total > THRESHOLD_ELEVATED)),
            dot=float(np.median(dot)),
        ))

    summary     = pd.DataFrame(rows).sort_values("median", ascending=False)
    short_names = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y           = np.arange(len(summary))
    n_rows      = len(summary)
    fig_height  = 2.5 if n_rows == 1 else max(3, n_rows * 0.7 + 1.5)
    colors      = pressure_colors(summary["p_high"])

    fig, axes = plt.subplots(1, 3, figsize=(14, fig_height), layout="constrained")

    ax = axes[0]
    xerr = np.array([summary["median"] - summary["lo"], summary["hi"] - summary["median"]])
    ax.barh(y, summary["median"], xerr=xerr, color=colors, alpha=0.75, height=0.6, capsize=3)
    ax.axvline(BED_OCC_REF_CONCERN, color="#d97706", linestyle="--", linewidth=1)
    ax.axvline(BED_OCC_REF_HIGH,    color="#b91c1c", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Bed occupancy % (median + 80% CI)"); ax.set_title("Current pressure (incl. season)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.barh(y, summary["p_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(RISK_THRESHOLD_HIGH,   color="#b91c1c", linestyle="--", linewidth=1, label="Elevated")
    ax.axvline(RISK_THRESHOLD_MEDIUM, color="#d97706", linestyle="--", linewidth=1, label="Medium")
    ax.set_yticks(y); ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel("P(pressure above high reference)"); ax.set_title("Risk probability")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[2]
    ax.barh(y, summary["dot"], color=dot_travel_colors(summary["dot"]), alpha=0.75, height=0.6)
    ax.axvline(0, color="#0f172a", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel("Underlying level change (% occ, last 4 weeks)"); ax.set_title("Direction of travel")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    return fig


