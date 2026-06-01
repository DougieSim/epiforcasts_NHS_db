"""
Plotting functions for the NHS system pressure model.

All functions that produce matplotlib figures live here.
The module is shared between:
  - bayesian_pressure_model.py  (offline inference diagnostics)
  - app.py                      (Streamlit dashboard figures)

Offline functions (call plt.show()):
    plot_prior_predictive
    plot_posterior_predictive
    plot_residuals
    plot_seasonal_effects
    plot_pressure_trajectories
    plot_direction_of_travel
    plot_clinical_summary

Dashboard functions (return a plt.Figure, no plt.show()):
    fig_pressure_question
    fig_pressure_trajectory    (Kalman-filtered, includes seasonal component)
    fig_seasonal_effects
    fig_clinical_summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

SEASON_NAMES = ["Winter", "Spring", "Summer", "Autumn"]


# ─────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────

def _week_season_map(enc: dict) -> np.ndarray:
    """
    Return an array of shape (n_weeks,) giving the season index
    for each unique week, in order.
    """
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    return np.array([
        int(season_idx[np.where(week_idx == w)[0][0]])
        for w in range(enc["n_weeks"])
    ])


def _season_per_week_samples(
    season_f: np.ndarray,
    week_season: np.ndarray,
) -> np.ndarray:
    """
    Map posterior season effect samples to per-week values.

    Parameters
    ----------
    season_f : (S, 4)   posterior season_effects samples
    week_season : (n_weeks,)  season index per week

    Returns
    -------
    (S, n_weeks)  season effect for each sample at each week
    """
    return season_f[:, week_season]


def _kalman_filter(
    obs_series: np.ndarray,
    sigma_drift: float,
    sigma_obs: float,
    level_init: float = 0.0,
    var_init: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward Kalman filter for the local level model.

    Operates on the latent level (observation noise is scaled to latent
    units internally). Returns filtered_mean and filtered_std on the
    latent scale. Uncertainty is honest — wide early, narrows with data.
    """
    n                = len(obs_series)
    filtered_mean    = np.zeros(n)
    filtered_std     = np.zeros(n)
    sigma_obs_latent = sigma_obs / 6.0
    m, v             = level_init, var_init

    for t, bed_occ in enumerate(obs_series):
        y_latent = (bed_occ - 85.0) / 6.0
        v_pred   = v + sigma_drift ** 2
        k        = v_pred / (v_pred + sigma_obs_latent ** 2)
        m        = m + k * (y_latent - m)
        v        = (1 - k) * v_pred
        filtered_mean[t] = m
        filtered_std[t]  = np.sqrt(v)

    return filtered_mean, filtered_std


# ─────────────────────────────────────────
# Offline plots  (call plt.show())
# ─────────────────────────────────────────

def plot_prior_predictive(idata: az.InferenceData) -> None:
    """Print summary statistics and show prior predictive distribution."""
    prior_beds = idata.prior_predictive["bed_obs"].values.squeeze()
    flat       = prior_beds.flatten()
    print(f"Mean:        {flat.mean():.1f}")
    print(f"Std:         {flat.std():.1f}")
    print(f"Range:       [{flat.min():.1f}, {flat.max():.1f}]")
    print(f"% above 100: {(flat > 100).mean()*100:.1f}%")
    print(f"% below 0:   {(flat < 0).mean()*100:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(100, prior_beds.shape[0])):
        ax.hist(prior_beds[i], bins=30, alpha=0.05, color="steelblue", density=True)
    ax.axvline(prior_beds.mean(), color="red",    linewidth=2,    label="Mean")
    ax.axvline(85,               color="black",  linestyle="--", label="Baseline (85%)")
    ax.axvline(100,              color="orange", linestyle="--", label="Max (100%)")
    ax.set_xlabel("Bed occupancy (%)")
    ax.set_title("Prior predictive check")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_posterior_predictive(
    idata: az.InferenceData,
    observed: np.ndarray,
) -> None:
    post_beds = idata.posterior_predictive["bed_obs"].values.squeeze()
    post_beds = post_beds.reshape(-1, len(observed))
    fig, ax   = plt.subplots(figsize=(8, 4))
    for i in range(min(100, post_beds.shape[0])):
        ax.hist(post_beds[i], bins=40, alpha=0.05, color="steelblue", density=True)
    ax.hist(observed, bins=40, alpha=0.8, color="red",
            density=True, histtype="step", linewidth=2, label="Observed")
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
    """
    Posterior distribution of each season's effect on bed occupancy.
    Answers: how much does winter raise occupancy above the annual mean?
    """
    effects = idata.posterior["season_effects"].values.reshape(-1, 4) * 6
    colors  = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, color) in enumerate(zip(SEASON_NAMES, colors)):
        samples     = effects[:, i]
        lo, mid, hi = np.percentile(samples, [5.5, 50, 94.5])
        ax.barh(i, mid, xerr=[[mid - lo], [hi - mid]],
                color=color, alpha=0.75, height=0.5, capsize=4)
        ax.text(mid + (0.05 if mid >= 0 else -0.05), i,
                f"  {mid:+.1f}%", va="center", fontsize=10,
                ha="left" if mid >= 0 else "right")

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(SEASON_NAMES)
    ax.set_xlabel("Effect on bed occupancy (%) relative to annual mean")
    ax.set_title("Seasonal effects — shared across all ICBs (89% CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_pressure_trajectories(
    idata: az.InferenceData,
    enc: dict,
    df: pd.DataFrame,
) -> None:
    """
    Per-ICB fitted bed occupancy (level + season) vs observed.
    Shows total pressure — the clinically meaningful quantity.
    """
    level_f    = idata.posterior["level"].values
    season_f   = idata.posterior["season_effects"].values
    level_f    = level_f.reshape(-1, level_f.shape[2], level_f.shape[3])
    season_f   = season_f.reshape(-1, 4)

    weeks           = np.arange(enc["n_weeks"]) + df["week"].min()
    week_season     = _week_season_map(enc)
    season_per_week = _season_per_week_samples(season_f, week_season)
    n_icb           = enc["n_icb"]

    fig, axes = plt.subplots(n_icb, 1, figsize=(12, 3 * n_icb), sharex=True)
    if n_icb == 1:
        axes = [axes]

    for i, (ax, icb_name) in enumerate(zip(axes, enc["categories"])):
        total = level_f[:, :, i] + season_per_week
        lo    = np.percentile(total, 10, axis=0)
        mid   = np.percentile(total, 50, axis=0)
        hi    = np.percentile(total, 90, axis=0)

        ax.fill_between(weeks, 85 + lo * 6, 85 + hi * 6,
                        alpha=0.3, color="steelblue", label="80% CI")
        ax.plot(weeks, 85 + mid * 6, color="steelblue",
                linewidth=1.5, label="Fitted median")
        obs = df[df["icb"] == icb_name]
        ax.scatter(obs["week"], obs["bed_occupancy"],
                   s=6, color="black", alpha=0.4, label="Observed", zorder=3)
        ax.axhline(95,  color="orange", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(100, color="red",    linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("Bed occupancy (%)")
        ax.set_title(icb_name)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Week")
    plt.suptitle("Fitted bed occupancy per ICB", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_direction_of_travel(
    idata: az.InferenceData,
    enc: dict,
    lookback_weeks: int = 4,
) -> None:
    from bayesian_pressure_model import direction_of_travel
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, icb_name in enumerate(enc["categories"]):
        dot      = direction_of_travel(idata, i, lookback_weeks)
        p_rising = float(np.mean(dot > 0))
        ax.hist(dot * 6, bins=40, density=True, alpha=0.5,
                label=f"{icb_name}  (P(rising)={p_rising:.0%})")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Change in underlying level (% occ) over last {lookback_weeks} weeks")
    ax.set_title("Direction of travel (seasonal component removed)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_clinical_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(14, max(3, len(summary) * 0.6 + 1.5))
    )
    icbs   = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y      = np.arange(len(icbs))
    colors = [
        "#b91c1c" if p > 0.25 else "#d97706" if p > 0.08 else "#15803d"
        for p in summary["p_above_high"]
    ]

    ax   = axes[0]
    xerr = np.array([
        summary["bed_occ_median"] - (85 + summary["pressure_lo"] * 6),
        (85 + summary["pressure_hi"] * 6) - summary["bed_occ_median"],
    ])
    ax.barh(y, summary["bed_occ_median"], xerr=xerr, color=colors,
            alpha=0.75, height=0.6, capsize=3)
    ax.axvline(95,  color="orange", linestyle="--", linewidth=1)
    ax.axvline(100, color="red",    linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(icbs, fontsize=9)
    ax.set_xlabel("Bed occupancy % (median + 80% CI)")
    ax.set_title("Current pressure (incl. season)")

    ax = axes[1]
    ax.barh(y, summary["p_above_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(0.25, color="red",    linestyle="--", linewidth=1, label="Elevated (0.25)")
    ax.axvline(0.08, color="orange", linestyle="--", linewidth=1, label="Medium (0.08)")
    ax.set_yticks(y); ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("P(pressure above high reference)")
    ax.set_title("Risk probability")
    ax.legend(fontsize=7)

    ax         = axes[2]
    dot_colors = [
        "#b91c1c" if d > 0.05 else "#15803d" if d < -0.05 else "#64748b"
        for d in summary["dot_median"]
    ]
    ax.barh(y, summary["dot_median"] * 6, color=dot_colors, alpha=0.75, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("Underlying level change (% occ, last 4 weeks)")
    ax.set_title("Direction of travel")

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
    threshold_baseline: float = 0.0,
    threshold_concern: float  = 0.5,
    threshold_elevated: float = 1.1,
) -> plt.Figure:
    """Evidence distribution plot for the dashboard."""
    from dashboard_shared import credible_triplet
    lo, mid, hi = credible_triplet(samples, credible_mass)
    pct_label   = f"{int(credible_mass * 100)}% plausible range"

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    ax.hist(samples, bins=40, density=True, alpha=0.78,
            color="#1d4ed8", edgecolor="white", linewidth=0.5)
    ax.axvspan(lo, hi, alpha=0.15, color="#1e3a8a", label=pct_label)
    ax.axvline(threshold_baseline, color="#64748b", linestyle="--",
               linewidth=1.5, label="Baseline reference (demo)")
    ax.axvline(threshold_concern,  color="#d97706", linestyle="--",
               linewidth=1.5, label="Concern reference (demo)")
    ax.axvline(threshold_elevated, color="#b91c1c", linestyle="--",
               linewidth=1.5, label="High pressure reference (demo)")
    if show_median_line:
        ax.axvline(mid, color="#0f172a", linestyle="-",
                   linewidth=1.0, alpha=0.85, label="Median")

    ax.set_xlabel(
        "System pressure index (modelled, unitless — not an NHS operational metric)"
    )
    ax.set_ylabel("Relative plausibility")
    ax.set_title(
        "Where does the evidence put system pressure for this area?",
        fontsize=13, pad=10,
    )
    ax.set_xlim(
        min(samples.min(), threshold_baseline) - 0.35,
        max(samples.max(), threshold_elevated) + 0.35,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def fig_pressure_trajectory(
    idata: az.InferenceData,
    df: pd.DataFrame,
    icb: str,
) -> plt.Figure | None:
    """
    Kalman-filtered pressure trajectory including seasonal component.

    Uses posterior mean parameters to run a causal forward filter.
    Seasonal effect is added back before converting to occupancy scale
    so the displayed trajectory matches observed values.
    CI is honest — wide early, narrows as evidence accumulates.
    """
    icbs = list(idata.attrs.get("icbs", []))
    if icb not in icbs:
        return None

    post         = idata.posterior
    sigma_drift  = float(post["sigma_drift"].mean())
    sigma_obs    = float(post["sigma_obs"].mean())
    level_init   = float(
        post["level_init"].mean(("chain", "draw")).values[icbs.index(icb)]
    )
    var_init     = float(post["sigma_icb"].mean()) ** 2

    icb_df       = df[df["icb"] == icb].sort_values("week")
    obs_beds     = icb_df["bed_occupancy"].values
    weeks        = icb_df["week"].values
    n_weeks      = len(weeks)

    filtered_mean, filtered_std = _kalman_filter(
        obs_beds,
        sigma_drift=sigma_drift,
        sigma_obs=sigma_obs,
        level_init=level_init,
        var_init=var_init,
    )

    # Seasonal contribution per week using posterior mean season effects.
    # Season index is derived from the month column stored in the dataframe —
    # same MONTH_TO_SEASON mapping used during encode().
    season_mean = post["season_effects"].values.reshape(-1, 4).mean(axis=0)
    _m2s = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}

    if "month" in icb_df.columns:
        season_indices = []
        for idx, (m, w) in enumerate(zip(icb_df["month"].values, weeks)):
            if pd.notna(m):
                season_indices.append(_m2s[int(m)])
            else:
                season_indices.append(int(idx % 4))
        season_week = season_mean[np.array(season_indices)]
    else:
        season_week = np.array([
            season_mean[int(i % 4)] for i in range(n_weeks)
        ])

    occ_mean = 85 + (filtered_mean + season_week) * 6
    occ_lo   = 85 + (filtered_mean - 1.645 * filtered_std + season_week) * 6
    occ_hi   = 85 + (filtered_mean + 1.645 * filtered_std + season_week) * 6

    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.fill_between(weeks, occ_lo, occ_hi,
                    alpha=0.25, color="#1d4ed8", label="90% filtered CI")
    ax.plot(weeks, occ_mean, color="#1d4ed8", linewidth=1.8,
            label="Filtered estimate (incl. season)")
    ax.scatter(weeks, obs_beds, s=8, color="#0f172a", alpha=0.45,
               label="Observed", zorder=3)
    ax.axhline(95,  color="#d97706", linestyle="--", linewidth=1,
               alpha=0.8, label="95% reference")
    ax.axhline(100, color="#b91c1c", linestyle="--", linewidth=1,
               alpha=0.8, label="100% reference")
    ax.axvline(weeks[-1], color="#64748b", linestyle=":",
               linewidth=1.2, alpha=0.8, label=f"Latest (week {weeks[-1]})")
    ax.set_ylabel("Bed occupancy (%)")
    ax.set_xlabel("Week")
    ax.set_title(
        f"Filtered pressure trajectory — {icb}  "
        "(CI widens at early weeks, narrows as evidence accumulates)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def fig_seasonal_effects(idata: az.InferenceData) -> plt.Figure:
    """Posterior seasonal effects — dashboard version (returns Figure)."""
    effects = idata.posterior["season_effects"].values.reshape(-1, 4) * 6
    colors  = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]

    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for i, (name, color) in enumerate(zip(SEASON_NAMES, colors)):
        samples     = effects[:, i]
        lo, mid, hi = np.percentile(samples, [5.5, 50, 94.5])
        ax.barh(i, mid, xerr=[[mid - lo], [hi - mid]],
                color=color, alpha=0.75, height=0.5, capsize=4)
        ax.text(
            mid + (0.05 if mid >= 0 else -0.05), i,
            f"  {mid:+.1f}%", va="center", fontsize=10,
            ha="left" if mid >= 0 else "right",
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(SEASON_NAMES)
    ax.set_xlabel("Effect on bed occupancy (%) relative to annual mean")
    ax.set_title("Seasonal effects — shared across all ICBs (89% CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def fig_clinical_summary(
    idata: az.InferenceData,
    icb_filter: str | None = None,
) -> plt.Figure:
    """Three-panel clinical summary — dashboard version (returns Figure)."""
    from dashboard_shared import current_season_index
    icbs        = list(idata.attrs.get("icbs", []))
    level_f     = idata.posterior["level"].values.reshape(
        -1, idata.posterior["level"].shape[2], idata.posterior["level"].shape[3]
    )
    season_f    = idata.posterior["season_effects"].values.reshape(-1, 4)
    season_now  = current_season_index(idata)

    icb_filter_str = str(icb_filter) if icb_filter is not None else None

    rows = []
    for i, icb_name in enumerate(icbs):
        if icb_filter_str is not None and str(icb_name) != icb_filter_str:
            continue
        lev   = level_f[:, -1, i]
        total = lev + season_f[:, season_now]
        dot   = (level_f[:, -1, i] - level_f[:, -4, i]) * 6
        rows.append(dict(
            icb=icb_name,
            median=float(85 + np.median(total) * 6),
            lo=float(85 + np.percentile(lev, 10) * 6),
            hi=float(85 + np.percentile(lev, 90) * 6),
            p_high=float(np.mean(total > 1.1)),
            dot=float(np.median(dot)),
        ))

    summary     = pd.DataFrame(rows).sort_values("median", ascending=False)
    short_names = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y           = np.arange(len(summary))
    n_rows      = len(summary)
    fig_height  = 2.5 if n_rows == 1 else max(3, n_rows * 0.7 + 1.5)
    colors      = [
        "#b91c1c" if p > 0.25 else "#d97706" if p > 0.08 else "#15803d"
        for p in summary["p_high"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, fig_height), layout="constrained")

    ax   = axes[0]
    xerr = np.array([
        summary["median"] - summary["lo"],
        summary["hi"]     - summary["median"],
    ])
    ax.barh(y, summary["median"], xerr=xerr, color=colors,
            alpha=0.75, height=0.6, capsize=3)
    ax.axvline(95,  color="#d97706", linestyle="--", linewidth=1)
    ax.axvline(100, color="#b91c1c", linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Bed occupancy % (median + 80% CI)")
    ax.set_title("Current pressure (incl. season)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.barh(y, summary["p_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(0.25, color="#b91c1c", linestyle="--", linewidth=1, label="Elevated")
    ax.axvline(0.08, color="#d97706", linestyle="--", linewidth=1, label="Medium")
    ax.set_yticks(y); ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel("P(pressure above high reference)")
    ax.set_title("Risk probability")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax         = axes[2]
    dot_colors = [
        "#b91c1c" if d > 0.5 else "#15803d" if d < -0.5 else "#64748b"
        for d in summary["dot"]
    ]
    ax.barh(y, summary["dot"], color=dot_colors, alpha=0.75, height=0.6)
    ax.axvline(0, color="#0f172a", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel("Underlying level change (% occ, last 4 weeks)")
    ax.set_title("Direction of travel")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)