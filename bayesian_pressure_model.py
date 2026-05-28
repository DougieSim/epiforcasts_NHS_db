"""
Bayesian NHS system pressure model — local level (dynamic linear model).

Each ICB has a latent pressure level that evolves as a random walk:

    level[icb, 0]   ~ Normal(mu_national, sigma_icb)
    level[icb, t]   = level[icb, t-1] + N(0, sigma_drift)
    bed_obs[i]      ~ Normal(85 + level[icb, week_i] * 6, sigma_obs)

This answers three clinical questions the static model cannot:
  1. What is pressure RIGHT NOW (not the historical average)?
  2. Is pressure rising or falling?
  3. Which weeks were anomalous?

The dashboard reads level[icb, T] — the final week's posterior — as the
current pressure index, directly compatible with dashboard_shared.py.
"""

from __future__ import annotations

import os
import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

_FAST = os.environ.get("PRESSURE_MODEL_FAST", "1").strip().lower() not in (
    "0", "false", "no",
)

rng = np.random.default_rng(42)
HOLDOUT_WEEKS = 12


# ─────────────────────────────────────────
# Data
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove England aggregate and split train / holdout."""
    df = df[df["icb"] != "England"].copy()
    cutoff = df["week"].max() - HOLDOUT_WEEKS + 1
    train = df[df["week"] < cutoff].copy()
    test  = df[df["week"] >= cutoff].copy()
    return train, test


def encode(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Encode training and test data into model-ready arrays.

    Returns a dict containing:
      icb_codes   — integer ICB index per training observation
      week_idx    — 0-based week index per training observation
      beds        — observed bed occupancy per training observation
      n_icb       — number of ICBs
      n_weeks     — number of training weeks
      categories  — ordered ICB names (index matches icb_codes)
      test_*      — equivalent arrays for holdout evaluation
    """
    cat        = train["icb"].astype("category")
    categories = cat.cat.categories
    icb_codes  = cat.cat.codes.values
    week_idx   = (train["week"].values - train["week"].min()).astype(int)
    beds       = train["bed_occupancy"].values
    n_icb      = len(categories)
    n_weeks    = int(week_idx.max()) + 1

    test_icb_codes = pd.Categorical(test["icb"], categories=categories).codes
    test_week_idx  = (test["week"].values - train["week"].min()).astype(int)
    test_beds      = test["bed_occupancy"].values

    return dict(
        icb_codes=icb_codes,
        week_idx=week_idx,
        beds=beds,
        n_icb=n_icb,
        n_weeks=n_weeks,
        categories=categories,
        test_icb_codes=test_icb_codes,
        test_week_idx=test_week_idx,
        test_beds=test_beds,
    )


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

def build_model(enc: dict) -> pm.Model:
    """
    Local level (dynamic linear) model.

    Each ICB has a latent pressure level that drifts week-by-week as a
    random walk. The national mean and cross-ICB variance anchor the
    initial levels; sigma_drift controls how quickly pressure can change.

    Priors
    ------
    mu_national ~ Normal(0, 1)
        National average pressure on the latent scale.
        Maps to ~85% bed occupancy at mu=0, ±6% per unit.

    sigma_icb ~ Exponential(1)
        Cross-ICB spread at initialisation. Controls how different
        ICBs are allowed to be at baseline.

    sigma_drift ~ Exponential(10)  [mean 0.1]
        Week-to-week drift in each ICB's pressure level.
        Small values = slow-moving pressure (requires sustained evidence
        to shift the estimate). This is the key clinical prior —
        tight enough that single noisy weeks don't trigger false alarms,
        but loose enough to track genuine sustained changes.

    sigma_obs ~ Exponential(5)  [mean 0.2]
        Observation noise on the bed occupancy measurement.

    Clinical outputs
    ----------------
    level[icb, T]           — current pressure posterior (dashboard)
    level[icb, T] - level[icb, T-k]  — direction of travel over k weeks
    P(level[icb,T] > threshold)       — probability of elevated pressure
    """
    icb_codes = enc["icb_codes"]
    week_idx  = enc["week_idx"]
    beds      = enc["beds"]
    n_icb     = enc["n_icb"]
    n_weeks   = enc["n_weeks"]

    with pm.Model() as model:

        # ── National anchor ───────────────────────────────────────────
        mu_national = pm.Normal("mu_national", 0, 1)
        sigma_icb   = pm.Exponential("sigma_icb", 1)

        # ── Initial pressure level per ICB ────────────────────────────
        # Each ICB starts at mu_national ± sigma_icb, allowing for
        # persistent structural differences between areas.
        level_init = pm.Normal(
            "level_init",
            mu=mu_national,
            sigma=sigma_icb,
            shape=n_icb,
        )

        # ── Week-to-week drift ────────────────────────────────────────
        # Tight prior: pressure can shift ~0.6% occupancy per week
        # on average (0.1 latent * 6). Sustained pressure over multiple
        # weeks compounds to meaningful signals.
        sigma_drift = pm.Exponential("sigma_drift", lam=10)

        # innovations: shape (n_weeks - 1, n_icb)
        innovations = pm.Normal(
            "innovations",
            0,
            sigma_drift,
            shape=(n_weeks - 1, n_icb),
        )

        # Build level via scan: level[t] = level[t-1] + innovation[t]
        # This is a pure random walk — no mean reversion.
        # Mean reversion would be appropriate if we believed pressure
        # always returns to a fixed baseline, but NHS pressure can
        # shift structurally (e.g. winter, staffing crises).
        levels_rest, _ = pytensor.scan(
            fn=lambda innov, prev: prev + innov,
            sequences=innovations,           # (n_weeks-1, n_icb)
            outputs_info=level_init,         # (n_icb,)
        )

        # level shape: (n_weeks, n_icb)
        # level[0] = level_init, level[t] = level[t-1] + innovation
        level = pm.Deterministic(
            "level",
            pt.concatenate(
                [level_init[None, :], levels_rest],
                axis=0,
            )
        )

        # ── Observation model ─────────────────────────────────────────
        sigma_obs = pm.Exponential("sigma_obs", lam=5)

        pm.Normal(
            "bed_obs",
            mu=85 + level[week_idx, icb_codes] * 6,
            sigma=sigma_obs,
            observed=beds,
        )

    return model


# ─────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────

def sample_prior(model: pm.Model, draws: int = 100) -> az.InferenceData:
    with model:
        idata = pm.sample_prior_predictive(draws=draws, random_seed=rng)
    return idata


def sample_posterior(model: pm.Model, draws: int, tune: int) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.95,
            progressbar=True,
            compute_convergence_checks=False,
        )
    return idata


def sample_posterior_predictive(
    model: pm.Model,
    idata: az.InferenceData,
) -> az.InferenceData:
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


# ─────────────────────────────────────────
# Clinical summaries
# ─────────────────────────────────────────

def current_pressure_samples(
    idata: az.InferenceData,
    icb_idx: int,
) -> np.ndarray:
    """
    Posterior samples for the CURRENT pressure level of one ICB.
    This is level[icb, T] — the final week — flattened across chains/draws.
    Used by the dashboard as the pressure index.
    """
    level = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
    return level[:, :, -1, icb_idx].ravel()


def direction_of_travel(
    idata: az.InferenceData,
    icb_idx: int,
    lookback_weeks: int = 4,
) -> np.ndarray:
    """
    Posterior samples for the change in pressure over the last N weeks.
    Positive = rising pressure, negative = falling.

    Returns samples of (level[T] - level[T - lookback_weeks]).
    """
    level = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
    current  = level[:, :, -1, icb_idx].ravel()
    previous = level[:, :, -lookback_weeks, icb_idx].ravel()
    return current - previous


def pressure_summary(
    idata: az.InferenceData,
    enc: dict,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """
    Clinical summary table: one row per ICB showing current pressure,
    direction of travel, and key probabilities.

    Columns
    -------
    icb             — ICB name
    pressure_median — median current pressure (latent scale)
    pressure_lo     — 10th percentile
    pressure_hi     — 90th percentile
    bed_occ_median  — implied bed occupancy (%)
    p_above_concern — P(pressure > 0.5)
    p_above_high    — P(pressure > 1.1)
    dot_median      — median direction of travel over lookback_weeks
    p_rising        — P(pressure rising over lookback_weeks)
    """
    rows = []
    for i, icb_name in enumerate(enc["categories"]):
        samples = current_pressure_samples(idata, i)
        dot     = direction_of_travel(idata, i, lookback_weeks)

        rows.append(dict(
            icb=icb_name,
            pressure_median=float(np.median(samples)),
            pressure_lo=float(np.percentile(samples, 10)),
            pressure_hi=float(np.percentile(samples, 90)),
            bed_occ_median=float(85 + np.median(samples) * 6),
            p_above_concern=float(np.mean(samples > 0.5)),
            p_above_high=float(np.mean(samples > 1.1)),
            dot_median=float(np.median(dot)),
            p_rising=float(np.mean(dot > 0)),
        ))

    return pd.DataFrame(rows).sort_values("pressure_median", ascending=False)


# ─────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────

def save_posteriors(
    idata: az.InferenceData,
    enc: dict,
    path: str = "posteriors.nc",
) -> None:
    """
    Save posteriors to NetCDF with a Windows-safe atomic swap.

    Strategy:
      1. Write fully to a temp file (.posteriors_tmp_*.nc)
      2. On POSIX: os.replace() is atomic
      3. On Windows: os.replace() fails if the destination is open.
         Instead write to a staging file (.posteriors_new.nc), then
         rename the old file to a backup and the new file into place.
         The window where neither file exists is microseconds — the
         dashboard's @st.cache_resource will serve the old cached copy
         during that window, which is safe.
    """
    import sys
    import tempfile
    from pathlib import Path

    final_path  = Path(path)
    idata.attrs["icbs"] = list(enc["categories"])

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=".posteriors_tmp_",
        suffix=".nc",
    )
    try:
        os.close(tmp_fd)
        idata.to_netcdf(tmp_path, engine="h5netcdf")

        if sys.platform == "win32":
            # Windows: rename old → backup, new → final, remove backup
            backup_path = final_path.with_suffix(".nc.bak")
            try:
                if final_path.exists():
                    os.replace(str(final_path), str(backup_path))
                os.replace(tmp_path, str(final_path))
                if backup_path.exists():
                    os.unlink(str(backup_path))
            except PermissionError:
                # Dashboard still has the file open — write alongside and
                # let the next cycle retry. Dashboard keeps serving old copy.
                staged = final_path.with_name(".posteriors_new.nc")
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, str(staged))
                raise RuntimeError(
                    f"posteriors.nc is locked by another process. "
                    f"Staged new posteriors at {staged} — will retry next cycle."
                )
        else:
            os.replace(tmp_path, str(final_path))

        print(f"Posteriors saved → {final_path}  ({len(enc['categories'])} ICBs)")

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────
# Plots
# ─────────────────────────────────────────

def plot_prior_predictive(idata: az.InferenceData) -> None:
    prior_beds = idata.prior_predictive["bed_obs"].values.squeeze()
    flat = prior_beds.flatten()
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
    fig, ax = plt.subplots(figsize=(8, 4))
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


def plot_pressure_trajectories(
    idata: az.InferenceData,
    enc: dict,
    train: pd.DataFrame,
) -> None:
    """
    Plot the posterior pressure trajectory for each ICB over time.
    Shows median + 80% credible band, making direction of travel visible.
    """
    level  = idata.posterior["level"].values  # (chains, draws, n_weeks, n_icb)
    level  = level.reshape(-1, level.shape[2], level.shape[3])  # (S, n_weeks, n_icb)
    weeks  = np.arange(enc["n_weeks"]) + train["week"].min()

    n_icb = enc["n_icb"]
    fig, axes = plt.subplots(n_icb, 1, figsize=(12, 3 * n_icb), sharex=True)
    if n_icb == 1:
        axes = [axes]

    for i, (ax, icb_name) in enumerate(zip(axes, enc["categories"])):
        icb_level = level[:, :, i]             # (S, n_weeks)
        lo  = np.percentile(icb_level, 10, axis=0)
        mid = np.percentile(icb_level, 50, axis=0)
        hi  = np.percentile(icb_level, 90, axis=0)

        # Convert to bed occupancy scale for clinical readability
        ax.fill_between(weeks, 85 + lo * 6, 85 + hi * 6,
                        alpha=0.3, color="steelblue", label="80% CI")
        ax.plot(weeks, 85 + mid * 6,
                color="steelblue", linewidth=1.5, label="Median")

        # Overlay observed data
        obs = train[train["icb"] == icb_name]
        ax.scatter(obs["week"], obs["bed_occupancy"],
                   s=6, color="black", alpha=0.4, label="Observed", zorder=3)

        ax.axhline(95, color="orange", linestyle="--", linewidth=1,
                   alpha=0.7, label="95% reference")
        ax.axhline(100, color="red", linestyle="--", linewidth=1,
                   alpha=0.7, label="100% reference")
        ax.set_ylabel("Bed occupancy (%)")
        ax.set_title(icb_name)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Week")
    plt.suptitle("Posterior pressure trajectories per ICB", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_direction_of_travel(
    idata: az.InferenceData,
    enc: dict,
    lookback_weeks: int = 4,
) -> None:
    """
    Show the posterior distribution of pressure change over the last N weeks
    for each ICB. Directly answers: is pressure rising or falling?
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, icb_name in enumerate(enc["categories"]):
        dot = direction_of_travel(idata, i, lookback_weeks)
        p_rising = float(np.mean(dot > 0))
        label = f"{icb_name}  (P(rising)={p_rising:.0%})"
        ax.hist(dot * 6, bins=40, density=True, alpha=0.5, label=label)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Change in bed occupancy (%) over last {lookback_weeks} weeks")
    ax.set_title("Direction of travel — posterior over pressure change")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_clinical_summary(summary: pd.DataFrame) -> None:
    """
    Heatmap-style clinical summary: ICBs ranked by current pressure
    with direction of travel and risk probabilities.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, len(summary) * 0.6 + 1.5)))

    icbs = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y    = np.arange(len(icbs))

    # Panel 1: current pressure (bed occupancy scale)
    ax = axes[0]
    xerr = np.array([
        summary["bed_occ_median"] - (85 + summary["pressure_lo"] * 6),
        (85 + summary["pressure_hi"] * 6) - summary["bed_occ_median"],
    ])
    colors = ["#b91c1c" if p > 0.25 else "#d97706" if p > 0.08 else "#15803d"
              for p in summary["p_above_high"]]
    ax.barh(y, summary["bed_occ_median"], xerr=xerr, color=colors,
            alpha=0.75, height=0.6, capsize=3)
    ax.axvline(95,  color="orange", linestyle="--", linewidth=1)
    ax.axvline(100, color="red",    linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(icbs, fontsize=9)
    ax.set_xlabel("Current bed occupancy % (median + 80% CI)")
    ax.set_title("Current pressure")

    # Panel 2: P(above high threshold)
    ax = axes[1]
    ax.barh(y, summary["p_above_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(0.25, color="red",    linestyle="--", linewidth=1, label="Elevated threshold")
    ax.axvline(0.08, color="orange", linestyle="--", linewidth=1, label="Medium threshold")
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("P(pressure above high reference)")
    ax.set_title("Risk probability")
    ax.legend(fontsize=7)

    # Panel 3: direction of travel
    ax = axes[2]
    dot_colors = ["#b91c1c" if d > 0.05 else "#15803d" if d < -0.05 else "#64748b"
                  for d in summary["dot_median"]]
    ax.barh(y, summary["dot_median"] * 6, color=dot_colors, alpha=0.75, height=0.6)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(icbs))
    ax.set_xlabel("Pressure change (% occ, last 4 weeks)")
    ax.set_title("Direction of travel")

    plt.suptitle("ICB System Pressure — Clinical Summary", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def fit_pressure_model(
    df: pd.DataFrame,
    *,
    fast: bool | None = None,
) -> tuple:
    if fast is None:
        fast = _FAST
    draws, tune = (400, 400) if fast else (2000, 2000)

    train, test = prepare_data(df)
    enc = encode(train, test)

    print(f"Training rows:  {len(train)}  |  Holdout rows: {len(test)}")
    print(f"Training weeks: {train['week'].min()}–{train['week'].max()}")
    print(f"Holdout  weeks: {test['week'].min()}–{test['week'].max()}")
    print(f"ICBs:           {list(enc['categories'])}")

    model = build_model(enc)

    # 1. Prior predictive check
    idata = sample_prior(model)

    # 2. Fit posterior
    idata = sample_posterior(model, draws, tune)

    # 3. Posterior predictive check
    idata = sample_posterior_predictive(model, idata)

    # 4. Convergence
    print("\nConvergence summary:")
    print(az.summary(idata, var_names=[
        "mu_national", "sigma_icb", "sigma_drift", "sigma_obs"
    ]))

    # 5. Clinical outputs

    summary = pressure_summary(idata, enc)
    print("\nClinical summary (ranked by current pressure):")
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    # 6. Save for dashboard
    save_posteriors(idata, enc)

    return model, idata, train, test, enc


def main():
    df = pd.read_csv("synthetic_nhs_pressure.csv")
    fit_pressure_model(df)


if __name__ == "__main__":
    main()