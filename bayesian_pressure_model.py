import os
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

_FAST = os.environ.get("PRESSURE_MODEL_FAST", "1").strip().lower() not in (
    "0", "false", "no",
)

rng = np.random.default_rng()


# ─────────────────────────────────────────
# Data
# ─────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df[df["icb"] != "England"].copy()
    icb_codes = df["icb"].astype("category").cat.codes.values
    beds = df["bed_occupancy"].values
    return icb_codes, beds


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

def build_model(icb_codes: np.ndarray, beds: np.ndarray) -> pm.Model:
    n_icb = len(np.unique(icb_codes))
    with pm.Model() as model:
        mu_national = pm.Normal("mu_national", 0, 1)
        sigma_icb   = pm.Exponential("sigma_icb", 1)
        icb_effect  = pm.Normal("icb_effect", 0, 1, shape=n_icb)
        sigma_obs   = pm.Exponential("sigma_obs", 5)

        latent_pressure = mu_national + icb_effect[icb_codes] * sigma_icb

        pm.Normal(
            "bed_obs",
            mu=85 + latent_pressure * 6,
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


def sample_posterior(
    model: pm.Model,
    idata: az.InferenceData,
    draws: int,
    tune: int,
) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            cores=4,
            target_accept=0.9,
            progressbar=False,
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


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def fit_pressure_model(df: pd.DataFrame, *, fast: bool | None = None):
    if fast is None:
        fast = _FAST
    draws, tune = (400, 400) if fast else (2000, 2000)

    icb_codes, beds = prepare_data(df)

    model = build_model(icb_codes, beds)

    idata = sample_prior(model)
    plot_prior_predictive(idata)

    idata = sample_posterior(model, idata, draws, tune)
    idata = sample_posterior_predictive(model, idata)

    plot_posterior_predictive(idata, beds)
    plot_residuals(idata, beds)

    print(az.summary(idata, var_names=["sigma_icb", "mu_national"]))

    return model, idata


def main():
    df = pd.read_csv("synthetic_nhs_pressure.csv")
    fit_pressure_model(df)


if __name__ == "__main__":
    main()