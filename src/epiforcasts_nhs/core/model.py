"""
Bayesian NHS system pressure model — AR(1) with seasonal decomposition.

Model structure:
    level[icb, 0]   ~ Normal(mu_national, sigma_icb)
    level[icb, t]   = rho * level[icb, t-1] + N(0, sigma_drift)
    season_effects  = [winter, spring, summer, autumn]  sum-to-zero
    bed_obs[i]      ~ Normal(85 + (level[icb, week] + season[week]) * 6, sigma_obs)
"""

from __future__ import annotations

import os
import tempfile

import pytensor
import pytensor.tensor as pt
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

import epiforcasts_nhs.core.constants as constants
from epiforcasts_nhs.core.utils import (
    check_season_alignment,
    current_season_from_enc,
    encode,
    prepare_data,
    pressure_summary,
)
from epiforcasts_nhs.config import DEFAULT_RANDOM_SEED

_FAST = os.environ.get("PRESSURE_MODEL_FAST", "1").strip().lower() not in ("0", "false", "no")

rng = np.random.default_rng(DEFAULT_RANDOM_SEED)


# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────

def build_model(enc: dict) -> pm.Model:
    """AR(1) local level model with shared four-season component."""
    icb_codes  = enc["icb_codes"]
    week_idx   = enc["week_idx"]
    season_idx = enc["season_idx"]
    beds       = enc["beds"]
    n_icb      = enc["n_icb"]
    n_weeks    = enc["n_weeks"]

    p  = constants.MODEL_PRIORS
    ls = constants.LATENT_SCALE

    with pm.Model() as model:
        mu_national = pm.Normal("mu_national", 0, p.mu_national_sd)
        sigma_icb   = pm.Exponential("sigma_icb", p.sigma_icb_rate)
        rho         = pm.Beta("rho", alpha=p.rho_alpha, beta=p.rho_beta)

        level_init  = pm.Normal("level_init", mu=mu_national, sigma=sigma_icb, shape=n_icb)
        sigma_drift = pm.Exponential("sigma_drift", lam=p.sigma_drift_lam)
        innovations = pm.Normal("innovations", 0, sigma_drift, shape=(n_weeks - 1, n_icb))

        levels_rest = pytensor.scan(
            fn=lambda innov, prev, rho: rho * prev + innov,
            sequences=innovations,
            outputs_info=level_init,
            non_sequences=rho,
            return_updates=False,
        )
        level = pm.Deterministic(
            "level",
            pt.concatenate([level_init[None, :], levels_rest], axis=0),
        )  # (n_weeks, n_icb)

        sigma_season   = pm.Exponential("sigma_season", lam=p.sigma_season_lam)
        season_raw     = pm.Normal("season_raw", 0, sigma_season, shape=constants.SEASONS.n_seasons)
        season_effects = pm.Deterministic("season_effects", season_raw - pt.mean(season_raw))

        latent    = level[week_idx, icb_codes] + season_effects[season_idx]
        sigma_obs = pm.Exponential("sigma_obs", lam=p.sigma_obs_lam)
        pm.Normal("bed_obs", mu=ls.baseline + latent * ls.scale, sigma=sigma_obs, observed=beds)

    return model


# ─────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────

def sample_prior(model: pm.Model, draws: int = 100) -> az.InferenceData:
    with model:
        return pm.sample_prior_predictive(draws=draws, random_seed=rng)


def sample_posterior(model: pm.Model, draws: int, tune: int) -> az.InferenceData:
    s = constants.MODEL_SAMPLING
    with model:
        return pm.sample(
            draws=draws, tune=tune,
            chains=s.chains, cores=s.cores,
            target_accept=s.target_accept,
            progressbar=True, compute_convergence_checks=False,
        )


def sample_posterior_predictive(model: pm.Model, idata: az.InferenceData) -> az.InferenceData:
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    return idata


# ─────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────

def save_posteriors(idata: az.InferenceData, enc: dict, path: str = "posteriors.nc") -> None:
    """Save posteriors to NetCDF with a Windows-safe atomic swap."""
    import sys
    from pathlib import Path
    from epiforcasts_nhs.config import STAGED_POSTERIORS

    final_path = Path(path)
    idata.attrs["icbs"]        = list(enc["categories"])
    idata.attrs["last_season"] = current_season_from_enc(enc)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=final_path.parent, prefix=".posteriors_tmp_", suffix=".nc")
    try:
        os.close(tmp_fd)
        idata.to_netcdf(tmp_path, engine="h5netcdf")

        if sys.platform == "win32":
            backup_path = final_path.with_suffix(".nc.bak")
            try:
                if final_path.exists():
                    os.replace(str(final_path), str(backup_path))
                os.replace(tmp_path, str(final_path))
                if backup_path.exists():
                    os.unlink(str(backup_path))
            except PermissionError:
                staged = final_path.parent / STAGED_POSTERIORS.name
                if os.path.exists(tmp_path):
                    os.replace(tmp_path, str(staged))
                raise RuntimeError(
                    f"posteriors.nc is locked. Staged at {staged} — retrying next cycle."
                )
        else:
            os.replace(tmp_path, str(final_path))

        print(f"Posteriors saved -> {final_path}  ({len(enc['categories'])} ICBs)")

    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def fit_pressure_model(df: pd.DataFrame, *, fast: bool | None = None) -> tuple:
    """
    Full inference pipeline: prepare → encode → build → sample → save.

    Parameters
    ----------
    df   : Raw weekly NHS data including England row.
    fast : Use fewer draws for quick iteration. Defaults to PRESSURE_MODEL_FAST env var.
    """
    if fast is None:
        fast = _FAST
    s = constants.MODEL_SAMPLING
    draws, tune = (s.draws_fast, s.tune_fast) if fast else (s.draws_full, s.tune_full)

    data = prepare_data(df)
    enc  = encode(data)

    print(f"Rows:  {len(data)}")
    print(f"Weeks: {data['week'].min()}-{data['week'].max()}")
    print(f"ICBs:  {list(enc['categories'])}")
    check_season_alignment(enc, data)

    model = build_model(enc)
    idata = sample_prior(model)
    idata = sample_posterior(model, draws, tune)
    idata = sample_posterior_predictive(model, idata)

    print("\nConvergence summary:")
    print(az.summary(idata, var_names=[
        "mu_national", "sigma_icb", "rho", "sigma_drift", "sigma_season", "sigma_obs",
    ]))

    current_season = current_season_from_enc(enc)
    print(f"\nCurrent season: {constants.SEASONS.names[current_season]}")

    summary = pressure_summary(idata, enc, current_season=current_season)
    print("\nClinical summary (ranked by current pressure):")
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    save_posteriors(idata, enc)
    return model, idata, enc


def main():
    from epiforcasts_nhs.config import WEEKLY_CSV
    df = pd.read_csv(WEEKLY_CSV)
    fit_pressure_model(df)


if __name__ == "__main__":
    main()
