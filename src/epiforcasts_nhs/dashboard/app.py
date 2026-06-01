import os
import io
import time
from datetime import datetime

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import arviz as az

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.dashboard.utils import (
    DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN,
    DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,
    DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,
    DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,
    SEASON_NAMES,
    THRESHOLD_BASELINE,
    THRESHOLD_CONCERN,
    THRESHOLD_ELEVATED,
    credible_triplet,
    current_season_index,
    level_only_samples,
    pressure_index_samples,
    resolve_icb_index,
    seasonal_effect_samples,
)
from epiforcasts_nhs.dashboard.plots import (
    fig_pressure_question,
    fig_pressure_trajectory,
    fig_seasonal_effects,
    fig_clinical_summary,
)

WEEKLY_CSV    = "synthetic_nhs_pressure.csv"
POSTERIORS_NC = "posteriors.nc"
_POLL_INTERVAL_S = 10


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _risk_band(
    p_elevated: float,
    p_concern: float,
    *,
    pe_hi: float,
    pc_hi: float,
    pe_med: float,
    pc_med: float,
) -> tuple[str, str]:
    if p_elevated >= pe_hi or p_concern >= pc_hi:
        return "Elevated", "Prioritise review of capacity, flow, and escalation plans (indicative only)."
    if p_elevated >= pe_med or p_concern >= pc_med:
        return "Medium", "Worth closer monitoring; corroborate with local intelligence."
    return "Low", "No strong signal of unusually high modelled pressure; stay vigilant to new data."


def _england_pressure_samples(idata: az.InferenceData) -> np.ndarray:
    """Mean pressure index across all ICBs — used for the England aggregate tab."""
    icbs = list(idata.attrs.get("icbs", []))
    season_now = current_season_index(idata)
    level_f    = idata.posterior["level"].values.reshape(
        -1,
        idata.posterior["level"].shape[2],
        idata.posterior["level"].shape[3],
    )
    season_f = idata.posterior["season_effects"].values.reshape(-1, 4)
    # Mean level across ICBs per sample, plus current season effect
    mean_level = level_f[:, -1, :].mean(axis=1)
    return (mean_level + season_f[:, season_now]).astype(float)


# ─────────────────────────────────────────
# NHS theme
# ─────────────────────────────────────────

def _inject_nhs_theme() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
:root {
    --nhs-blue:#005eb8; --nhs-dark-blue:#003087;
    --nhs-white:#ffffff; --ink-strong:#1d1d1d;
    --ink-soft:#4f4f4f; --border-soft:#d8dde0;
}
.stApp { background:linear-gradient(180deg,#f5f9ff 0%,#ffffff 35%);
         font-family:'Public Sans','Segoe UI',sans-serif; }
h1,h2,h3 { color:var(--nhs-dark-blue); letter-spacing:-.01em; }
.hero { background:linear-gradient(120deg,var(--nhs-dark-blue) 0%,var(--nhs-blue) 70%);
        color:#fff; border-radius:16px; padding:1.1rem 1.2rem;
        margin-bottom:.9rem; box-shadow:0 6px 24px rgba(0,48,135,.2); }
.hero h1 { color:#fff; margin:0; font-size:1.65rem; }
.hero p  { margin:.45rem 0 0; opacity:.96; font-size:.96rem; }
.section-card { background:#fff; border:1px solid var(--border-soft);
                border-radius:14px; padding:.95rem 1rem; margin-bottom:.75rem;
                box-shadow:0 2px 10px rgba(0,48,135,.06); }
.section-title { color:var(--nhs-dark-blue); font-weight:700; margin:0 0 .35rem; }
.section-text  { color:var(--ink-soft); font-size:.94rem; margin:0; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--border-soft);
    border-radius:12px; padding:.55rem .7rem; }
[data-testid="stSidebar"] { border-right:1px solid var(--border-soft); background:#f8fbff; }
</style>""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────

@st.cache_data(show_spinner="Loading weekly panel…")
def load_weekly_data(path: str, _file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading cached posteriors…")
def load_posteriors(posteriors_path: str, posteriors_mtime: float) -> az.InferenceData:
    idata = az.from_netcdf(posteriors_path, engine="h5netcdf")
    buf   = io.BytesIO()
    idata.to_netcdf(buf, engine="h5netcdf")
    buf.seek(0)
    return az.from_netcdf(buf, engine="h5netcdf")


@st.cache_data(show_spinner="Loading cached statistics…")
def load_summary_stats(
    _cache_manager: CacheManager,
    posteriors_mtime: float,
) -> dict:
    return _cache_manager.load_summary_stats()


# ─────────────────────────────────────────
# Page content for one geography
# ─────────────────────────────────────────

def _render_geography(
    *,
    label: str,
    area_caption: str,
    samples: np.ndarray,
    idata: az.InferenceData,
    traj_icbs: list[str],
    full_panel_df: pd.DataFrame,
    summary_icb_filter: str | None,
    credible_mass: float,
    show_median_line: bool,
    season_now_str: str,
    pe_hi: float,
    pc_hi: float,
    pe_med: float,
    pc_med: float,
) -> None:
    """
    Render the full pressure view for one geography.

    Parameters
    ----------
    label             : display name shown in headings
    area_caption      : short caption used below the evidence-distribution plot
    samples           : posterior pressure-index samples for this geography
    idata             : full InferenceData object
    traj_icbs         : ICB names to plot trajectories for.
                        Pass [icb_name] for a specific ICB, all ICBs for England.
    full_panel_df     : the complete weekly panel (all ICBs, no England row).
                        Each trajectory is extracted from this by ICB name.
    summary_icb_filter: passed directly to fig_clinical_summary as icb_filter.
                        None → show all ICBs (England). str → show only that ICB.
    """
    p_above_baseline = float(np.mean(samples > THRESHOLD_BASELINE))
    p_above_concern  = float(np.mean(samples > THRESHOLD_CONCERN))
    p_above_elevated = float(np.mean(samples > THRESHOLD_ELEVATED))
    lo, mid, hi      = credible_triplet(samples, credible_mass)
    cred_pct         = int(credible_mass * 100)

    risk_label, risk_hint = _risk_band(
        p_above_elevated, p_above_concern,
        pe_hi=pe_hi, pc_hi=pc_hi, pe_med=pe_med, pc_med=pc_med,
    )

    st.markdown("""
<div class="section-card">
  <p class="section-title">Current Interpretation</p>
  <p class="section-text">Use this view to judge whether pressure signals are strengthening,
  how uncertain they are, and where local operational review should focus first.</p>
</div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Summary — Current Pressure Risk (Indicative)")
        c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1.2])
        with c0:
            st.markdown(f"### {risk_label}")
            st.caption(risk_hint)
        with c1:
            st.metric("Chance pressure exceeds baseline reference",
                      f"{p_above_baseline:.0%}",
                      help="Posterior probability above baseline reference.")
        with c2:
            st.metric("Chance pressure exceeds concern reference",
                      f"{p_above_concern:.0%}")
        with c3:
            st.metric("Chance pressure exceeds high reference",
                      f"{p_above_elevated:.0%}")

        st.markdown(
            f"**{cred_pct}% plausible range** (equal-tailed) for the pressure index: "
            f"**{lo:.2f}** to **{hi:.2f}** (median **{mid:.2f}**). "
            f"Current season: **{season_now_str}** — seasonal contribution is included in these estimates."
        )

    with st.expander("How this system updates its view of pressure", expanded=False):
        st.markdown("""
This system starts with a **baseline expectation** of NHS system pressure, informed by
historical patterns and prior knowledge.

As new data arrive the system:
- **Assesses** how consistent new signals are with previous evidence
- **Updates** its belief **gradually**, rather than jumping to conclusions
- **Separates** underlying structural pressure from seasonal effects
- **Maintains uncertainty**, reflecting data gaps, noise, and model simplifications

The pressure index shown includes both the underlying AR(1) level and the current season's
contribution. Direction of travel is computed on the seasonal-adjusted level only.
        """)

    st.markdown("---")
    st.markdown("### Evidence Distribution")
    st.caption(
        "Question answered: **How much posterior weight sits above routine vs elevated pressure?** "
        "Dashed lines are **demo reference** cut-points — not national operational thresholds. "
        f"Total pressure shown includes the **{season_now_str}** seasonal adjustment."
    )

    fig = fig_pressure_question(
        samples, credible_mass=credible_mass, show_median_line=show_median_line,
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption(area_caption)

    st.markdown("---")
    st.markdown("### Pressure Trajectory")
    st.caption(
        "Kalman-filtered estimate of underlying pressure over time — **seasonal component removed**. "
        "The shaded band is the 90% filtered CI, which is wide early "
        "(little evidence) and narrows as data accumulates."
    )

    # traj_icbs is always an explicit list — no conditional needed here.
    for traj_icb in traj_icbs:
        icb_rows = full_panel_df[full_panel_df["icb"] == traj_icb]
        traj_fig = fig_pressure_trajectory(idata, icb_rows, traj_icb)
        if traj_fig is not None:
            st.pyplot(traj_fig, clear_figure=True)
            plt.close(traj_fig)
        else:
            st.warning(f"Trajectory unavailable for {traj_icb}.")

    st.markdown("---")
    st.markdown("### Clinical Summary")

    # summary_icb_filter is always passed explicitly — no conditional needed here.
    if summary_icb_filter is None:
        st.caption(
            "All ICBs ranked by current modelled pressure (including seasonal adjustment). "
            "Colour: red = elevated, amber = medium, green = low. "
            "Direction of travel uses the seasonal-adjusted underlying level. "
            "Data derived from all ICBs."
        )
    else:
        st.caption(
            f"Current pressure, risk probability, and direction of travel for "
            f"**{summary_icb_filter}** only."
        )

    summary_fig = fig_clinical_summary(idata, icb_filter=summary_icb_filter)
    st.pyplot(summary_fig, clear_figure=True)
    plt.close(summary_fig)
    st.caption(
        "Reference lines: amber dashed = 95% occupancy, red dashed = 100% occupancy. "
        "Risk probability thresholds can be adjusted in the Advanced sidebar panel."
    )

    st.info(
        "**Uncertainty:** Spreads reflect sparse weeks, seasonal estimation uncertainty, "
        "model simplifications, and reporting noise. Use alongside operational intelligence."
    )


# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────

st.set_page_config(layout="wide", page_title="System pressure — early signal (demo)")
_inject_nhs_theme()
st.markdown("""
<div class="hero">
  <h1>System Pressure Early Signal</h1>
  <p>Decision-support view for Trust and ICB teams. Probabilistic, uncertainty-aware, and explicitly non-clinical.</p>
</div>""", unsafe_allow_html=True)

# Staged file pickup (Windows atomic swap fallback)
_staged = os.path.join(os.path.dirname(POSTERIORS_NC) or ".", ".posteriors_new.nc")
if os.path.isfile(_staged):
    try:
        _backup = POSTERIORS_NC + ".bak"
        if os.path.isfile(POSTERIORS_NC):
            os.replace(POSTERIORS_NC, _backup)
        os.replace(_staged, POSTERIORS_NC)
        if os.path.isfile(_backup):
            os.unlink(_backup)
    except PermissionError:
        pass

# ── Mtime-based auto-rerun ─────────────────────────────────────────────────
# Detect a new posterior the moment its mtime changes, then call st.rerun()
# so the UI refreshes without any human interaction.
posteriors_mtime = (
    os.path.getmtime(POSTERIORS_NC) if os.path.isfile(POSTERIORS_NC) else 0.0
)
csv_mtime = os.path.getmtime(WEEKLY_CSV) if os.path.isfile(WEEKLY_CSV) else 0.0

_last_pm = st.session_state.get("last_posteriors_mtime", -1.0)
if posteriors_mtime != _last_pm and _last_pm != -1.0:
    # File changed since last render — bust caches and immediately rerun
    load_posteriors.clear()
    load_summary_stats.clear()
    load_weekly_data.clear()
    st.session_state["last_posteriors_mtime"] = posteriors_mtime
    st.rerun()
st.session_state["last_posteriors_mtime"] = posteriors_mtime

# ── Data ──────────────────────────────────────────────────────────────────
df    = load_weekly_data(WEEKLY_CSV, csv_mtime)
cache = CacheManager(posteriors_path=POSTERIORS_NC)

if not cache.is_valid():
    st.error(
        "Posterior cache is not ready\n\n"
        "**Fix:**\n```bash\nepiforcasts-daemon --once\n```"
    )
    # Poll and retry
    time.sleep(_POLL_INTERVAL_S)
    st.rerun()

# Explicit cache bust when file changes (first render after mtime flip)
if st.session_state.get("_cache_busted_for") != posteriors_mtime:
    load_posteriors.clear()
    load_summary_stats.clear()
    st.session_state["_cache_busted_for"] = posteriors_mtime

idata = load_posteriors(POSTERIORS_NC, posteriors_mtime)
stats = load_summary_stats(cache, posteriors_mtime)

season_now     = current_season_index(idata)
season_now_str = SEASON_NAMES[season_now]

# ─────────────────────────────────────────
# Sidebar — display settings only
# ─────────────────────────────────────────

icbs_in_posterior = list(idata.attrs.get("icbs", []))

with st.sidebar:
    st.header("Scope")
    geo_options = ["England"] + icbs_in_posterior
    selected_geo = st.selectbox(
        "Geography",
        geo_options,
        help="'England' aggregates across all ICBs. Select an ICB for its data only.",
    )

    st.header("Display")
    credible_choice = st.radio(
        "Credible band on chart & summary",
        options=["90% (default)", "50% (tighter)"],
        index=0,
    )
    credible_mass    = 0.9 if credible_choice.startswith("90") else 0.5
    show_median_line = st.checkbox("Show median line on chart", value=True)

    if posteriors_mtime > 0:
        updated_str = datetime.fromtimestamp(posteriors_mtime).strftime(
            "%d %b %Y %H:%M:%S"
        )
        st.caption(f"Posteriors last updated: **{updated_str}**")
        st.caption(f"Current season: **{season_now_str}**")
        st.caption(f"Data weeks: {int(df[df['icb'] != 'England']['week'].max())}")

    with st.expander("Assumptions & limits", expanded=False):
        st.markdown("""
- Reference lines are **demo cut-points** on a unitless model index — not operational thresholds.
- Risk band is a **UI heuristic** — co-design with ops to tune cut-offs.
- Seasonal component assumes four equal 13-week seasons. Verify `winter_start_week` alignment.
        """)

    with st.expander("Advanced: risk band heuristics", expanded=False):
        st.caption("Thresholds on posterior probabilities for the summary label only.")
        c1, c2 = st.columns(2)
        with c1:
            pe_hi  = st.number_input("Elevated if P(high) >=",  0.0, 1.0,
                                     DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,  0.01, key="pe_hi")
            pe_med = st.number_input("Medium if P(high) >=",    0.0, 1.0,
                                     DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,    0.01, key="pe_med")
        with c2:
            pc_hi  = st.number_input("... or P(concern) >=",      0.0, 1.0,
                                     DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN, 0.01, key="pc_hi")
            pc_med = st.number_input("... or P(concern) >=",      0.0, 1.0,
                                     DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,   0.01, key="pc_med")

# ─────────────────────────────────────────
# Geography view  (atomic placeholder — replaced in full on each rerun)
# ─────────────────────────────────────────

_all_icb_df   = df[df["icb"] != "England"]
_main_area    = st.empty()

with _main_area.container():

    if selected_geo == "England":
        samples               = _england_pressure_samples(idata)
        render_traj_icbs      = list(icbs_in_posterior)  # copy — one per ICB
        render_summary_filter = None
        render_area_caption   = (
            "Area: All ICBs (England aggregate). Values to the right of dashed "
            "references indicate stronger posterior support for elevated modelled pressure."
        )

        # Seasonal pattern is a national (shared) component — England view only.
        st.markdown("---")
        st.markdown("### Seasonal Pattern")
        st.caption(
            "Posterior estimates of how each season shifts bed occupancy relative to the "
            "annual mean. Shared across all ICBs — applies equally regardless of geography."
        )
        season_fig = fig_seasonal_effects(idata)
        st.pyplot(season_fig, clear_figure=True)
        plt.close(season_fig)

    else:
        try:
            icb_idx = resolve_icb_index(idata, selected_geo)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        if df[df["icb"] == selected_geo].empty:
            st.error(f"No panel rows found for: {selected_geo}")
            st.stop()
        samples               = pressure_index_samples(idata, icb_idx)
        render_traj_icbs      = [selected_geo]   # exactly one trajectory
        render_summary_filter = str(selected_geo)
        render_area_caption   = (
            f"Area: {selected_geo}. Values to the right of dashed references "
            "indicate stronger posterior support for elevated modelled pressure."
        )

    _render_geography(
        label=selected_geo,
        area_caption=render_area_caption,
        samples=samples,
        idata=idata,
        traj_icbs=render_traj_icbs,
        full_panel_df=_all_icb_df,
        summary_icb_filter=render_summary_filter,
        credible_mass=credible_mass,
        show_median_line=show_median_line,
        season_now_str=season_now_str,
        pe_hi=pe_hi, pc_hi=pc_hi, pe_med=pe_med, pc_med=pc_med,
    )

# ─────────────────────────────────────────
# Auto-rerun polling loop
# ─────────────────────────────────────────
# Sleep briefly then restart the script so the mtime check at the top
# fires again. If the posterior has been updated, the cache is busted
# and the UI reflects fresh results with no human interaction needed.
time.sleep(_POLL_INTERVAL_S)
st.rerun()
