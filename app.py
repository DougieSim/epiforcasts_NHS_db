import os

# Headless / Cloud-friendly backend before pyplot (faster, no GUI toolkit on servers).
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import arviz as az

from cache_manager import CacheManager
from bayesian_pressure_model import (
    current_pressure_samples,
    direction_of_travel,
    pressure_summary,
)
from dashboard_shared import (
    DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN,
    DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,
    DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,
    DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,
    THRESHOLD_BASELINE,
    THRESHOLD_CONCERN,
    THRESHOLD_ELEVATED,
    credible_triplet,
    pressure_index_samples,
    resolve_icb_index,
)

WEEKLY_CSV    = "synthetic_nhs_pressure.csv"
POSTERIORS_NC = "posteriors.nc"


def _risk_band(
    p_elevated: float,
    p_concern: float,
    *,
    pe_hi: float,
    pc_hi: float,
    pe_med: float,
    pc_med: float,
) -> tuple[str, str]:
    """Map probabilities to an indicative label — heuristic only."""
    if p_elevated >= pe_hi or p_concern >= pc_hi:
        return "Elevated", "Prioritise review of capacity, flow, and escalation plans (indicative only)."
    if p_elevated >= pe_med or p_concern >= pc_med:
        return "Medium", "Worth closer monitoring; corroborate with local intelligence."
    return "Low", "No strong signal of unusually high modelled pressure; stay vigilant to new data."


def _inject_nhs_theme() -> None:
    """Apply a modern NHS-inspired visual system for clarity and consistency."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --nhs-blue: #005eb8;
    --nhs-dark-blue: #003087;
    --nhs-light-blue: #e8f3ff;
    --nhs-white: #ffffff;
    --ink-strong: #1d1d1d;
    --ink-soft: #4f4f4f;
    --border-soft: #d8dde0;
}

.stApp {
    background: linear-gradient(180deg, #f5f9ff 0%, #ffffff 35%);
    color: var(--ink-strong);
    font-family: 'Public Sans', 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: var(--nhs-dark-blue);
    letter-spacing: -0.01em;
}

.hero {
    background: linear-gradient(120deg, var(--nhs-dark-blue) 0%, var(--nhs-blue) 70%);
    color: var(--nhs-white);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.9rem;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 6px 24px rgba(0, 48, 135, 0.2);
}

.hero h1 {
    color: var(--nhs-white);
    margin: 0;
    font-size: 1.65rem;
}

.hero p {
    margin: 0.45rem 0 0;
    opacity: 0.96;
    font-size: 0.96rem;
}

.section-card {
    background: var(--nhs-white);
    border: 1px solid var(--border-soft);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 10px rgba(0, 48, 135, 0.06);
}

.section-title {
    color: var(--nhs-dark-blue);
    font-weight: 700;
    margin: 0 0 0.35rem;
}

.section-text {
    color: var(--ink-soft);
    font-size: 0.94rem;
    margin: 0;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 0.55rem 0.7rem;
    box-shadow: 0 1px 5px rgba(0, 48, 135, 0.05);
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-soft);
    background: #f8fbff;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _plot_pressure_question(
    samples: np.ndarray,
    *,
    credible_mass: float,
    show_median_line: bool,
) -> plt.Figure:
    lo, mid, hi = credible_triplet(samples, credible_mass)
    pct_label = f"{int(credible_mass * 100)}% plausible range"

    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    ax.hist(
        samples,
        bins=40,
        density=True,
        alpha=0.78,
        color="#1d4ed8",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvspan(lo, hi, alpha=0.15, color="#1e3a8a", label=pct_label)
    ax.axvline(
        THRESHOLD_BASELINE,
        color="#64748b",
        linestyle="--",
        linewidth=1.5,
        label="Baseline reference (demo)",
    )
    ax.axvline(
        THRESHOLD_CONCERN,
        color="#d97706",
        linestyle="--",
        linewidth=1.5,
        label="Concern reference (demo)",
    )
    ax.axvline(
        THRESHOLD_ELEVATED,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.5,
        label="High pressure reference (demo)",
    )
    if show_median_line:
        ax.axvline(mid, color="#0f172a", linestyle="-", linewidth=1.0, alpha=0.85, label="Median")

    ax.set_xlabel(
        "System pressure index (modelled, unitless — not an NHS operational metric)"
    )
    ax.set_ylabel("Relative plausibility")
    ax.set_title(
        "Where does the evidence put system pressure for this area?",
        fontsize=13,
        pad=10,
    )
    ax.set_xlim(
        min(samples.min(), THRESHOLD_BASELINE) - 0.35,
        max(samples.max(), THRESHOLD_ELEVATED) + 0.35,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


# ── Inline figure builders ───────────────────────────────────────────────────
#
# These mirror the plot_* functions in bayesian_pressure_model.py but return
# figures without calling plt.show(), so Streamlit can render them inline.

def _fig_pressure_trajectory(
    idata: az.InferenceData,
    df: pd.DataFrame,
    icb: str,
) -> plt.Figure:
    """Posterior level trajectory for one ICB with 80% CI and observed data."""
    post  = idata.posterior
    level = post["level"].values                         # (chains, draws, n_weeks, n_icb)
    icbs  = list(idata.attrs.get("icbs", []))

    if icb not in icbs:
        return None
    i = icbs.index(icb)

    level_flat = level.reshape(-1, level.shape[2], level.shape[3])  # (S, n_weeks, n_icb)
    icb_level  = level_flat[:, :, i]                                 # (S, n_weeks)
    n_weeks    = icb_level.shape[1]

    lo  = np.percentile(icb_level, 10, axis=0)
    mid = np.percentile(icb_level, 50, axis=0)
    hi  = np.percentile(icb_level, 90, axis=0)

    # Derive week axis from the posterior shape, not the dataframe length.
    # The posterior covers exactly the weeks the model was trained on —
    # week 0..n_weeks-1 on a 0-based index. Map back to real week numbers
    # using the minimum week in the ICB's data.
    icb_df  = df[df["icb"] == icb].sort_values("week")
    min_week = int(icb_df["week"].min())
    weeks   = np.arange(min_week, min_week + n_weeks)

    # Observed data aligned to the same week range
    obs = icb_df[icb_df["week"].isin(weeks)]

    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.fill_between(weeks, 85 + lo * 6, 85 + hi * 6,
                    alpha=0.3, color="#1d4ed8", label="80% CI")
    ax.plot(weeks, 85 + mid * 6,
            color="#1d4ed8", linewidth=1.8, label="Posterior median")
    ax.scatter(obs["week"], obs["bed_occupancy"],
               s=8, color="#0f172a", alpha=0.45, label="Observed", zorder=3)
    ax.axhline(95,  color="#d97706", linestyle="--", linewidth=1, alpha=0.8, label="95% reference")
    ax.axhline(100, color="#b91c1c", linestyle="--", linewidth=1, alpha=0.8, label="100% reference")

    # Mark the most recent week clearly
    ax.axvline(weeks[-1], color="#64748b", linestyle=":", linewidth=1.2,
               alpha=0.8, label=f"Latest (week {weeks[-1]})")

    ax.set_ylabel("Bed occupancy (%)")
    ax.set_xlabel("Week")
    ax.set_title(f"Posterior pressure trajectory — {icb}", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def _fig_direction_of_travel(
    idata: az.InferenceData,
    lookback_weeks: int = 4,
) -> plt.Figure:
    """Posterior distribution of pressure change over the last N weeks, all ICBs."""
    icbs  = list(idata.attrs.get("icbs", []))
    level = idata.posterior["level"].values
    level_flat = level.reshape(-1, level.shape[2], level.shape[3])

    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")

    colors = ["#1d4ed8", "#0f766e", "#b45309", "#7c3aed", "#be123c", "#15803d"]
    for i, icb_name in enumerate(icbs):
        current  = level_flat[:, -1, i]
        previous = level_flat[:, -lookback_weeks, i]
        dot      = (current - previous) * 6          # convert to % occupancy
        p_rising = float(np.mean(dot > 0))
        short    = icb_name.replace("NHS ", "").replace(" ICB", "")
        label    = f"{short}  (P(↑)={p_rising:.0%})"
        ax.hist(dot, bins=50, density=True, alpha=0.55,
                color=colors[i % len(colors)], label=label)

    ax.axvline(0, color="#0f172a", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Change in bed occupancy (%) over last {lookback_weeks} weeks")
    ax.set_title("Direction of travel — all ICBs", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def _fig_clinical_summary(
    idata: az.InferenceData,
    icb_filter: str | None = None,
) -> plt.Figure:
    """
    Three-panel clinical summary: pressure, risk probability, direction of travel.

    Parameters
    ----------
    icb_filter : str or None
        If provided, show only this ICB. If None, show all ICBs.
    """
    icbs       = list(idata.attrs.get("icbs", []))
    level      = idata.posterior["level"].values
    level_flat = level.reshape(-1, level.shape[2], level.shape[3])

    rows = []
    for i, icb_name in enumerate(icbs):
        if icb_filter is not None and icb_name != icb_filter:
            continue
        samples  = level_flat[:, -1, i]
        dot      = (level_flat[:, -1, i] - level_flat[:, -4, i]) * 6
        rows.append(dict(
            icb=icb_name,
            median=float(85 + np.median(samples) * 6),
            lo=float(85 + np.percentile(samples, 10) * 6),
            hi=float(85 + np.percentile(samples, 90) * 6),
            p_high=float(np.mean(samples > 1.1)),
            dot=float(np.median(dot)),
        ))

    summary = pd.DataFrame(rows).sort_values("median", ascending=False)
    short_names = summary["icb"].str.replace("NHS ", "").str.replace(" ICB", "")
    y = np.arange(len(summary))

    colors = ["#b91c1c" if p > 0.25 else "#d97706" if p > 0.08 else "#15803d"
              for p in summary["p_high"]]

    n_rows = len(summary)
    fig_height = 2.5 if n_rows == 1 else max(3, n_rows * 0.7 + 1.5)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, fig_height),
        layout="constrained",
    )

    # Panel 1 — current pressure
    ax = axes[0]
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
    ax.set_title("Current pressure")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2 — risk probability
    ax = axes[1]
    ax.barh(y, summary["p_high"], color=colors, alpha=0.75, height=0.6)
    ax.axvline(0.25, color="#b91c1c", linestyle="--", linewidth=1, label="Elevated (0.25)")
    ax.axvline(0.08, color="#d97706", linestyle="--", linewidth=1, label="Medium (0.08)")
    ax.set_yticks(y); ax.set_yticklabels([""] * len(summary))
    ax.set_xlabel("P(pressure above high reference)")
    ax.set_title("Risk probability")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3 — direction of travel
    ax = axes[2]
    dot_colors = ["#b91c1c" if d > 0.5 else "#15803d" if d < -0.5 else "#64748b"
                  for d in summary["dot"]]
    ax.barh(y, summary["dot"], color=dot_colors, alpha=0.75, height=0.6)
    ax.axvline(0, color="#0f172a", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels([""] * len(summary))
    ax.set_xlabel("Pressure change (% occ, last 4 weeks)")
    ax.set_title("Direction of travel")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    return fig


# ── Cached loaders ────────────────────────────────────────────────────────────
#
# Both posteriors and summary stats use _posteriors_mtime as a cache key.
# When the daemon atomically replaces posteriors.nc, the mtime changes,
# Streamlit sees a new key and reloads — the old cached objects are discarded.
# The leading underscore on _cache_manager tells Streamlit not to hash it.

@st.cache_data(show_spinner="Loading weekly panel…")
def load_weekly_data(path: str, _file_mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading cached posteriors…")
def load_posteriors(posteriors_path: str, posteriors_mtime: float) -> az.InferenceData:
    """
    Load posteriors keyed purely on path + mtime.

    mtime is the only cache key — when the daemon writes a new file
    the mtime changes, Streamlit discards the old cache entry and
    reloads. Round-tripping through BytesIO ensures the h5netcdf file
    handle is fully closed before returning.
    """
    import io
    idata = az.from_netcdf(posteriors_path, engine="h5netcdf")
    buf = io.BytesIO()
    idata.to_netcdf(buf, engine="h5netcdf")
    buf.seek(0)
    return az.from_netcdf(buf, engine="h5netcdf")


@st.cache_data(show_spinner="Loading cached statistics…")
def load_summary_stats(_cache_manager: CacheManager, posteriors_mtime: float) -> dict:
    return _cache_manager.load_summary_stats()


# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="System pressure — early signal (demo)")
_inject_nhs_theme()

# ── Auto-rerun ────────────────────────────────────────────────────────────────
# Inject an HTML meta-refresh tag so the browser reloads the page every
# N seconds. This is the most reliable cross-version approach — it works
# regardless of Streamlit version and doesn't require extra packages.
# On each reload, the mtime check below determines whether to bust the cache.
_POLL_INTERVAL_S = 30
st.markdown(
    f'<meta http-equiv="refresh" content="{_POLL_INTERVAL_S}">',
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero">
  <h1>System Pressure Early Signal</h1>
  <p>Decision-support view for Trust and ICB teams. Probabilistic, uncertainty-aware, and explicitly non-clinical.</p>
</div>
    """,
    unsafe_allow_html=True,
)

# ── Data ──────────────────────────────────────────────────────────────────────

csv_mtime = os.path.getmtime(WEEKLY_CSV) if os.path.isfile(WEEKLY_CSV) else 0.0
df = load_weekly_data(WEEKLY_CSV, csv_mtime)

# ── Cache ─────────────────────────────────────────────────────────────────────

# On Windows, if the daemon couldn't atomically replace posteriors.nc
# while it was open, it stages the new file as .posteriors_new.nc.
# Pick it up here before the cache manager validates.
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
        pass  # still locked — serve old copy, retry next rerun

cache = CacheManager(posteriors_path=POSTERIORS_NC)

if not cache.is_valid():
    st.error(
        f"❌ Posterior cache is not ready\n\n"
        f"**Status:**\n"
        f"- Posterior available: {cache.is_posterior_available()}\n"
        f"- Cache warm: {cache.is_cache_warm()}\n\n"
        f"**Fix:**\n"
        f"```bash\n"
        f"python inference_daemon.py --once\n"
        f"```"
    )
    st.stop()

# mtime of posteriors.nc — changes every time the daemon writes a new file,
# which invalidates the Streamlit cache and triggers a fresh load.
posteriors_mtime = (
    os.path.getmtime(POSTERIORS_NC)
    if os.path.isfile(POSTERIORS_NC)
    else 0.0
)

idata = load_posteriors(POSTERIORS_NC, posteriors_mtime)
stats = load_summary_stats(cache, posteriors_mtime)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Scope")
    icb = st.selectbox(
        "Geography",
        df["icb"].unique(),
        help="Synthetic demo panel; organisational names for grounding only.",
    )

    st.header("Display")
    credible_choice = st.radio(
        "Credible band on chart & summary",
        options=["90% (default)", "50% (tighter)"],
        index=0,
        help="Equal-tailed interval from the posterior — not a prediction interval.",
    )
    credible_mass = 0.9 if credible_choice.startswith("90") else 0.5

    show_median_line = st.checkbox(
        "Show median line on chart",
        value=True,
        help="Turn off for a more probability-first / interval-first visual.",
    )

    # Show when the posteriors were last updated so the user knows how fresh
    # the data is without needing to inspect files directly.
    if posteriors_mtime > 0:
        from datetime import datetime
        updated_str = datetime.fromtimestamp(posteriors_mtime).strftime(
            "%d %b %Y %H:%M"
        )
        st.caption(f"Posteriors last updated: **{updated_str}**")

    with st.expander("Assumptions & limits (serious deployments)", expanded=False):
        st.markdown(
            """
- **Reference line positions** (baseline / concern / high) are **fixed demo cut-points** on a **unitless model index**. They are **not** national operational thresholds.
- **Risk band** (Low / Medium / Elevated) is a **UI heuristic**, **not** validated against real escalation policy — **co-design with ops** to tune cut-offs (see Advanced).
- The index is **not calibrated** to NHS units; lean on **probabilities** and **credible intervals**, not the absolute value of the index.
            """
        )

    with st.expander("Advanced: risk band heuristics (probabilities)", expanded=False):
        st.caption("Thresholds on **posterior probabilities** for the summary label only.")
        c1, c2 = st.columns(2)
        with c1:
            pe_hi = st.number_input(
                "Elevated if P(high) ≥",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_RISK_ELEVATED_TIER_MIN_P_HIGH,
                step=0.01, key="pe_hi",
            )
            pe_med = st.number_input(
                "Medium if P(high) ≥",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_RISK_MEDIUM_TIER_MIN_P_HIGH,
                step=0.01, key="pe_med",
            )
        with c2:
            pc_hi = st.number_input(
                "… or P(concern) ≥",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_RISK_ELEVATED_TIER_MIN_P_CONCERN,
                step=0.01, key="pc_hi",
            )
            pc_med = st.number_input(
                "… or P(concern) ≥",
                min_value=0.0, max_value=1.0,
                value=DEFAULT_RISK_MEDIUM_TIER_MIN_P_CONCERN,
                step=0.01, key="pc_med",
            )

# ── ICB resolution ────────────────────────────────────────────────────────────

subset = df[df["icb"] == icb]
if subset.empty:
    st.error(f"No panel rows found for selected geography: {icb}")
    st.stop()

try:
    icb_idx = resolve_icb_index(idata, icb)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ── Posterior samples ─────────────────────────────────────────────────────────

samples = pressure_index_samples(idata, icb_idx)

p_above_baseline = float(np.mean(samples > THRESHOLD_BASELINE))
p_above_concern  = float(np.mean(samples > THRESHOLD_CONCERN))
p_above_elevated = float(np.mean(samples > THRESHOLD_ELEVATED))
lo, mid, hi = credible_triplet(samples, credible_mass)
cred_pct = int(credible_mass * 100)

risk_label, risk_hint = _risk_band(
    p_above_elevated,
    p_above_concern,
    pe_hi=pe_hi,
    pc_hi=pc_hi,
    pe_med=pe_med,
    pc_med=pc_med,
)

# ── Main content ──────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="section-card">
  <p class="section-title">Current Interpretation</p>
  <p class="section-text">Use this view to judge whether pressure signals are strengthening, how uncertain they are, and where local operational review should focus first.</p>
</div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Summary — Current Pressure Risk (Indicative)")
    c0, c1, c2, c3 = st.columns([1.1, 1, 1, 1.2])
    with c0:
        st.markdown(f"### {risk_label}")
        st.caption(risk_hint)
    with c1:
        st.metric(
            label="Chance pressure exceeds baseline reference",
            value=f"{p_above_baseline:.0%}",
            help="Posterior probability that the modelled index sits above the baseline reference line.",
        )
    with c2:
        st.metric(
            label="Chance pressure exceeds concern reference",
            value=f"{p_above_concern:.0%}",
        )
    with c3:
        st.metric(
            label="Chance pressure exceeds high reference",
            value=f"{p_above_elevated:.0%}",
            help="Stricter threshold on the same index — useful as a stronger (still non-clinical) flag.",
        )

    st.markdown(
        f"**{cred_pct}% plausible range** (equal-tailed) for the system pressure index: "
        f"**{lo:.2f}** to **{hi:.2f}** (median **{mid:.2f}**). "
        "This describes **where posterior mass sits**, not NHS-calibrated units. "
        "It is not a confidence interval on a future outcome."
    )

with st.expander("How this system updates its view of pressure", expanded=False):
    st.markdown(
        """
This system starts with a **baseline expectation** of NHS system pressure, informed by historical patterns and prior knowledge.

As new data arrive (e.g. bed occupancy, flow indicators, respiratory demand), the system:

- **Assesses** how consistent the new signals are with previous evidence
- **Updates** its belief **gradually**, rather than jumping to conclusions
- **Maintains uncertainty**, reflecting data gaps, noise, and disagreement between signals

The result is a **probabilistic assessment** of current system pressure, **not a point prediction**.
Strong or consistent evidence moves the estimate more; weak or noisy evidence moves it less.
        """
    )

st.markdown("---")
st.markdown("### Evidence Distribution")
st.caption(
    "Question answered: **How much posterior weight sits above routine vs elevated pressure?** "
    "Dashed lines are **demo reference** cut-points — not national operational thresholds."
)

fig = _plot_pressure_question(
    samples,
    credible_mass=credible_mass,
    show_median_line=show_median_line,
)
st.pyplot(fig, clear_figure=True)
plt.close(fig)

st.caption(
    f"Area: {icb}. Values to the right of the dashed references indicate stronger posterior "
    "support for elevated modelled pressure (given assumptions and available indicators in this demo)."
)

st.markdown(
    f"""
**Reading the chart:** The blue bars show which values of the **system pressure index** are most plausible **after** seeing bed occupancy in this area.
The shaded band is the **{cred_pct}% plausible range** (see sidebar). Dashed lines are **demo reference levels** for conversation — they do not replace local judgement or official escalation rules.
**Calibration:** The index is an internal model construct; interpret **probabilities and intervals**, not the numeric scale as an NHS operational measure.
    """
)

st.info(
    "**Uncertainty:** Spreads can reflect sparse weeks, conflicting signals, reporting gaps, and model simplifications. "
    "Use alongside operational intelligence and governance processes."
)

st.markdown("---")
st.markdown("### Pressure Trajectory")
st.caption(
    "How has modelled pressure evolved over time for this area? "
    "The shaded band is the **80% posterior credible interval**. "
    "Dots are observed bed occupancy values."
)

traj_fig = _fig_pressure_trajectory(idata, df, icb)
if traj_fig is not None:
    st.pyplot(traj_fig, clear_figure=True)
    plt.close(traj_fig)
else:
    st.warning("Trajectory unavailable — ICB not found in posterior metadata.")

st.markdown("---")
st.markdown("### Clinical Summary")

if icb == "England":
    st.caption(
        "All ICBs ranked by current modelled pressure. "
        "Colour indicates risk band: red = elevated, amber = medium, green = low. "
        "Direction of travel shows whether pressure has risen or fallen over the last 4 weeks."
    )
    summary_fig = _fig_clinical_summary(idata, icb_filter=None)
else:
    st.caption(
        f"Current pressure, risk probability, and direction of travel for **{icb}**."
    )
    summary_fig = _fig_clinical_summary(idata, icb_filter=icb)

st.pyplot(summary_fig, clear_figure=True)
plt.close(summary_fig)
st.caption(
    "Reference lines: amber dashed = 95% occupancy, red dashed = 100% occupancy. "
    "Risk probability thresholds can be adjusted in the Advanced sidebar panel."
)