"""
Fast serving UI for pre-computed posteriors.

Minimal overhead version that loads only cached results.
Ideal for dashboards, read-only access, or high-traffic scenarios.

Compare to app.py:
- app.py: Full UI with tuning, data loading, comprehensive explainers
- app_fast.py: Lightweight display of current estimates only

Usage:
    streamlit run src/epiforcasts_nhs/dashboard/app_fast.py
"""

import os

# Headless backend
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import arviz as az

from epiforcasts_nhs.core.cache import CacheManager
from epiforcasts_nhs.dashboard.utils import (
    THRESHOLD_BASELINE,
    THRESHOLD_CONCERN,
    THRESHOLD_ELEVATED,
    credible_triplet,
    pressure_index_samples,
    risk_band,
)

POSTERIORS_NC = "posteriors.nc"


def _plot_minimal(samples: np.ndarray) -> plt.Figure:
    """Minimal histogram for fast rendering."""
    lo, mid, hi = credible_triplet(samples, 0.9)

    fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
    ax.hist(
        samples,
        bins=40,
        density=True,
        alpha=0.75,
        color="#1d4ed8",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axvspan(lo, hi, alpha=0.15, color="#1e3a8a", label="90% plausible range")
    ax.axvline(THRESHOLD_BASELINE, color="#64748b", linestyle="--", linewidth=1.2, label="Baseline")
    ax.axvline(THRESHOLD_CONCERN, color="#d97706", linestyle="--", linewidth=1.2, label="Concern")
    ax.axvline(THRESHOLD_ELEVATED, color="#b91c1c", linestyle="--", linewidth=1.2, label="High pressure")

    ax.set_xlabel("System pressure index (unitless model)")
    ax.set_ylabel("Plausibility")
    ax.set_title(f"Where does evidence put pressure? (median: {mid:.2f})", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig


def _inject_nhs_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --nhs-blue: #005eb8;
    --nhs-dark-blue: #003087;
    --nhs-light-blue: #e8f3ff;
    --border-soft: #d8dde0;
}

.stApp {
    background: linear-gradient(180deg, #f4f8ff 0%, #ffffff 45%);
    font-family: 'Public Sans', 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: var(--nhs-dark-blue);
    letter-spacing: -0.01em;
}

.hero {
    background: linear-gradient(120deg, var(--nhs-dark-blue), var(--nhs-blue));
    color: white;
    border-radius: 14px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 5px 18px rgba(0, 48, 135, 0.18);
}

.hero h1 {
    color: white;
    margin: 0;
    font-size: 1.5rem;
}

.hero p {
    margin: 0.35rem 0 0;
    opacity: 0.96;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border-soft);
    border-radius: 12px;
    padding: 0.5rem 0.65rem;
    box-shadow: 0 1px 5px rgba(0, 48, 135, 0.06);
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-soft);
    background: #f8fbff;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading cached posteriors…")
def load_posteriors(_cache_manager: CacheManager) -> az.InferenceData:
    """Load posteriors from cache (no MCMC, guaranteed)."""
    return _cache_manager.load_posteriors()


@st.cache_resource(show_spinner="Loading cached statistics…")
def load_summary_stats(_cache_manager: CacheManager) -> dict:
    """Load pre-computed summary statistics (instant access)."""
    return _cache_manager.load_summary_stats()


def get_last_update_time(path: str) -> str:
    """Get human-readable last update time."""
    if not Path(path).exists():
        return "never"
    mtime = Path(path).stat().st_mtime
    from datetime import datetime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S UTC")


# ============================================================================
# UI
# ============================================================================

st.set_page_config(layout="wide", page_title="NHS Pressure — Fast View (demo)")
_inject_nhs_theme()
st.markdown(
    """
<div class="hero">
  <h1>NHS System Pressure — Fast View</h1>
  <p>Read-only operational snapshot for quick review in huddles and escalation calls.</p>
</div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Updated from cached posterior artifacts. Use the full dashboard for richer controls and explainability."
)

# Initialize cache manager
cache = CacheManager(posteriors_path=POSTERIORS_NC)

# Validate cache is ready (no MCMC runs guaranteed here)
if not cache.is_valid():
    st.error(
        f"Cache not ready\n\n"
        f"**Status:** Posterior {'ok' if cache.is_posterior_available() else 'missing'} | "
        f"Cache {'ok' if cache.is_cache_warm() else 'not warmed'}\n\n"
        f"**Run:** `epiforcasts-daemon --once`"
    )
    st.stop()

# Load from cache (guaranteed zero computation)
idata = load_posteriors(cache)
stats = load_summary_stats(cache)
last_update = get_last_update_time(POSTERIORS_NC)

# Get ICB list from metadata
icbs = idata.attrs.get("icbs", [])
if not icbs:
    st.error("No ICB metadata in posterior file. Please regenerate.")
    st.stop()

# Sidebar: quick select
with st.sidebar:
    st.header("Quick Select")
    selected_icb = st.radio(
        "Geography",
        options=icbs,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"**Last updated:** {last_update}")
    st.caption(f"**Posterior samples:** {int(idata.posterior.dims.get('draw', 0)):,}")
    st.caption(f"**Geographic areas:** {len(icbs)}")

# Main display
icb_idx = icbs.index(selected_icb)
samples = pressure_index_samples(idata, icb_idx)

p_above_baseline = float(np.mean(samples > THRESHOLD_BASELINE))
p_above_concern = float(np.mean(samples > THRESHOLD_CONCERN))
p_above_elevated = float(np.mean(samples > THRESHOLD_ELEVATED))

lo, mid, hi = credible_triplet(samples, 0.9)
risk_label, risk_hint = risk_band(p_above_elevated, p_above_concern)

# Summary card
with st.container(border=True):
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

    with col1:
        st.markdown(f"## {risk_label}")
        st.caption(risk_hint)

    with col2:
        st.metric(
            label="P(baseline)",
            value=f"{p_above_baseline:.0%}",
            delta="threshold: 0.0",
        )

    with col3:
        st.metric(
            label="P(concern)",
            value=f"{p_above_concern:.0%}",
            delta="threshold: 0.5",
        )

    with col4:
        st.metric(
            label="P(high)",
            value=f"{p_above_elevated:.0%}",
            delta="threshold: 1.1",
        )

st.markdown(
    f"**90% plausible range:** {lo:.2f} — {hi:.2f} "
    f"(median {mid:.2f}). Not NHS-calibrated units."
)

# Chart
st.markdown("### Evidence Distribution")
fig = _plot_minimal(samples)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Footer
st.divider()
st.caption(
    f"**Area:** {selected_icb} | "
    f"**Model:** Latent pressure on bed occupancy (demo only)"
)
