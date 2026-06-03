import streamlit as st 

def inject_nhs_theme() -> None:
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

