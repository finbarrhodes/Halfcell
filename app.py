"""
app.py — Streamlit Community Cloud entry point
===============================================
Run locally:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable when launched from any working directory
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Halfcell",
    layout="wide",
)

st.sidebar.markdown(
    """
    <style>
    a {
        color: #C9400A !important;
        text-decoration: underline !important;
    }
    [data-testid="stSidebarNav"] a {
        color: inherit !important;
        text-decoration: none !important;
    }
    [data-testid="stSidebarNav"] a:hover {
        text-decoration: none !important;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 1.5rem;
        font-size: 0.85rem;
        text-align: center;
        width: 16rem;
    }
    .sidebar-footer a {
        color: #C9400A !important;
        text-decoration: none !important;
    }
    .sidebar-footer a:hover {
        text-decoration: underline !important;
    }
    </style>
    <div class="sidebar-footer">
    Last updated: March 2026<br><br>
    <a href="https://github.com/finbarrhodes" target="_blank">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/finbar-rhodes-637650210/" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("src/visualization/home.py",        title="Home"),
    st.Page("src/visualization/dashboard.py",   title="Market Overview"),
    st.Page("src/visualization/backtester.py",  title="Forecasting & Dispatch"),
    st.Page("src/visualization/methodology.py", title="Methodology & Data"),
]

pg = st.navigation(pages)
pg.run()
