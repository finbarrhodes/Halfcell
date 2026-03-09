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
    page_title="UK BESS Market Analysis",
    layout="wide",
)

pages = [
    st.Page("src/visualization/home.py",        title="Home"),
    st.Page("src/visualization/dashboard.py",   title="Market Overview"),
    st.Page("src/visualization/backtester.py",  title="Forecasting & Dispatch"),
    st.Page("src/visualization/methodology.py", title="Methodology & Data"),
]

pg = st.navigation(pages)
pg.run()
