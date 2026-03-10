"""
Home — GB BESS Market Analysis Tool
=====================================
Landing page. Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/home.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="GB BESS Market Analysis",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_, banner_col, _ = st.columns([1, 8, 1])
with banner_col:
    st.image("bess-banner.png", use_container_width=True)

st.title("BESS Analytics Tools")
st.markdown(
    "Hello — my name is Finbar Rhodes. I have a passion for the energy transition and am particularly "
    "interested in flexibility and grid-scale energy storage. This role, particularly in GB, has "
    "increasingly been filled by battery energy storage systems (BESS). The combination of improving "
    "battery technology, cheaper components, and the growing need for flexibility assets in a grid "
    "increasingly reliant on renewable energy sources leaves BESS with a bright future in the energy "
    "transition. Batteries are playing a major role beyond the scope of this project, and if you're "
    "interested, [here](https://climate.benjames.io/batteries/) is a great guide I used.\n\n"
    "This is a personal project I have undertaken to dive into the GB clean tech and grid-scale "
    "battery landscape — learning through doing while aiming to surface some useful insights. "
    "With a background in data science, statistics, and economics, I have tried to bring my "
    "analytical skills and machine learning experience to bear on how impactful this technology "
    "is and how it is making its mark on our grid. I have focused on two key revenue verticals for "
    "BESS: frequency response services and energy arbitrage, which I go into more detail about below. "
    "There is a *Market Overview* section showing how market conditions have evolved over time, and a "
    "*Forecasting & Dispatch* model that applies ML price forecasting and Model Predictive Control "
    "(MPC) optimisation to the day-ahead planning layer of BESS operations. "
    "The data powering this tool is sourced from the **Elexon Insights Solution API** "
    "and **NESO Data Portal**, with methodology based on well-established literature."
)

st.divider()

# ---------------------------------------------------------------------------
# About — BESS revenue routes & market context
# ---------------------------------------------------------------------------

st.markdown(
    """
    Battery Energy Storage Systems (BESS) do not rely on a single revenue source — they
    **stack** income from multiple markets, often simultaneously:

    | Revenue route | How it works |
    |---|---|
    | **Frequency response** (Dynamic Services) | Paid a £/MWh availability fee to hold discharge or charge headroom; activated when grid frequency deviates from 50 Hz by even a fraction of a percentage. |
    | **Wholesale arbitrage** | Trading on the wholesale market where batteries charge during low-price periods (high wind, low demand) and discharge during high-price periods |
    | **Balancing Mechanism (BM)** | Dispatched by NESO in real time via bids/auctions to correct short-term supply/demand imbalance |
    | **Capacity Market** | Longer-term availability payments for committing to providing energy capacity during periods of system stress |

    **Why this tool focuses on Dynamic Services and wholesale activity:**
    Dynamic services (DC, DM, DR) dominated the GB BESS revenue stack from roughly 2021 through
    early 2023 — at times accounting for over 70% of total asset revenue, with some assets
    earning ~£156k/MW/year at the 2022 peak
    ([Modo Energy](https://modoenergy.com/research/future-of-battery-energy-storage-buildout-in-great-britain);
    [Timera Energy](https://timera-energy.com/blog/battery-investors-confront-revenue-shift-in-2023/)).
    From late 2022, a rapid influx of new BESS capacity saturated the frequency response
    markets and revenues compressed sharply — a trend visible in the clearing price charts in
    the Market Overview. Dynamic services remain a core stack component, but arbitrage and
    Capacity Market income have grown in relative importance since.

    **BESS is not the only participant in these markets.** Pumped-storage hydro (e.g. Dinorwig,
    Cruachan) has provided fast frequency response for decades. Demand-side response and some
    gas peakers also qualify, particularly for the slower DR service. However, BESS's
    sub-second response capability has made it the dominant and marginal price-setting
    technology in DC and DM auctions.

    **A note on data and scope:** Building a truly complete BESS revenue model would require
    asset-level operational data — dispatch logs, BM unit IDs, exact SoC histories, and grid
    connection limits. Much of this is commercially sensitive and not publicly disclosed;
    operators reasonably keep detailed performance data private. This tool is built entirely
    on publicly available data from NESO and Elexon, and the modelling approach applies the
    best analytical methods available within that constraint. The goal is not to replicate a
    proprietary trading system, but to quantify — as rigorously as the data allows — how
    market conditions and forecast quality interact to drive BESS revenue outcomes.
    """
)

# ---------------------------------------------------------------------------
# Key market events timeline
# ---------------------------------------------------------------------------

st.subheader("GB BESS Market: Key Events")
st.markdown(
    """
    | Year | Event |
    |------|-------|
    | **2020** | NESO launches Dynamic Containment (DC) — BESS becomes the dominant provider within months, displacing gas peakers |
    | **2021** | Dynamic Regulation (DR) and Dynamic Moderation (DM) introduced; revenue stacking across all three services becomes standard |
    | **2022** | Revenue peak — DC High averaging £15–20/MW/h; leading assets earning ~£156k/MW/year |
    | **Late 2022** | Rapid capacity influx saturates frequency response markets; clearing prices begin a sharp, sustained decline |
    | **2023** | Revenue compression accelerates; wholesale arbitrage and Capacity Market grow significantly in relative importance |
    | **2024–25** | Stack diversification — operators blend FR, arbitrage, and BM participation; long-duration projects begin to emerge |
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Navigation guide
# ---------------------------------------------------------------------------

st.subheader("What's in this tool")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("**Market Overview**")
        st.markdown(
            "Explore GB frequency response auction clearing prices (DC, DR, DM), "
            "High vs Low spread dynamics, system settlement prices (SBP/SSP), "
            "and generation mix trends. Includes a correlation analysis between "
            "system prices and DC High auction outcomes."
        )
        st.page_link("src/visualization/dashboard.py", label="Market Overview →")

with col2:
    with st.container(border=True):
        st.markdown("**Forecasting & Dispatch Model**")
        st.markdown(
            "A day-ahead modelling framework for FR/arbitrage capacity allocation and "
            "MPC dispatch. Benchmarks three price forecasting strategies — Naive D-1 "
            "baseline, Random Forest, and perfect foresight ceiling — to isolate how "
            "much forecast quality affects operational outcomes."
        )
        st.page_link("src/visualization/backtester.py", label="Forecasting & Dispatch →")

with col3:
    with st.container(border=True):
        st.markdown("**Methodology & Data**")
        st.markdown(
            "Understand the modelling assumptions, data sources, and known limitations "
            "of the backtester. Explains the two-stage participation model, dispatch "
            "strategies, battery cycling cost, and the ML price forecast approach."
        )
        st.page_link("src/visualization/methodology.py", label="Methodology & Data →")

st.divider()

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

st.caption(
    "Data sourced from the "
    "[Elexon Insights Solution API](https://bmrs.elexon.co.uk/) "
    "and the [NESO Data Portal](https://www.neso.energy/data-portal). "
    "All prices in GBP. "
    "Frequency response auction data (DC, DR, DM) covers September 2021 onwards. "
    "APXMIDP wholesale price data and generation mix data cover July 2023 – present. "
    "The ML price forecast model is trained on July 2023 – February 2025 with a "
    "held-out test set from March 2025 onwards."
)
