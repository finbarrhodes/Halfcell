"""
GB BESS Market Dashboard

Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="GB BESS Market Dashboard",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"

# Consistent colour map for DC / DR / DM across all spread plots
MARKET_COLORS = {"DC": "#0D7680", "DR": "#C9400A", "DM": "#4E8A3C"}

# EFA block timings (each block = 4 hours; EFA 1 spans midnight)
EFA_BLOCKS = {
    1: "23:00 – 03:00",
    2: "03:00 – 07:00",
    3: "07:00 – 11:00",
    4: "11:00 – 15:00",
    5: "15:00 – 19:00",
    6: "19:00 – 23:00",
}


# ---------------------------------------------------------------------------
# Data loading (cached, reads pre-processed Parquet files)
# ---------------------------------------------------------------------------

@st.cache_data
def load_auctions() -> pd.DataFrame:
    p = PROCESSED / "auctions.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_system_prices() -> pd.DataFrame:
    p = PROCESSED / "system_prices.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_market_index() -> pd.DataFrame:
    p = PROCESSED / "market_index.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_generation() -> pd.DataFrame:
    """
    Returns daily generation totals by fuel group.
    Pre-aggregated by scripts/prepare_data.py — columns: settlementDate, fuelGroup, generation.
    """
    p = PROCESSED / "generation_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


auctions   = load_auctions()
sys_prices = load_system_prices()
mkt_index  = load_market_index()
gen_fuel   = load_generation()  # already has fuelGroup column

# Sidebar filters
st.sidebar.title("Filters")

if not auctions.empty:
    all_services = sorted(auctions["Service"].unique())
    selected_services = st.sidebar.multiselect(
        "DC/DR/DM Services", all_services, default=all_services
    )

    date_min = auctions["EFA Date"].min().date()
    date_max = auctions["EFA Date"].max().date()
    date_range = st.sidebar.date_input(
        "Auction date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    auction_filtered = auctions[auctions["Service"].isin(selected_services)]
    if len(date_range) == 2:
        auction_filtered = auction_filtered[
            (auction_filtered["EFA Date"].dt.date >= date_range[0])
            & (auction_filtered["EFA Date"].dt.date <= date_range[1])
        ]
else:
    auction_filtered = auctions
    selected_services = []

# Header
st.title("Market Overview")
st.markdown("Data from the **Elexon Insights Solution API** and **NESO Data Portal**.")

# Top-level metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Auction Records", f"{len(auction_filtered):,}")
col2.metric("System Price Records", f"{len(sys_prices):,}")
col3.metric("Market Index Records", f"{len(mkt_index):,}")
col4.metric("Generation Records", f"{len(gen_fuel):,}")

# ---------------------------------------------------------------------------
# Outer tab layout
# ---------------------------------------------------------------------------
outer_fr, outer_grid = st.tabs(["Frequency Response Markets", "Grid & Settlement Prices"])

# ---------------------------------------------------------------------------
# Outer Tab 1: Frequency Response Markets → sub-tabs
# ---------------------------------------------------------------------------
with outer_fr:
    sub_auction, sub_spread = st.tabs(["Dynamic Services", "H vs L Spread"])

# ---------------------------------------------------------------------------
# Sub-tab 1a: Dynamic Services
# ---------------------------------------------------------------------------
with sub_auction:
    if auction_filtered.empty:
        st.warning("No auction data loaded. Run scripts/prepare_data.py first.")
    else:
        st.markdown(
            """
            GB frequency response is procured through three [**dynamic** services](https://www.neso.energy/industry-information/balancing-services/frequency-response-services/dynamic-services-dcdmdr),
            each split into **High** (charge — activated when frequency rises above 50 Hz) and
            **Low** (discharge — activated when frequency falls below 50 Hz) auctions:

            | Service | Frequency band | Role |
            |---------|---------------|------|
            | **DC** – Dynamic Containment | ±0.2–0.5 Hz | Arrests large deviations within ~1 second |
            | **DR** – Dynamic Regulation | ±0.015–0.2 Hz | Maintains frequency in normal operation |
            | **DM** – Dynamic Moderation | ±0.1–0.5 Hz | Moderates frequency during stressed conditions |

            Auctions are run daily for each **EFA block** (six 4-hour windows covering the full day).
            The clearing price is the marginal accepted bid for that block and service.
            """
        )

        with st.expander("EFA Block Timings"):
            efa_df = pd.DataFrame(
                [(k, v) for k, v in EFA_BLOCKS.items()],
                columns=["EFA Block", "Time Window (local clock)"],
            )
            st.dataframe(efa_df, hide_index=True, use_container_width=True)
            st.caption(
                "EFA Block 1 spans midnight (23:00 the previous calendar day to 03:00). "
                "All times are local GB time."
            )

        st.subheader("Clearing Prices — 28-Day Rolling Average by Service")
        st.caption(
            "Individual auction results are first averaged to a daily figure per service, "
            "then smoothed with a 28-day rolling window. This makes the trend for each of "
            "the six services readable without the daily noise obscuring the signal."
        )

        # Daily average per service → 28-day rolling average
        daily_auction = (
            auction_filtered
            .groupby(["EFA Date", "Service"])["Clearing Price"]
            .mean()
            .reset_index()
            .sort_values("EFA Date")
        )
        rolling_parts = []
        for svc, grp in daily_auction.groupby("Service"):
            grp = grp.set_index("EFA Date").sort_index()
            grp["Rolling Avg (£/MW/h)"] = grp["Clearing Price"].rolling("28D").mean()
            grp["Service"] = svc
            rolling_parts.append(grp.reset_index())
        rolling_df = pd.concat(rolling_parts)

        # Order legend by mean rolling average descending so highest-value service appears first
        service_order = (
            rolling_df.groupby("Service")["Rolling Avg (£/MW/h)"].mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig = px.line(
            rolling_df,
            x="EFA Date",
            y="Rolling Avg (£/MW/h)",
            color="Service",
            labels={"EFA Date": "Date"},
            category_orders={"Service": service_order},
        )
        fig.update_layout(height=450, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Key takeaways — Clearing Price Trends"):
            st.markdown(
                """
                - **2022 peak then sharp compression.** DCH clearing prices peaked at £15–20/MW/h
                  in 2022 as NESO expanded DC procurement ahead of renewable growth. From late 2022
                  a rapid wave of new GB BESS capacity entered the frequency response markets,
                  outpacing NESO's procurement volumes and driving prices steeply lower across all
                  services — a trend that is clearly visible in the chart.
                - **Charge (High) services generally clear above discharge (Low) services.**
                  Fleet-wide discharge headroom tends to be more available than charge headroom —
                  particularly during high-wind periods — so the Low-side auctions typically clear
                  at lower prices.
                - **DRH and DRL behave differently from DC and DM.** DR's sustained 60-minute
                  delivery requirement couples the two sides operationally, which is why the DRL
                  spread sometimes inverts relative to DCL and DML (explored further in the
                  H vs L Spread tab).
                """
            )

        left, right = st.columns(2)

        with left:
            st.subheader("Price Distribution by Service")
            fig = px.box(
                auction_filtered,
                x="Service",
                y="Clearing Price",
                color="Service",
                labels={"Clearing Price": "£/MW/h"},
            )
            fig.update_layout(height=400, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "DCH shows the widest spread of outcomes, reflecting its role as the primary "
                "fast-charge service and its early-market dominance at elevated prices. "
                "Low-side services (DCL, DRL, DML) cluster at lower prices as discharge headroom "
                "has generally been more plentiful than charge headroom across the fleet."
            )

        with right:
            st.subheader("Price by EFA Block")
            st.caption(
                "Evening blocks (EFA 5: 15:00–19:00, EFA 6: 19:00–23:00) and the overnight "
                "block (EFA 1: 23:00–03:00) often attract different premia depending on "
                "wind output and demand shape that day."
            )
            fig = px.box(
                auction_filtered,
                x="EFA",
                y="Clearing Price",
                color="Service",
                labels={
                    "Clearing Price": "£/MW/h",
                    "EFA": "EFA Block",
                },
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(EFA_BLOCKS.keys()),
                ticktext=[f"EFA {k}<br>{v}" for k, v in EFA_BLOCKS.items()],
            )
            fig.update_layout(height=400, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Evening blocks (EFA 5 and 6: 15:00–23:00) tend to attract higher premia as "
                "demand peaks and wind output often eases. The overnight block (EFA 1) is "
                "typically cheapest to procure, reflecting lower system demand and more available "
                "response headroom from assets running light overnight."
            )

        st.subheader("Summary Statistics")
        stats = (
            auction_filtered.groupby("Service")
            .agg(
                avg_price=("Clearing Price", "mean"),
                median_price=("Clearing Price", "median"),
                max_price=("Clearing Price", "max"),
                avg_volume=("Cleared Volume", "mean"),
                records=("Clearing Price", "count"),
            )
            .round(2)
        )
        st.dataframe(
            stats,
            use_container_width=True,
            column_config={
                "avg_price": st.column_config.ProgressColumn(
                    "Avg Price (£/MW/h)", format="£%.2f",
                    min_value=0, max_value=float(stats["avg_price"].max()),
                ),
                "median_price": st.column_config.ProgressColumn(
                    "Median (£/MW/h)", format="£%.2f",
                    min_value=0, max_value=float(stats["avg_price"].max()),
                ),
                "max_price": st.column_config.ProgressColumn(
                    "Max (£/MW/h)", format="£%.2f",
                    min_value=0, max_value=float(stats["max_price"].max()),
                ),
                "avg_volume": st.column_config.ProgressColumn(
                    "Avg Cleared Volume (MW)", format="%.1f MW",
                    min_value=0, max_value=float(stats["avg_volume"].max()),
                ),
                "records": st.column_config.NumberColumn("Records"),
            },
        )


# ---------------------------------------------------------------------------
# Sub-tab 1b: H vs L Spread
# ---------------------------------------------------------------------------
with sub_spread:
    if auctions.empty:
        st.warning("No auction data loaded. Run scripts/prepare_data.py first.")
    else:
        st.markdown(
            """
            Each frequency response service runs two separate auctions: **High** (responds to
            rising frequency — BESS charges) and **Low** (responds to falling frequency —
            BESS discharges). The clearing prices can differ because the amount of available
            discharge vs charge headroom across the fleet is rarely symmetric.

            **Spread = H clearing price − L clearing price.** Positive = discharge capacity
            scarcer (H > L). Negative = charge capacity scarcer (L > H).
            """
        )

        # Build a wide table: one row per (EFA Date, EFA block), columns H and L per market
        PAIRS = [("DC", "DCH", "DCL"), ("DR", "DRH", "DRL"), ("DM", "DMH", "DML")]

        spread_frames = []
        for market, h_svc, l_svc in PAIRS:
            h = auctions[auctions["Service"] == h_svc][["EFA Date", "EFA", "Clearing Price"]].rename(
                columns={"Clearing Price": "H"}
            )
            l = auctions[auctions["Service"] == l_svc][["EFA Date", "EFA", "Clearing Price"]].rename(
                columns={"Clearing Price": "L"}
            )
            merged = pd.merge(h, l, on=["EFA Date", "EFA"], how="inner")
            merged["Spread"] = merged["H"] - merged["L"]
            merged["Market"] = market
            spread_frames.append(merged)

        if not spread_frames:
            st.warning("Could not compute spreads — check that H and L services are both present.")
        else:
            spread_df = pd.concat(spread_frames, ignore_index=True)

            # ---- Chart 1: daily average spread over time ----
            st.subheader("Daily Average H − L Spread Over Time")
            daily_spread = (
                spread_df.groupby(["EFA Date", "Market"])["Spread"]
                .mean()
                .reset_index()
            )
            fig = px.line(
                daily_spread,
                x="EFA Date",
                y="Spread",
                color="Market",
                color_discrete_map=MARKET_COLORS,
                labels={"Spread": "£/MW/h", "EFA Date": "Date"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_layout(height=420, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
            st.plotly_chart(fig, use_container_width=True)

            # DR negative spread explanation (only show if the data supports it)
            dr_spread = spread_df[spread_df["Market"] == "DR"]["Spread"]
            if not dr_spread.empty and dr_spread.mean() < 0:
                st.info(
                    f"**Why is the DR spread consistently negative "
                    f"(avg {dr_spread.mean():.2f} £/MW/h)?**\n\n"
                    "DR (Dynamic Regulation) operates continuously in the normal frequency band "
                    "(very close to 50 Hz), and NESO's energy management rules require providers "
                    "to sustain their contracted position for the full **60-minute** delivery "
                    "window — much longer than DC (15 min) or DM (30 min). In practice, "
                    "providers holding both DRH and DRL positions need to maintain a SoC close "
                    "to the midpoint to honour either commitment for the full hour. The "
                    "longer window effectively couples the two sides together in a way that "
                    "DC and DM — with their shorter windows and genuinely independent High/Low "
                    "auctions — do not.\n\n"
                    "*(Note: DRH and DRL are technically separate auctions and providers can "
                    "bid into one without the other; this coupling is a practical consequence "
                    "of the 60-minute sustained delivery requirement, not an explicit equal-MW "
                    "rule. See NESO's "
                    "[Dynamic Response Services Provider Guidance](https://www.neso.energy/document/276606/download) "
                    "for the formal terms.)*\n\n"
                    "The practical consequence is that DRH — the charge-side service — ends up "
                    "structurally scarcer than DRL. The most intuitive way to think about it: "
                    "during periods of high renewable output, batteries providing DR tend to "
                    "fill up. A nearly-full battery simply has less room to absorb more energy, "
                    "so fewer assets across the fleet can realistically offer DRH at any given "
                    "time. On top of that, operators are cautious about committing the little "
                    "remaining charge headroom they do have — accepting more charge now could "
                    "leave them unable to discharge when the grid needs it later. With DRH supply "
                    "squeezed from both sides and operators needing more compensation to provide "
                    "it, DRH clearing prices are regularly pushed above DRL — flipping the "
                    "spread negative."
                )

            # ---- Chart 2: spread distribution per market ----
            left, right = st.columns(2)

            with left:
                st.subheader("Spread Distribution by Market")
                fig = px.box(
                    spread_df,
                    x="Market",
                    y="Spread",
                    color="Market",
                    color_discrete_map=MARKET_COLORS,
                    labels={"Spread": "£/MW/h"},
                    points="outliers",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
                fig.update_layout(height=400, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("Average Spread by EFA Block")
                efa_spread = (
                    spread_df.groupby(["EFA", "Market"])["Spread"]
                    .mean()
                    .reset_index()
                )
                fig = px.bar(
                    efa_spread,
                    x="EFA",
                    y="Spread",
                    color="Market",
                    color_discrete_map=MARKET_COLORS,
                    barmode="group",
                    labels={"Spread": "Avg £/MW/h", "EFA": "EFA Block"},
                )
                fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(EFA_BLOCKS.keys()),
                    ticktext=[f"EFA {k}<br>{v}" for k, v in EFA_BLOCKS.items()],
                )
                fig.update_layout(height=400, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                "DC shows the widest range of spread outcomes — its higher median reflects the "
                "relative scarcity of fast-discharge capacity, while the tail of outliers "
                "captures periods when that scarcity was acute. DR sits firmly in negative "
                "territory across both charts, confirming the structural inversion described "
                "above. DM occupies the middle ground, broadly positive but with a narrower "
                "spread than DC. In the EFA block view, evening blocks (EFA 5–6: 15:00–23:00) "
                "tend to show the most pronounced spreads as demand peaks and the balance "
                "between available charge and discharge headroom is at its tightest."
            )

            # ---- Chart 3: heatmap — EFA block × month ----
            st.subheader("H − L Spread Heatmap: EFA Block × Month")
            spread_df["Month"] = spread_df["EFA Date"].dt.to_period("M").astype(str)
            selected_market = st.selectbox("Market", ["DC", "DR", "DM"], key="heatmap_market")
            heatmap_data = (
                spread_df[spread_df["Market"] == selected_market]
                .groupby(["Month", "EFA"])["Spread"]
                .mean()
                .reset_index()
                .pivot(index="Month", columns="EFA", values="Spread")
            )
            # Label columns with EFA block times
            heatmap_data.columns = [
                f"EFA {c} ({EFA_BLOCKS.get(c, '')})" for c in heatmap_data.columns
            ]
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                labels={"color": "£/MW/h", "x": "EFA Block", "y": "Month"},
                aspect="auto",
            )
            fig.update_layout(height=420, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each cell shows the average H − L spread for that market, EFA block, and "
                "calendar month. Red = discharge capacity was scarcer than charge (H > L); "
                "blue = charge capacity was scarcer (L > H). Use the selector above to switch "
                "between DC, DR, and DM. Gaps indicate months with no auction data."
            )

            # ---- Summary table ----
            st.subheader("Summary Statistics")
            summary = (
                spread_df.groupby("Market")["Spread"]
                .agg(
                    mean="mean",
                    median="median",
                    std="std",
                    min="min",
                    max="max",
                    pct_positive=lambda s: (s > 0).mean() * 100,
                )
                .round(2)
            )
            summary.columns = ["Mean £/MW/h", "Median", "Std Dev", "Min", "Max", "% Days H > L"]
            st.dataframe(
                summary,
                use_container_width=True,
                column_config={
                    "Std Dev": st.column_config.ProgressColumn(
                        "Std Dev", format="£%.2f",
                        min_value=0, max_value=float(summary["Std Dev"].max()),
                    ),
                    "% Days H > L": st.column_config.ProgressColumn(
                        "% Days H > L", format="%.1f%%",
                        min_value=0, max_value=100,
                    ),
                },
            )


# ---------------------------------------------------------------------------
# Outer Tab 2: Grid & Settlement Prices → sub-tabs
# ---------------------------------------------------------------------------
with outer_grid:
    sub_system, sub_gen = st.tabs(["System Prices", "Generation Mix"])

# ---------------------------------------------------------------------------
# Sub-tab 2a: System Prices
# ---------------------------------------------------------------------------
with sub_system:
    if sys_prices.empty:
        st.warning("No system price data loaded.")
    else:
        st.markdown(
            """
            **System Buy Price (SBP)** and **System Sell Price (SSP)** are the cash-out prices
            used to settle imbalance in the GB Balancing Mechanism. Parties that are *short*
            (consumed more than they contracted for) pay the SBP; parties that are *long* receive
            the SSP. SBP ≥ SSP always — the gap between them incentivises generators and suppliers
            to self-balance rather than rely on the system operator.
            """
        )

        st.subheader("Daily Average System Prices")
        daily_sp = (
            sys_prices.groupby("settlementDate")
            .agg(
                avg_ssp=("systemSellPrice", "mean"),
                avg_sbp=("systemBuyPrice", "mean"),
            )
            .reset_index()
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_ssp"],
                name="Avg SSP",
                line=dict(color="#C9400A"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_sp["settlementDate"],
                y=daily_sp["avg_sbp"],
                name="Avg SBP",
                line=dict(color="#0D7680"),
            )
        )
        fig.update_layout(
            height=450,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            yaxis_title="£/MWh",
        )
        st.plotly_chart(fig, use_container_width=True)

        if not auctions.empty:
            with st.expander("SSP vs DC High Clearing Price — correlation analysis"):
                st.markdown(
                    "These two prices come from different markets — imbalance settlement "
                    "(real-time) vs contracted frequency response (day-ahead) — but may "
                    "co-move if they share common drivers. Tight supply margins, for example, "
                    "could simultaneously push up energy prices and increase willingness to pay "
                    "for frequency response capacity. A low correlation suggests DCH prices are "
                    "driven primarily by the frequency response fleet's own supply/demand "
                    "dynamics, independent of wholesale energy conditions."
                )
                daily_sys = (
                    sys_prices.groupby("settlementDate")["systemSellPrice"]
                    .mean()
                    .reset_index()
                )
                daily_sys.columns = ["date", "avg_system_price"]
                dc_high = auctions[auctions["Service"] == "DCH"]
                daily_dc = (
                    dc_high.groupby("EFA Date")["Clearing Price"].mean().reset_index()
                )
                daily_dc.columns = ["date", "avg_dc_clearing_price"]
                merged = pd.merge(daily_sys, daily_dc, on="date", how="inner")
                if merged.empty:
                    st.info("No overlapping dates between system prices and DC auctions.")
                else:
                    corr = merged[["avg_system_price", "avg_dc_clearing_price"]].corr().iloc[0, 1]
                    st.metric("Pearson Correlation (SSP vs DCH)", f"{corr:.3f}")
                    if abs(corr) < 0.3:
                        st.caption(
                            "Low correlation — DCH prices appear to be driven primarily by "
                            "frequency response supply/demand dynamics rather than wholesale "
                            "energy price levels."
                        )
                    elif corr > 0.5:
                        st.caption(
                            "Moderate-to-strong correlation — common market drivers may be at "
                            "work, such as tight supply conditions lifting both energy and "
                            "frequency response prices simultaneously."
                        )
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(
                        go.Scatter(
                            x=merged["date"],
                            y=merged["avg_system_price"],
                            name="Avg SSP",
                            line=dict(color="#C9400A"),
                        ),
                        secondary_y=False,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=merged["date"],
                            y=merged["avg_dc_clearing_price"],
                            name="Avg DC High",
                            line=dict(color="#0D7680"),
                        ),
                        secondary_y=True,
                    )
                    fig.update_yaxes(title_text="System Price (£/MWh)", secondary_y=False)
                    fig.update_yaxes(title_text="DC Clearing (£/MW/h)", secondary_y=True)
                    fig.update_layout(
                        height=450,
                        template="plotly_white",
                        paper_bgcolor="#FFF1E5",
                        plot_bgcolor="#FFF1E5",
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sub-tab 2b: Generation Mix
# ---------------------------------------------------------------------------
with sub_gen:
    if gen_fuel.empty:
        st.warning("No generation data loaded.")
    else:
        st.markdown(
            """
            GB grid generation broken down by fuel group. The mix matters for BESS because it
            shapes the underlying risk of frequency deviation: high wind + low demand tends to
            push system frequency high (requiring charge / High-side response), while low wind +
            high demand can cause frequency dips (requiring discharge / Low-side response).
            Tracking the long-run shift from fossil to renewable generation gives context for
            why frequency response procurement requirements have grown over time.
            """
        )

        gen_pivot = gen_fuel.pivot_table(
            index="settlementDate",
            columns="fuelGroup",
            values="generation",
            fill_value=0,
        )
        gen_pivot.index = pd.DatetimeIndex(gen_pivot.index)

        st.subheader("14-Day Rolling Average Generation by Fuel Group")
        st.caption(
            "Daily generation totals are smoothed with a 14-day rolling mean to remove "
            "day-to-day noise while still resolving seasonal swings within a month. "
            "Each line represents one fuel group; click items in the legend to isolate sources."
        )

        gen_smooth = gen_pivot.rolling(14, min_periods=7).mean()
        col_order = gen_smooth.sum().sort_values(ascending=False).index
        gen_smooth = gen_smooth[col_order]

        melted_weekly = gen_smooth.reset_index().melt(
            id_vars="settlementDate",
            var_name="Fuel Group",
            value_name="Avg Generation (MW)",
        )
        fig = px.line(
            melted_weekly,
            x="settlementDate",
            y="Avg Generation (MW)",
            color="Fuel Group",
            labels={"settlementDate": "Date"},
        )
        fig.update_layout(height=500, template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)

        fuel_share = gen_pivot.mean()
        fuel_share = fuel_share[fuel_share > 0].sort_values(ascending=False)

        with left:
            st.subheader("Average Share by Fuel Group")
            fig = px.pie(values=fuel_share.values, names=fuel_share.index)
            fig.update_layout(height=420, paper_bgcolor="#FFF1E5")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Clean vs Fossil vs Other")
            # Map Elexon fuel groups to three high-level buckets
            # Actual fuelGroup values in data: Wind, Gas, Nuclear, Biomass, Coal,
            # Hydro, Interconnectors, Oil, Pumped Storage, Other
            CLEAN = {"WIND", "HYDRO", "BIOMASS", "SOLAR"}
            FOSSIL = {"GAS", "COAL", "OIL"}
            bucket_map = {
                g: "Renewables" if g.upper() in CLEAN
                else "Fossil Fuels" if g.upper() in FOSSIL
                else "Other (nuclear, imports, storage)"
                for g in fuel_share.index
            }
            bucketed = fuel_share.groupby(bucket_map).sum()
            BUCKET_COLORS = {
                "Renewables": "#4E8A3C",
                "Fossil Fuels": "#C9400A",
                "Other (nuclear, imports, storage)": "#8B8B8B",
            }
            fig = px.pie(
                values=bucketed.values,
                names=bucketed.index,
                color=bucketed.index,
                color_discrete_map=BUCKET_COLORS,
            )
            fig.update_layout(height=420, paper_bgcolor="#FFF1E5")
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "**Sources:** Transmission-connected generation is sourced from Elexon FUELHH, "
            "which covers large grid-connected assets metered at the boundary. Embedded solar "
            "and embedded wind are sourced from the NESO Historic Demand Data dataset and "
            "represent half-hourly estimates derived from installed capacity registers and "
            "Met Office irradiance and wind-speed models — the same figures NESO uses "
            "internally for real-time grid balancing. "
            "**Why estimates, not meter readings?** Most rooftop solar and small distribution-"
            "connected wind falls below the metering threshold for direct settlement data. "
            "NESO back-calculates embedded generation by subtracting metered demand from "
            "modelled total consumption, producing the operationally relevant signal even "
            "though it is not a direct physical measurement. "
            "**Relevance for BESS analytics:** This mix reflects the generation profile that "
            "drives system frequency deviations and therefore FR procurement need — the "
            "operational picture that BESS operators respond to, rather than a statistical "
            "accounting of all electrons produced. Including embedded renewables meaningfully "
            "raises the measured renewable fraction, particularly in summer midday periods "
            "when embedded solar peaks. "
            "**Why this may differ from published energy statistics:** Sources such as Ember "
            "or BEIS DUKES reconcile generation against annual surveys of small-scale "
            "generators and FiT/SEG administrative data, and adjust for "
            "transmission/distribution losses and auxiliary consumption. Those adjustments "
            "are appropriate for energy accounting but unnecessary for market signal analysis."
        )


