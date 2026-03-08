"""
BESS Revenue Backtester
==============================
Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/backtester.py
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when launched standalone from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.analysis.revenue_stack import (
    BatterySpec,
    run_backtest,
    sensitivity_table,
    ALL_SERVICES,
    SERVICE_LABELS,
    SERVICE_COLOURS,
)
from src.analysis.price_forecast import (
    build_feature_matrix,
    train_forecast_model,
    run_forecast_backtest,
    get_feature_importances,
    compute_revenue_gap,
    DEFAULT_TEST_START,
)

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="BESS Revenue Backtester",
        page_icon="🔋",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

PROCESSED = Path(__file__).parent.parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_auctions() -> pd.DataFrame:
    p = PROCESSED / "auctions.parquet"
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
def load_generation_daily() -> pd.DataFrame:
    p = PROCESSED / "generation_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


auctions    = load_auctions()
mkt_index   = load_market_index()
gen_daily   = load_generation_daily()

# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

st.sidebar.title("Battery Parameters")

power_mw = st.sidebar.slider("Power (MW)", min_value=1, max_value=500, value=50, step=1)

duration_h = st.sidebar.select_slider(
    "Duration (hours)",
    options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    value=2.0,
)

efficiency_pct = st.sidebar.slider(
    "Round-trip efficiency (%)", min_value=80, max_value=98, value=90, step=1
)

cycling_cost = st.sidebar.number_input(
    "Cycling wear cost (£/MWh)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.5,
    help="Cost per MWh of usable energy cycled — accounts for battery degradation.",
)

availability_pct = st.sidebar.slider(
    "Availability factor (%)",
    min_value=80,
    max_value=100,
    value=95,
    step=1,
    help=(
        "Fraction of periods the asset is available. 95% reflects the minimum threshold "
        "specified in NESO's DC and EAC service agreements, and is consistent with observed "
        "GB BESS fleet availability (Modo Energy, 'GB Battery Storage Report', 2024)."
    ),
)

initial_soc_pct = st.sidebar.slider(
    "Initial SoC (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help=(
        "Battery state-of-charge at the start of the backtest period. "
        "SoC is tracked continuously across EFA blocks and days thereafter. "
        "The battery must remain within 10–90% to maintain headroom for "
        "simultaneous DC High (discharge) and DC Low (charge) delivery."
    ),
)

st.sidebar.divider()
st.sidebar.subheader("Services to Include")

selected_services = st.sidebar.multiselect(
    "Frequency response services",
    options=ALL_SERVICES,
    default=ALL_SERVICES,
    format_func=lambda s: f"{s} — {SERVICE_LABELS[s]}",
)

include_imbalance = st.sidebar.checkbox("Include imbalance arbitrage", value=True)

st.sidebar.divider()
st.sidebar.subheader("Date Range")

# Determine available overlap between auction and market index data
if not auctions.empty and not mkt_index.empty:
    data_start = max(auctions["EFA Date"].min(), mkt_index["settlementDate"].min()).date()
    data_end   = min(auctions["EFA Date"].max(), mkt_index["settlementDate"].max()).date()
else:
    import datetime
    data_start = datetime.date(2023, 7, 1)
    data_end   = datetime.date(2026, 2, 19)

date_range = st.sidebar.date_input(
    "Backtest period",
    value=(data_start, data_end),
    min_value=data_start,
    max_value=data_end,
)

start_date = date_range[0] if len(date_range) == 2 else data_start
end_date   = date_range[1] if len(date_range) == 2 else data_end

st.sidebar.divider()
st.sidebar.subheader("Dispatch Strategy")

dispatch_strategy = st.sidebar.radio(
    "Price signal used for arbitrage scheduling",
    options=["Perfect Foresight", "Naive (D-1 prices)", "ML Model"],
    index=0,
    help=(
        "**Perfect Foresight**: schedules dispatch using actual day-D prices — "
        "the revenue ceiling, not achievable in practice.\n\n"
        "**Naive**: uses yesterday's prices as the forecast for today — "
        "a zero-skill baseline.\n\n"
        "**ML Model**: trains on historical features (lagged prices, generation mix, "
        "seasonality) and forecasts day-D prices to drive dispatch."
    ),
)

ml_model_type = "rf"
if dispatch_strategy == "ML Model":
    ml_model_type = st.sidebar.selectbox(
        "Model",
        options=["rf", "xgb"],
        format_func=lambda x: "Random Forest" if x == "rf" else "XGBoost",
        index=0,
    )

st.sidebar.divider()
st.sidebar.subheader("Dispatch Method")

dispatch_method_label = st.sidebar.radio(
    "Optimisation approach",
    options=["Greedy", "MPC (rolling horizon)"],
    index=0,
    help=(
        "**Greedy**: at each EFA block, picks the N cheapest periods to charge "
        "and N most expensive to discharge independently. Fast, no look-ahead.\n\n"
        "**MPC**: re-solves a linear programme every 30 minutes over a rolling horizon. "
        "The FR SoC band [10–90%] is enforced as a hard constraint at all future periods, "
        "forcing the battery to pre-condition its state of charge for upcoming FR delivery "
        "obligations. Slower to compute — expect 1–3 minutes for a full backtest."
    ),
)
use_mpc = dispatch_method_label == "MPC (rolling horizon)"

if use_mpc:
    horizon_periods = st.sidebar.select_slider(
        "MPC horizon",
        options=[48, 72, 96],
        value=96,
        format_func=lambda x: f"{x} periods ({x // 2}h)",
        help="Number of 30-min settlement periods the LP plans over. Longer horizons "
             "allow better SoC pre-conditioning but increase solve time linearly.",
    )
else:
    horizon_periods = 96  # unused when greedy

dm_arg = "mpc" if use_mpc else "greedy"


@st.cache_resource
def load_forecast_model(model_type: str):
    """Train and cache the ML price forecast model. Runs once per deployment."""
    feature_df = build_feature_matrix(load_market_index(), load_generation_daily())
    model, feature_cols, train_metrics, test_metrics = train_forecast_model(
        feature_df, model_type=model_type, test_start=DEFAULT_TEST_START
    )
    return model, feature_df, feature_cols, train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

battery = BatterySpec(
    power_mw=power_mw,
    duration_h=duration_h,
    efficiency_rt=efficiency_pct / 100,
    cycling_cost_per_mwh=cycling_cost,
    availability_factor=availability_pct / 100,
)

mi_input = mkt_index if include_imbalance else pd.DataFrame()

if auctions.empty:
    st.error("No auction data found. Run scripts/prepare_data.py first.")
    st.stop()


# Run the primary backtest for the selected strategy
if dispatch_strategy == "Perfect Foresight":
    result = run_backtest(
        auctions, mi_input, battery, selected_services, start_date, end_date,
        initial_soc_frac=initial_soc_pct / 100,
        dispatch_method=dm_arg, horizon=horizon_periods,
    )
    # Add total_mwh_cycled to summary for the comparison chart
    if result["monthly"] is not None and not result["monthly"].empty:
        mwh_cycled = float(
            (result["monthly"]["cycling_cost_gbp"] / battery.cycling_cost_per_mwh).sum()
            if "cycling_cost_gbp" in result["monthly"].columns and battery.cycling_cost_per_mwh > 0
            else 0.0
        )
        result["summary"]["total_mwh_cycled"] = round(mwh_cycled, 1)

elif dispatch_strategy == "Naive (D-1 prices)":
    if not include_imbalance:
        result = run_backtest(
            auctions, pd.DataFrame(), battery, selected_services, start_date, end_date,
            initial_soc_frac=initial_soc_pct / 100,
            dispatch_method=dm_arg, horizon=horizon_periods,
        )
        result["summary"]["total_mwh_cycled"] = 0.0
    else:
        result = run_forecast_backtest(
            strategy="naive",
            market_index=mkt_index,
            auctions=auctions,
            battery=battery,
            services=selected_services,
            start_date=start_date,
            end_date=end_date,
            initial_soc_frac=initial_soc_pct / 100,
            dispatch_method=dm_arg, horizon=horizon_periods,
        )

else:  # ML Model
    if not include_imbalance:
        result = run_backtest(
            auctions, pd.DataFrame(), battery, selected_services, start_date, end_date,
            initial_soc_frac=initial_soc_pct / 100,
            dispatch_method=dm_arg, horizon=horizon_periods,
        )
        result["summary"]["total_mwh_cycled"] = 0.0
    else:
        with st.spinner("Training forecast model… (cached after first run)"):
            _model, _feature_df, _feature_cols, _train_m, _test_m = load_forecast_model(ml_model_type)
        result = run_forecast_backtest(
            strategy="ml",
            market_index=mkt_index,
            auctions=auctions,
            battery=battery,
            services=selected_services,
            start_date=start_date,
            end_date=end_date,
            model=_model,
            feature_df=_feature_df,
            feature_cols=_feature_cols,
            initial_soc_frac=initial_soc_pct / 100,
            dispatch_method=dm_arg, horizon=horizon_periods,
        )

monthly = result["monthly"]
summary = result["summary"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("BESS Revenue Backtester")
st.markdown(
    f"Modelling a **{power_mw} MW / {power_mw * duration_h:.0f} MWh** battery "
    f"({duration_h}h duration, {efficiency_pct}% round-trip efficiency) "
    f"from **{start_date}** to **{end_date}**."
)

# ---------------------------------------------------------------------------
# Content tabs
# ---------------------------------------------------------------------------

tab_results, tab_strategy, tab_sensitivity = st.tabs(
    ["Results", "Strategy Comparison", "Sensitivity"]
)

# ---------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------

with tab_results:
    if summary:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Total Net Revenue",
            f"£{summary['total_net'] / 1_000:,.0f}k",
        )
        c2.metric(
            "Annualised Net Revenue",
            f"£{summary['annualised_net'] / 1_000:,.0f}k / yr",
        )
        c3.metric(
            "Revenue per MW",
            f"£{summary['annualised_per_mw'] / 1_000:,.1f}k / MW / yr",
        )
        c4.metric(
            "Top Revenue Stream",
            SERVICE_LABELS.get(summary.get("top_service", ""), summary.get("top_service", "N/A")),
        )
        c5.metric(
            "Avg Capacity Split",
            f"{summary.get('fr_mw', 0):.0f} MW FR / {summary.get('arb_mw', 0):.0f} MW arb",
            help="Average daily FR/arb allocation over the backtest period.",
        )
    else:
        st.warning("No results — check that services are selected and data covers the chosen period.")
        st.stop()

    st.divider()

    st.caption(
        f"Charts below model a **{power_mw} MW / {power_mw * duration_h:.0f} MWh** asset "
        f"using **{dispatch_strategy}** price signals and **{dispatch_method_label}** dispatch. "
        "FR availability fees are earned on the FR-committed portion of capacity; wholesale "
        "arbitrage is modelled on the remainder. Cycling wear cost is deducted from gross "
        "revenue to arrive at the net figures shown in the header metrics above."
    )

    # Monthly stacked bar — revenue by stream + cycling cost deduction
    if not monthly.empty:
        st.subheader("Monthly Revenue Stack")

        fig = go.Figure()

        stream_cols = {
            **{f"{s}_rev": s for s in ALL_SERVICES},
            "imbalance_revenue_gbp": "Imbalance",
        }

        for col, label in stream_cols.items():
            if col not in monthly.columns:
                continue
            if monthly[col].sum() == 0:
                continue
            display_label = SERVICE_LABELS.get(label, label)
            fig.add_trace(go.Bar(
                x=monthly["month_dt"],
                y=monthly[col] / 1_000,
                name=display_label,
                marker_color=SERVICE_COLOURS.get(label, "#888"),
            ))

        if "cycling_cost_gbp" in monthly.columns and monthly["cycling_cost_gbp"].sum() > 0:
            fig.add_trace(go.Bar(
                x=monthly["month_dt"],
                y=-monthly["cycling_cost_gbp"] / 1_000,
                name="Cycling wear cost",
                marker_color=SERVICE_COLOURS["Cycling cost"],
                opacity=0.8,
            ))

        fig.update_layout(
            barmode="relative",
            height=450,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            yaxis_title="£k",
            xaxis_title="Month",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each bar shows gross revenue by stream for that month (positive) and cycling "
            "wear cost (negative, dark red). Net revenue is the algebraic sum of all segments. "
            "Months with heavier arbitrage dispatch show larger cycling cost deductions."
        )

    # SoC trajectory — only shown when MPC dispatch is selected
    soc_traj = result.get("soc_trajectory")
    if use_mpc and soc_traj is not None and not soc_traj.empty:
        st.subheader("SoC Trajectory (MPC)")
        soc_plot = soc_traj.copy()
        soc_plot["timestamp"] = (
            pd.to_datetime(soc_plot["date"])
            + pd.to_timedelta((soc_plot["sp"] - 1) * 30, unit="min")
        )

        fig_soc = go.Figure()
        # FR band shading
        fig_soc.add_hrect(
            y0=0.10, y1=0.90,
            fillcolor="rgba(13,118,128,0.08)",
            line_width=0,
        )
        # SoC line
        fig_soc.add_trace(go.Scatter(
            x=soc_plot["timestamp"],
            y=soc_plot["soc_frac"],
            mode="lines",
            line=dict(color="#C9400A", width=1.2),
            name="SoC",
        ))
        # Band boundaries
        fig_soc.add_hline(
            y=0.10, line_dash="dash", line_color="#0D7680", line_width=1,
            annotation_text="10%", annotation_position="right",
        )
        fig_soc.add_hline(
            y=0.90, line_dash="dash", line_color="#0D7680", line_width=1,
            annotation_text="90%", annotation_position="right",
        )
        fig_soc.update_layout(
            height=280,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            yaxis=dict(title="SoC", tickformat=".0%", range=[0, 1]),
            xaxis_title="Date",
            showlegend=False,
            margin=dict(t=10, b=40, r=60),
        )
        st.plotly_chart(fig_soc, use_container_width=True)
        st.caption(
            "MPC state-of-charge trajectory across the backtest. The shaded region marks the "
            "FR feasibility band [10–90%] — enforced as a hard constraint at every future "
            "period in the rolling LP horizon."
        )

    # Cumulative revenue | Revenue breakdown pie
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Cumulative Revenue by Stream")
        monthly_sorted = monthly.sort_values("month_dt")

        stream_cols_cum = {
            **{f"{s}_rev": s for s in ALL_SERVICES},
            "imbalance_revenue_gbp": "Imbalance",
        }

        fig2 = go.Figure()
        for col, label in stream_cols_cum.items():
            if col not in monthly_sorted.columns:
                continue
            if monthly_sorted[col].sum() == 0:
                continue
            display_label = SERVICE_LABELS.get(label, label)
            fig2.add_trace(go.Scatter(
                x=monthly_sorted["month_dt"],
                y=monthly_sorted[col].cumsum() / 1_000,
                name=display_label,
                mode="lines",
                stackgroup="one",
                line=dict(color=SERVICE_COLOURS.get(label, "#888"), width=0.5),
            ))

        fig2.update_layout(
            height=380,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            yaxis_title="£k (cumulative)",
            xaxis_title="Month",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Cumulative gross revenue by income stream. The height of each coloured band "
            "shows how much that stream contributed over the backtest period. "
            "Cycling wear cost is shown separately in the monthly bar chart above."
        )

    with right:
        st.subheader("Revenue Breakdown")
        bd = summary.get("breakdown", {})
        if bd:
            labels = [SERVICE_LABELS.get(k, k) for k in bd.keys()]
            values = list(bd.values())
            colours = [SERVICE_COLOURS.get(k, "#888") for k in bd.keys()]

            fig3 = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker_colors=colours,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="%{label}: £%{value:,.0f}<extra></extra>",
            ))
            fig3.update_layout(height=380, showlegend=False, margin=dict(t=20, b=20), paper_bgcolor="#FFF1E5")
            st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: Strategy Comparison
# ---------------------------------------------------------------------------

with tab_strategy:
    st.caption(
        "Compares **Greedy** and **MPC** dispatch for the currently selected price signal. "
        "Identical battery parameters, date range, and services — only the dispatch method differs. "
        "A strategy to the upper-left earns more revenue while consuming less cycle life."
    )

    if include_imbalance and not mkt_index.empty:
        model_label = "Random Forest" if ml_model_type == "rf" else "XGBoost"

        # Load ML model before the comparison spinner (cached, fast after first run)
        if dispatch_strategy == "ML Model":
            with st.spinner("Training ML model for comparison… (cached after first run)"):
                _cmp_model, _cmp_feat_df, _cmp_feat_cols, _, _ = load_forecast_model(ml_model_type)

        with st.spinner("Computing dispatch comparison (Greedy + MPC — MPC may take 1–3 min)…"):
            if dispatch_strategy == "Perfect Foresight":
                greedy_cmp = run_backtest(
                    auctions, mi_input, battery, selected_services, start_date, end_date,
                    initial_soc_frac=initial_soc_pct / 100, dispatch_method="greedy",
                )
                mpc_cmp = run_backtest(
                    auctions, mi_input, battery, selected_services, start_date, end_date,
                    initial_soc_frac=initial_soc_pct / 100,
                    dispatch_method="mpc", horizon=horizon_periods,
                )
            elif dispatch_strategy == "Naive (D-1 prices)":
                greedy_cmp = run_forecast_backtest(
                    strategy="naive", market_index=mkt_index, auctions=auctions,
                    battery=battery, services=selected_services,
                    start_date=start_date, end_date=end_date,
                    initial_soc_frac=initial_soc_pct / 100, dispatch_method="greedy",
                )
                mpc_cmp = run_forecast_backtest(
                    strategy="naive", market_index=mkt_index, auctions=auctions,
                    battery=battery, services=selected_services,
                    start_date=start_date, end_date=end_date,
                    initial_soc_frac=initial_soc_pct / 100,
                    dispatch_method="mpc", horizon=horizon_periods,
                )
            else:  # ML Model
                greedy_cmp = run_forecast_backtest(
                    strategy="ml", market_index=mkt_index, auctions=auctions,
                    battery=battery, services=selected_services,
                    start_date=start_date, end_date=end_date,
                    model=_cmp_model, feature_df=_cmp_feat_df, feature_cols=_cmp_feat_cols,
                    initial_soc_frac=initial_soc_pct / 100, dispatch_method="greedy",
                )
                mpc_cmp = run_forecast_backtest(
                    strategy="ml", market_index=mkt_index, auctions=auctions,
                    battery=battery, services=selected_services,
                    start_date=start_date, end_date=end_date,
                    model=_cmp_model, feature_df=_cmp_feat_df, feature_cols=_cmp_feat_cols,
                    initial_soc_frac=initial_soc_pct / 100,
                    dispatch_method="mpc", horizon=horizon_periods,
                )

        greedy_net = greedy_cmp["summary"].get("total_net", 0) / 1_000
        mpc_net    = mpc_cmp["summary"].get("total_net",    0) / 1_000

        def _mwh(r):
            m = r.get("monthly")
            if m is None or m.empty or "cycling_cost_gbp" not in m.columns:
                return 0.0
            return float((m["cycling_cost_gbp"] / battery.cycling_cost_per_mwh).sum()) if battery.cycling_cost_per_mwh > 0 else 0.0

        greedy_mwh    = _mwh(greedy_cmp)
        mpc_mwh       = _mwh(mpc_cmp)
        mpc_vs_greedy = round(mpc_net / greedy_net * 100, 1) if greedy_net > 0 else 0.0

        comparison_df = pd.DataFrame([
            {"Method": "Greedy",        "Net Revenue (£k)": greedy_net, "MWh Cycled": round(greedy_mwh, 0), "vs Greedy": "—"},
            {"Method": "MPC (rolling)", "Net Revenue (£k)": mpc_net,    "MWh Cycled": round(mpc_mwh,    0), "vs Greedy": f"{mpc_vs_greedy:.1f}%"},
        ])

        fig_cmp = go.Figure()
        cmp_colours = {"Greedy": "#4E8A3C", "MPC (rolling)": "#0D7680"}
        for _, row in comparison_df.iterrows():
            fig_cmp.add_trace(go.Scatter(
                x=[row["MWh Cycled"]],
                y=[row["Net Revenue (£k)"]],
                mode="markers+text",
                name=row["Method"],
                text=[row["Method"]],
                textposition="top center",
                marker=dict(size=16, color=cmp_colours.get(row["Method"], "#888")),
            ))

        fig_cmp.update_layout(
            height=360,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            xaxis_title="Total MWh Cycled",
            yaxis_title="Total Net Revenue (£k)",
            showlegend=False,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.dataframe(
            comparison_df.style.format({"Net Revenue (£k)": "{:,.1f}", "MWh Cycled": "{:,.0f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Price signal: **{dispatch_strategy}**. "
            "**vs Greedy** = MPC net revenue ÷ Greedy net revenue for this price signal. "
            "Comparison isolates the effect of dispatch method holding forecast signal fixed."
        )

        # ML model detail — shown when ML strategy is selected
        if dispatch_strategy == "ML Model":
            st.divider()
            st.subheader(f"ML Model Detail — {model_label}")
            st.markdown(f"""
The ML strategy uses a **{model_label}** regressor to predict the 48 half-hourly APXMIDP
prices for day D using features available at the end of day D-1.

*Why {model_label}?* Tree-based ensemble methods are well suited to this problem: the
feature set is tabular (lagged prices, generation mix ratios, temporal encodings) rather
than raw sequences; they require no feature scaling; they are robust on datasets of this
size (~30,000 training rows); and they provide interpretable feature importances.
An LSTM was considered but is likely overkill given ~2 years of training data and would
be harder to explain. A naive lag model sets the zero-skill baseline.

**Features used (all available at end of day D-1):**
- Same-period lagged prices: price at the same settlement period 1, 2, 7, and 14 days prior
- Previous-day price statistics: mean, standard deviation, max, and min across all 48 periods
- 7-day rolling mean of daily average price (captures medium-term price level shifts)
- Generation mix (daily, from D-1): total generation, renewable fraction, fossil fraction,
  and per-fuel breakdown (gas, wind, nuclear, hydro, etc.)
- Cyclical temporal encodings: settlement period, day-of-week, and day-of-year encoded
  as sin/cos pairs to preserve circularity (e.g. period 48 and period 1 are adjacent)
- Weekend flag; UK bank holiday flag (holidays have Sunday-like price profiles)

**Target transform:** a signed-log1p transform is applied to prices before fitting
(`sign(y)·log1p(|y|)`) and inverted on predictions. This compresses the heavy-tailed
price distribution so the model does not under-weight high-price periods during training.

**Train/test split:** strict temporal split — training data ends before
`{DEFAULT_TEST_START}` to prevent any look-ahead bias. The model never sees
future prices during training.

**Known limitations:** tree-based models cannot extrapolate beyond the price ranges seen
during training; electricity price forecasting is inherently noisy; and the model improves
dispatch quality on average but does not eliminate forecast error on individual days.
            """)

            with st.spinner("Loading model metrics…"):
                _model_exp, _, _feat_cols_exp, _train_m, _test_m = load_forecast_model(ml_model_type)

            col_a, col_b = st.columns(2)
            col_a.metric("Train RMSE (£/MWh)", f"{_train_m['rmse']:.2f}", help=f"n = {_train_m['n_samples']:,} periods")
            col_b.metric("Test RMSE (£/MWh)",  f"{_test_m['rmse']:.2f}",  help=f"n = {_test_m['n_samples']:,} periods (held-out, after {DEFAULT_TEST_START})")
            col_a.metric("Train MAE (£/MWh)",  f"{_train_m['mae']:.2f}")
            col_b.metric("Test MAE (£/MWh)",   f"{_test_m['mae']:.2f}")
            col_a.metric(
                "Train Spearman ρ", f"{_train_m['spearman']:.3f}",
                help="Rank correlation of 48-period ordering vs actuals. Ordinal accuracy governs greedy dispatch quality.",
            )
            col_b.metric(
                "Test Spearman ρ",  f"{_test_m['spearman']:.3f}",
                help="Rank correlation of 48-period ordering vs actuals. Ordinal accuracy governs greedy dispatch quality.",
            )
            if "spike_rmse" in _test_m:
                col_a.metric(
                    "Train Spike-RMSE (£/MWh)", f"{_train_m.get('spike_rmse', '—')}",
                    help="RMSE on top-decile price periods — where BESS arbitrage revenue is concentrated.",
                )
                col_b.metric(
                    "Test Spike-RMSE (£/MWh)",  f"{_test_m['spike_rmse']}",
                    help="RMSE on top-decile price periods — where BESS arbitrage revenue is concentrated.",
                )

            st.caption(
                "**Interpretation:** MAE < 15 £/MWh is adequate for greedy dispatch; "
                "MAE < 8 £/MWh is 'good' for MPC. "
                "Spearman ρ > 0.8 indicates strong ordinal ranking accuracy. "
                "Spike-RMSE reflects model quality on the high-price periods that drive arbitrage revenue."
            )

            # Revenue gap: fraction of the perfect-foresight improvement over naive
            # that the ML forecast actually captures.
            if include_imbalance:
                with st.spinner("Computing revenue gap benchmarks (naive & perfect foresight)…"):
                    _naive_ref = run_forecast_backtest(
                        strategy="naive",
                        market_index=mkt_index, auctions=auctions,
                        battery=battery, services=selected_services,
                        start_date=start_date, end_date=end_date,
                        initial_soc_frac=initial_soc_pct / 100,
                        dispatch_method="greedy",
                    )
                    _pf_ref = run_backtest(
                        auctions, mkt_index, battery, selected_services,
                        start_date, end_date,
                        initial_soc_frac=initial_soc_pct / 100,
                        dispatch_method="greedy",
                    )
                _ml_net    = result["summary"].get("total_net", 0)
                _naive_net = _naive_ref["summary"].get("total_net", 0)
                _pf_net    = _pf_ref["summary"].get("total_net", 0)
                _gap       = compute_revenue_gap(_ml_net, _naive_net, _pf_net)

                if _gap is not None:
                    col_g1, col_g2, col_g3 = st.columns(3)
                    col_g1.metric(
                        "Naive net revenue",
                        f"£{_naive_net:,.0f}",
                        help="Zero-skill baseline: dispatch using yesterday's prices as forecast.",
                    )
                    col_g2.metric(
                        "ML net revenue",
                        f"£{_ml_net:,.0f}",
                        delta=f"£{_ml_net - _naive_net:+,.0f} vs naive",
                    )
                    col_g3.metric(
                        "Perfect foresight net",
                        f"£{_pf_net:,.0f}",
                        help="Upper bound: dispatch using actual prices known in advance.",
                    )
                    st.metric(
                        "Revenue gap",
                        f"{_gap:.1%}",
                        help=(
                            "Fraction of the theoretically capturable improvement over naive "
                            "that the ML forecast actually delivers. "
                            "gap = (ML − naive) / (perfect − naive). "
                            "Literature benchmark: ML typically achieves 70–85% in normal markets."
                        ),
                    )

            importances = get_feature_importances(_model_exp, _feat_cols_exp).head(10)
            fig_imp = go.Figure(go.Bar(
                x=importances.values[::-1],
                y=importances.index[::-1],
                orientation="h",
                marker_color="#0D7680",
            ))
            fig_imp.update_layout(
                height=320,
                template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
                title="Top 10 feature importances",
                xaxis_title="Importance",
                margin=dict(t=40, l=160),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.info("Enable 'Include imbalance arbitrage' to see the dispatch comparison.")

# ---------------------------------------------------------------------------
# Tab: Sensitivity
# ---------------------------------------------------------------------------

with tab_sensitivity:
    st.subheader("Sensitivity: Revenue by Battery Size")
    st.caption("Other parameters held constant at sidebar values.")

    sens_df = sensitivity_table(
        auctions,
        mi_input,
        battery,
        power_range=[5, 10, 25, 50, 100, 200],
        start_date=start_date,
        end_date=end_date,
    )

    st.dataframe(
        sens_df.style.format({
            "Energy (MWh)":             "{:.0f}",
            "Total Net Revenue (£k)":   "{:,.1f}",
            "Ann. Net Revenue (£k/yr)": "{:,.1f}",
            "Revenue / MW (£k/MW/yr)":  "{:,.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
