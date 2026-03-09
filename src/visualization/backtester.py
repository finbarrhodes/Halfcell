"""
BESS Revenue Backtester — Viewer
==================================
Pure viewer: all heavy computation is done offline via scripts/precompute_cache.py.

Can still be run standalone for local development:
    streamlit run src/visualization/backtester.py
"""

import json
import sys
from pathlib import Path

# Ensure src/ is on the path when launched standalone from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.price_forecast import (
    DEFAULT_TEST_START,
    build_feature_matrix,
    compute_revenue_gap,
    get_feature_importances,
    train_forecast_model,
)
from src.analysis.revenue_stack import (
    ALL_SERVICES,
    SERVICE_COLOURS,
    SERVICE_LABELS,
    BatterySpec,
    sensitivity_table,
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
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# Fixed battery configuration (all parameters documented in methodology expander below)
FIXED_BATTERY = BatterySpec(
    power_mw=50.0,
    duration_h=2.0,
    efficiency_rt=0.90,
    cycling_cost_per_mwh=3.0,
    availability_factor=0.95,
)
BASE_POWER_MW = 50.0  # all cache figures are at this MW rating

# Map sidebar radio label → cache file key
STRATEGY_KEYS = {
    "Perfect Foresight": "pf_mpc",
    "Naive (D-1 prices)": "naive_mpc",
    "ML Model": "ml_mpc",
}


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_monthly(strategy_key: str) -> pd.DataFrame | None:
    p = CACHE_DIR / f"{strategy_key}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["month_dt"] = pd.to_datetime(df["month_dt"])
    return df


@st.cache_data
def load_soc(strategy_key: str) -> pd.DataFrame | None:
    p = CACHE_DIR / f"soc_{strategy_key}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_manifest() -> dict:
    p = CACHE_DIR / "manifest.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Source data (used only for the Sensitivity tab)
# ---------------------------------------------------------------------------

@st.cache_data
def load_auctions() -> pd.DataFrame:
    p = PROCESSED / "auctions.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_market_index() -> pd.DataFrame:
    p = PROCESSED / "market_index.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_generation_daily() -> pd.DataFrame:
    p = PROCESSED / "generation_daily.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_resource
def load_forecast_model(model_type: str = "rf"):
    """Train and cache the ML model for feature importance display (not for backtesting)."""
    feature_df = build_feature_matrix(load_market_index(), load_generation_daily())
    model, feature_cols, train_metrics, test_metrics = train_forecast_model(
        feature_df, model_type=model_type, test_start=DEFAULT_TEST_START
    )
    return model, feature_cols, train_metrics, test_metrics


# ---------------------------------------------------------------------------
# Post-filtering & summary helpers
# ---------------------------------------------------------------------------

def _apply_filters(
    monthly: pd.DataFrame,
    start_date,
    end_date,
    selected_services: list,
    power_mw: float,
) -> pd.DataFrame:
    """Filter cached monthly data by date and services, then scale to display power."""
    df = monthly.copy()

    # Date filter (inclusive month-level bounds)
    start_ts = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end_ts   = pd.Timestamp(end_date).to_period("M").to_timestamp("M")
    df = df[(df["month_dt"] >= start_ts) & (df["month_dt"] <= end_ts)]

    # Service filter — zero out excluded services
    excluded = [s for s in ALL_SERVICES if s not in selected_services]
    for s in excluded:
        col = f"{s}_rev"
        if col in df.columns:
            df[col] = 0.0

    # Power scaling (all cache figures are at BASE_POWER_MW = 50 MW)
    scale     = power_mw / BASE_POWER_MW
    rev_cols  = [f"{s}_rev" for s in ALL_SERVICES]
    rev_cols += ["imbalance_revenue_gbp", "cycling_cost_gbp"]
    for col in rev_cols:
        if col in df.columns:
            df[col] = df[col] * scale

    return df


def _recompute_summary(monthly: pd.DataFrame, power_mw: float) -> dict:
    """Derive headline metrics from a (possibly filtered + scaled) monthly DataFrame."""
    if monthly.empty:
        return {}

    svc_totals = {s: monthly[f"{s}_rev"].sum() for s in ALL_SERVICES if f"{s}_rev" in monthly.columns}
    imb_rev    = monthly["imbalance_revenue_gbp"].sum() if "imbalance_revenue_gbp" in monthly.columns else 0.0
    cycling    = monthly["cycling_cost_gbp"].sum()      if "cycling_cost_gbp" in monthly.columns else 0.0

    total_gross = sum(svc_totals.values()) + imb_rev
    total_net   = total_gross - cycling

    months_n      = len(monthly)
    years_covered = months_n / 12
    ann_net       = total_net   / years_covered if years_covered > 0 else 0.0
    ann_per_mw    = ann_net     / power_mw      if power_mw > 0 else 0.0

    breakdown = {**svc_totals, "Imbalance": imb_rev}
    breakdown = {k: v for k, v in breakdown.items() if v > 0}
    top_service = max(breakdown, key=breakdown.get) if breakdown else ""

    return {
        "total_gross":      round(total_gross, 0),
        "total_cycling_cost": round(cycling, 0),
        "total_net":        round(total_net, 0),
        "years_covered":    round(years_covered, 2),
        "annualised_net":   round(ann_net, 0),
        "annualised_per_mw": round(ann_per_mw, 0),
        "breakdown":        breakdown,
        "top_service":      top_service,
    }


# ---------------------------------------------------------------------------
# Cache presence check
# ---------------------------------------------------------------------------

manifest = load_manifest()
_missing = [k for k in STRATEGY_KEYS.values() if not (CACHE_DIR / f"{k}.parquet").exists()]

if _missing:
    st.error(
        "Pre-computed results not found — run `scripts/precompute_cache.py` to generate them.\n\n"
        f"Missing files: {', '.join(f'data/cache/{k}.parquet' for k in _missing)}"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar — viewer controls only (no recompute triggered)
# ---------------------------------------------------------------------------

st.sidebar.title("Battery Parameters")

power_mw = st.sidebar.slider(
    "Power (MW)",
    min_value=1, max_value=500, value=50, step=1,
    help=(
        "Display scaling only — scales all revenue figures linearly. "
        "The underlying backtest uses a fixed 50 MW / 100 MWh asset; "
        "all revenue streams are proportional to power, so scaling is exact."
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

# Determine bounds from manifest (full cache range)
_manifest_start = None
_manifest_end   = None
for k in STRATEGY_KEYS.values():
    if k in manifest:
        p = manifest[k].get("params", {})
        s = p.get("start_date"); e = p.get("end_date")
        if s and (_manifest_start is None or s < _manifest_start):
            _manifest_start = s
        if e and (_manifest_end is None or e > _manifest_end):
            _manifest_end = e

import datetime as _dt
_fallback_start = _dt.date(2023, 7, 1)
_fallback_end   = _dt.date(2026, 2, 19)
data_start = _dt.date.fromisoformat(_manifest_start) if _manifest_start else _fallback_start
data_end   = _dt.date.fromisoformat(_manifest_end)   if _manifest_end   else _fallback_end

date_range = st.sidebar.date_input(
    "Date range (post-filter on cached data)",
    value=(data_start, data_end),
    min_value=data_start,
    max_value=data_end,
)
start_date = date_range[0] if len(date_range) == 2 else data_start
end_date   = date_range[1] if len(date_range) == 2 else data_end

st.sidebar.divider()
st.sidebar.subheader("Strategy")

dispatch_strategy = st.sidebar.radio(
    "Price signal used for arbitrage scheduling",
    options=list(STRATEGY_KEYS.keys()),
    index=0,
    help=(
        "**Perfect Foresight**: revenue ceiling — dispatches using actual day-D prices.\n\n"
        "**Naive**: zero-skill floor — uses yesterday's prices as today's forecast.\n\n"
        "**ML Model**: realistic best case — Random Forest trained on lagged prices, "
        "generation mix, and seasonality features."
    ),
)
cache_key = STRATEGY_KEYS[dispatch_strategy]

# Show cache vintage in sidebar
if cache_key in manifest:
    _ts = manifest[cache_key].get("computed_at", "")
    if _ts:
        try:
            _computed = _dt.datetime.fromisoformat(_ts).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            _computed = _ts
        st.sidebar.caption(f"Cache computed: {_computed}")


# ---------------------------------------------------------------------------
# Load and filter cached data for the selected strategy
# ---------------------------------------------------------------------------

_monthly_raw = load_monthly(cache_key)
if _monthly_raw is None:
    st.error(
        f"Cache file `data/cache/{cache_key}.parquet` not found. "
        "Run `scripts/precompute_cache.py` to generate it."
    )
    st.stop()

monthly = _apply_filters(_monthly_raw, start_date, end_date, selected_services, power_mw)
if not include_imbalance and "imbalance_revenue_gbp" in monthly.columns:
    monthly["imbalance_revenue_gbp"] = 0.0
    if "cycling_cost_gbp" in monthly.columns:
        monthly["cycling_cost_gbp"] = 0.0  # no arbitrage → no cycling cost

summary = _recompute_summary(monthly, power_mw)

# SoC trajectory (filtered by date, scaled by power — positions are fractional so no MW scaling)
_soc_raw = load_soc(cache_key)
soc_traj = None
if _soc_raw is not None and not _soc_raw.empty:
    _soc = _soc_raw.copy()
    _soc["date"] = pd.to_datetime(_soc["date"])
    soc_traj = _soc[
        (_soc["date"] >= pd.Timestamp(start_date)) &
        (_soc["date"] <= pd.Timestamp(end_date))
    ]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("BESS Revenue Backtester")
st.markdown(
    f"Modelling a **{power_mw} MW / {power_mw * FIXED_BATTERY.duration_h:.0f} MWh** battery "
    f"({FIXED_BATTERY.duration_h:.0f}h duration, {FIXED_BATTERY.efficiency_rt*100:.0f}% "
    f"round-trip efficiency) from **{start_date}** to **{end_date}**."
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
    if not summary:
        st.warning("No results for the selected date range / services.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
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

    st.divider()

    st.caption(
        f"Charts below model a **{power_mw} MW / {power_mw * FIXED_BATTERY.duration_h:.0f} MWh** asset "
        f"using **{dispatch_strategy}** price signals and MPC (rolling horizon, 48h) dispatch. "
        "FR availability fees are earned on the FR-committed portion of capacity; wholesale "
        "arbitrage is modelled on the remainder. Cycling wear cost is deducted from gross "
        "revenue to arrive at the net figures shown in the header metrics above."
    )

    # Monthly stacked bar — revenue by stream + cycling cost deduction
    if not monthly.empty:
        st.subheader("Monthly Revenue Stack")

        stream_cols = {
            **{f"{s}_rev": s for s in ALL_SERVICES},
            "imbalance_revenue_gbp": "Imbalance",
        }

        fig = go.Figure()
        for col, label in stream_cols.items():
            if col not in monthly.columns or monthly[col].sum() == 0:
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

    # SoC trajectory
    if soc_traj is not None and not soc_traj.empty:
        st.subheader("SoC Trajectory (MPC)")
        soc_plot = soc_traj.copy()
        soc_plot["timestamp"] = (
            pd.to_datetime(soc_plot["date"])
            + pd.to_timedelta((soc_plot["sp"] - 1) * 30, unit="min")
        )

        fig_soc = go.Figure()
        fig_soc.add_hrect(
            y0=0.10, y1=0.90,
            fillcolor="rgba(13,118,128,0.08)",
            line_width=0,
        )
        fig_soc.add_trace(go.Scatter(
            x=soc_plot["timestamp"],
            y=soc_plot["soc_frac"],
            mode="lines",
            line=dict(color="#C9400A", width=1.2),
            name="SoC",
        ))
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

        fig2 = go.Figure()
        for col, label in {
            **{f"{s}_rev": s for s in ALL_SERVICES},
            "imbalance_revenue_gbp": "Imbalance",
        }.items():
            if col not in monthly_sorted.columns or monthly_sorted[col].sum() == 0:
                continue
            fig2.add_trace(go.Scatter(
                x=monthly_sorted["month_dt"],
                y=monthly_sorted[col].cumsum() / 1_000,
                name=SERVICE_LABELS.get(label, label),
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
            "shows how much that stream contributed over the backtest period."
        )

    with right:
        st.subheader("Revenue Breakdown")
        bd = summary.get("breakdown", {})
        if bd:
            labels  = [SERVICE_LABELS.get(k, k) for k in bd]
            values  = list(bd.values())
            colours = [SERVICE_COLOURS.get(k, "#888") for k in bd]

            fig3 = go.Figure(go.Pie(
                labels=labels,
                values=values,
                marker_colors=colours,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="%{label}: £%{value:,.0f}<extra></extra>",
            ))
            fig3.update_layout(
                height=380, showlegend=False,
                margin=dict(t=20, b=20),
                paper_bgcolor="#FFF1E5",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Methodology expander
    with st.expander("Methodology — fixed battery parameters"):
        st.markdown(f"""
All backtests use a fixed **50 MW / 100 MWh** (2h) representative GB BESS asset.
Revenue figures are scaled linearly to the display power ({power_mw} MW) via the sidebar.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Power | 50 MW | Median representative GB BESS site |
| Duration | 2h | Most common duration in GB BESS fleet |
| Round-trip efficiency | 90% | Industry standard for modern Li-ion (NESO / Modo Energy fleet data) |
| Availability factor | 95% | Minimum threshold in DC/EAC service agreements; consistent with observed GB fleet |
| Initial SoC | 50% | Neutral midpoint; SoC tracked continuously across all blocks and days thereafter |
| Cycling wear cost | £3/MWh | Mid-range estimate consistent with published Li-ion degradation literature |
| Dispatch engine | MPC (48h rolling LP) | Re-solves a linear programme every 30 minutes; enforces FR SoC band [10–90%] as hard constraint, forcing correct pre-conditioning |
        """)


# ---------------------------------------------------------------------------
# Tab: Strategy Comparison
# ---------------------------------------------------------------------------

with tab_strategy:
    st.caption(
        "Compares **Perfect Foresight** (revenue ceiling), **Naive D-1** (zero-skill floor), "
        "and **ML Model** (realistic best case) — all using identical MPC dispatch and "
        "the fixed 50 MW / 2h battery.  Date range and services filters apply."
    )

    # Load and filter all three cached results
    _all = {}
    _missing_strats = []
    for label, key in STRATEGY_KEYS.items():
        _m = load_monthly(key)
        if _m is None:
            _missing_strats.append(key)
            continue
        _filtered = _apply_filters(_m, start_date, end_date, selected_services, power_mw)
        if not include_imbalance and "imbalance_revenue_gbp" in _filtered.columns:
            _filtered["imbalance_revenue_gbp"] = 0.0
            if "cycling_cost_gbp" in _filtered.columns:
                _filtered["cycling_cost_gbp"] = 0.0
        _all[label] = _recompute_summary(_filtered, power_mw)

    if _missing_strats:
        st.warning(
            f"Some cache files are missing ({', '.join(_missing_strats)}). "
            "Run `scripts/precompute_cache.py` to regenerate."
        )
    elif _all:
        pf_net    = _all["Perfect Foresight"].get("total_net", 0)
        naive_net = _all["Naive (D-1 prices)"].get("total_net", 0)
        ml_net    = _all["ML Model"].get("total_net", 0)

        # Revenue gap
        gap = compute_revenue_gap(ml_net, naive_net, pf_net)

        # Bar chart: annualised per MW for each strategy
        strat_labels = ["Naive\n(D-1 prices)", "ML Model\n(Random Forest)", "Perfect Foresight"]
        strat_vals   = [
            _all["Naive (D-1 prices)"].get("annualised_per_mw", 0) / 1_000,
            _all["ML Model"].get("annualised_per_mw", 0) / 1_000,
            _all["Perfect Foresight"].get("annualised_per_mw", 0) / 1_000,
        ]
        strat_colours = ["#C9400A", "#0D7680", "#4E8A3C"]

        fig_cmp = go.Figure(go.Bar(
            x=strat_labels,
            y=strat_vals,
            marker_color=strat_colours,
            text=[f"£{v:,.1f}k" for v in strat_vals],
            textposition="outside",
        ))
        fig_cmp.update_layout(
            height=380,
            template="plotly_white", paper_bgcolor="#FFF1E5", plot_bgcolor="#FFF1E5",
            yaxis_title="Annualised net revenue (£k / MW / yr)",
            xaxis_title=None,
            showlegend=False,
            margin=dict(t=40, b=40),
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Summary table
        cmp_rows = []
        for label in ["Naive (D-1 prices)", "ML Model", "Perfect Foresight"]:
            s = _all[label]
            cmp_rows.append({
                "Strategy":                label,
                "Total Net Rev (£k)":      round(s["total_net"]        / 1_000, 1),
                "Ann. Net Rev (£k/yr)":    round(s["annualised_net"]   / 1_000, 1),
                "Rev / MW (£k/MW/yr)":     round(s["annualised_per_mw"] / 1_000, 1),
            })
        cmp_df = pd.DataFrame(cmp_rows)
        st.dataframe(
            cmp_df.style.format({
                "Total Net Rev (£k)":   "{:,.1f}",
                "Ann. Net Rev (£k/yr)": "{:,.1f}",
                "Rev / MW (£k/MW/yr)":  "{:,.1f}",
            }),
            hide_index=True,
            use_container_width=True,
        )

        if gap is not None:
            col_g1, col_g2, col_g3, col_g4 = st.columns(4)
            col_g1.metric(
                "Naive net revenue",
                f"£{naive_net / 1_000:,.0f}k",
                help="Zero-skill baseline — dispatch using yesterday's prices as forecast.",
            )
            col_g2.metric(
                "ML net revenue",
                f"£{ml_net / 1_000:,.0f}k",
                delta=f"£{(ml_net - naive_net) / 1_000:+,.0f}k vs naive",
            )
            col_g3.metric(
                "Perfect Foresight net",
                f"£{pf_net / 1_000:,.0f}k",
                help="Upper bound — dispatch using actual prices known in advance.",
            )
            col_g4.metric(
                "Revenue gap",
                f"{gap:.1%}",
                help=(
                    "Fraction of the theoretically capturable improvement over naive "
                    "that the ML forecast actually delivers. "
                    "gap = (ML − naive) / (perfect − naive). "
                    "Literature benchmark: ML typically achieves 70–85% in normal markets."
                ),
            )
        st.caption(
            "All three strategies use identical MPC dispatch with the fixed 50 MW / 2h battery. "
            "The chart isolates the value of forecast quality: Naive sets the zero-skill floor, "
            "Perfect Foresight the ceiling, and ML captures a realistic fraction of the gap."
        )

    # ML model detail
    st.divider()
    st.subheader("ML Model Detail — Random Forest")
    st.markdown(f"""
The ML strategy uses a **Random Forest** regressor to predict the 48 half-hourly APXMIDP
prices for day D using features available at the end of day D-1.

*Why Random Forest?* Tree-based ensemble methods are well suited to this problem: the
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

    with st.spinner("Loading model metrics… (cached after first load)"):
        _model, _feat_cols, _train_m, _test_m = load_forecast_model("rf")

    col_a, col_b = st.columns(2)
    col_a.metric("Train RMSE (£/MWh)", f"{_train_m['rmse']:.2f}", help=f"n = {_train_m['n_samples']:,} periods")
    col_b.metric("Test RMSE (£/MWh)",  f"{_test_m['rmse']:.2f}",  help=f"n = {_test_m['n_samples']:,} periods (held-out, after {DEFAULT_TEST_START})")
    col_a.metric("Train MAE (£/MWh)",  f"{_train_m['mae']:.2f}")
    col_b.metric("Test MAE (£/MWh)",   f"{_test_m['mae']:.2f}")
    col_a.metric(
        "Train Spearman ρ", f"{_train_m['spearman']:.3f}",
        help="Rank correlation — ordinal accuracy governs MPC dispatch quality.",
    )
    col_b.metric(
        "Test Spearman ρ",  f"{_test_m['spearman']:.3f}",
        help="Rank correlation — ordinal accuracy governs MPC dispatch quality.",
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
        "**Interpretation:** MAE < 15 £/MWh is adequate for MPC dispatch; "
        "MAE < 8 £/MWh is 'good'. "
        "Spearman ρ > 0.8 indicates strong ordinal ranking accuracy. "
        "Spike-RMSE reflects model quality on the high-price periods that drive arbitrage revenue."
    )

    importances = get_feature_importances(_model, _feat_cols).head(10)
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


# ---------------------------------------------------------------------------
# Tab: Sensitivity
# ---------------------------------------------------------------------------

with tab_sensitivity:
    st.subheader("Sensitivity: Revenue by Battery Size")
    st.caption(
        "Fixed battery parameters (efficiency 90%, availability 95%, cycling cost £3/MWh) "
        "with greedy dispatch. Duration fixed at 2h. Power range only."
    )

    auctions_s  = load_auctions()
    mkt_index_s = load_market_index()

    if auctions_s.empty:
        st.info("Auction data not available — run `scripts/prepare_data.py` first.")
    else:
        sens_df = sensitivity_table(
            auctions_s,
            mkt_index_s,
            FIXED_BATTERY,
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
