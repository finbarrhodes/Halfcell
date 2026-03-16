"""
Methodology & Data Sources — Halfcell
=======================================
Static reference page. Launched as a page via app.py (st.navigation).
Can still be run standalone for local development:
    streamlit run src/visualization/methodology.py
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Standalone guard — set_page_config only when run directly, not via app.py
# ---------------------------------------------------------------------------
try:
    st.set_page_config(
        page_title="Methodology & Data — Halfcell",
        page_icon="⚡",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # already set by app.py

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Methodology & Data Sources")
st.markdown(
    "This page documents the modelling approach, assumptions, and data sources "
    "used by the Revenue Backtester."
)

st.divider()

# ---------------------------------------------------------------------------
# Two-stage participation model
# ---------------------------------------------------------------------------

st.header("Two-stage participation model")
st.markdown(
    """
This backtester uses a two-stage model to separate FR availability revenue from energy
arbitrage revenue without double-counting the same physical capacity:

- **Stage 1 — Day-ahead capacity allocation:** For each day D, the model decides how
  many MW to commit to FR services vs hold back for arbitrage. Two signals are compared
  using information available at end of day D-1:
    - **FR value per MW**: the confirmed clearing price for day D from the EAC day-ahead
      auction (which clears on D-1), summed across all selected services and EFA blocks.
      No forecasting is needed — this price is already known.
    - **Shadow arb value per MW**: a per-unit estimate of the net arbitrage profit for
      day D, derived from the same price forecast used for dispatch:
      `(avg_discharge − avg_charge / η − cycling_cost) × duration_h`.

  Capacity is then allocated proportionally:
  `fr_fraction = fr_value / (fr_value + arb_value)`,
  so more MW flows toward whichever stream looks more attractive on that day —
  without all-or-nothing switching.

- **Stage 2 — Intraday dispatch:** Within the allocated arbitrage MW, the selected
  dispatch strategy (Perfect Foresight, Naive, or ML) schedules charge/discharge against
  forecast prices and realises revenue against actual day-D prices.

The dispatch strategy is applied consistently to both stages: the same price signal that
drives dispatch also drives the shadow arb estimate in the allocation step.

In Stage 2, the MPC dispatch engine (described below) tracks SoC continuously at
30-minute resolution, enforcing the FR headroom band `[10%, 90%]` as a hard constraint
at every step of the planning horizon. The remaining simplification is that Stage 1
capacity allocation is a daily heuristic rather than being jointly co-optimised with the
intraday LP — see Known Limitations below.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Dispatch strategies
# ---------------------------------------------------------------------------

st.header("Dispatch strategies")
st.markdown(
    """
Intraday dispatch is driven by a **rolling Model Predictive Control (MPC) linear
programme (LP)**, re-solved at every 30-minute settlement period. At each period *t* the
LP plans over a 48-hour (96-period) horizon, returns only the first period's
charge/discharge decision, then re-solves next period — a classic receding-horizon
approach that reflects real operational constraints (the decision must be made before
future prices are known).

**LP formulation.** Decision variables are charge power *p_chg[t]*, discharge power
*p_dis[t]*, and state of charge *SoC[t+1]* for each of the H horizon periods. The
objective maximises net arbitrage revenue minus cycling degradation cost:

`maximise  Σ price[t] × (p_dis[t] − p_chg[t]) × 0.5 h  −  cycling_cost × Σ p_dis[t] × 0.5 h`

subject to:
- SoC state equation with round-trip efficiency applied on the charge side
- FR feasibility band `[10%, 90%]` of capacity enforced as a **hard constraint** at all
  H+1 SoC points, forcing the battery to pre-condition its SoC for upcoming FR delivery
  obligations
- Power bounded to the residual MW available for arbitrage after FR commitment in each
  period

Mutual exclusion of simultaneous charge and discharge is handled via LP relaxation:
since the objective penalises cycling, simultaneous charge+discharge is never optimal
at a positive spread, so no binary variables are required. The LP is solved with the
**CLARABEL** interior-point solver, bundled with cvxpy ≥ 1.4.0
([Diamond & Boyd, 2016](https://www.jmlr.org/papers/v17/15-408.html)).

**Three price signals are benchmarked** — all three run the identical MPC dispatch
engine; only the price forecast fed to the LP differs:

| Strategy | Price signal fed to LP | What it represents |
|---|---|---|
| **Perfect Foresight** | Actual day-D wholesale prices | Theoretical revenue ceiling — achievable only with advance knowledge of the future |
| **Naive (D-1 prices)** | Yesterday's 48 half-hourly prices | Zero-skill floor — the simplest possible baseline; any real model must beat this |
| **ML Model** | Random Forest forecast for day D | Realistic best case using features available at end of day D-1 |

Dispatch decisions are executed unconditionally at actual prices. Per-period revenue can
be negative when forecast error causes an unfavourable trade — this is the realistic
operational outcome and is intentional.

The **capture rate** (realised revenue ÷ perfect-foresight revenue) summarises how much
of the theoretical ceiling each strategy achieves. For LP-based joint co-optimisation
of energy arbitrage and frequency response in the GB market see
[Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365).
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Ancillary service availability revenue
# ---------------------------------------------------------------------------

st.header("Ancillary service availability revenue")
st.markdown(
    """
- Revenue = `clearing_price (£/MW/h) × MW committed to FR × 4 hours per EFA block`
- Services of different response speeds (DC, DR, DM) can be stacked on the same physical MW
  in the GB market — each earns a separate availability payment.
- High (discharge) and Low (charge) services are modelled as independent and simultaneous,
  assuming the battery maintains sufficient SoC headroom to respond in both directions.
- Clearing prices sourced from NESO Data Portal (legacy DC/DR/DM auctions Sep 2021–Nov 2023,
  EAC service Nov 2023–present).
- Ancillary revenue is identical across all three dispatch strategies — it does not depend
  on price forecasting.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Wholesale energy arbitrage revenue
# ---------------------------------------------------------------------------

st.header("Wholesale energy arbitrage revenue")
st.markdown(
    """
- Arbitrage revenue is computed period-by-period as the MPC LP dispatches:
  `revenue[t] = actual_price[t] × (e_dis[t] − e_chg[t])`, summed across all 48 settlement
  periods in the day.
- Power in each period is bounded to the residual MW available for arbitrage (total power
  minus the MW committed to FR for that EFA block). The LP respects round-trip efficiency
  by applying it to the charge side of the SoC state equation.
- The cycling wear cost `cycling_cost_per_mwh × e_dis[t]` is deducted each period and
  enters the LP objective, so the optimiser naturally avoids unprofitable cycles.
- **Price reference: APXMIDP market index** (APX Power UK) from Elexon Insights
  (Jul 2023–present). This is the actual GB spot settlement reference price, giving a
  materially more realistic daily spread than the imbalance settlement price (SSP),
  which can reach extreme negative values during high-renewable periods and would
  otherwise inflate arbitrage revenue.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Availability factor
# ---------------------------------------------------------------------------

st.header("Availability factor")
st.markdown(
    """
- Applied as a uniform multiplier to all revenue streams and cycling costs.
- Models periods where the asset is unavailable due to planned maintenance, unplanned
  faults, grid curtailment, or service delivery failures.
- The default of 95% reflects the minimum availability threshold mandated in NESO's Dynamic
  Containment and Enduring Auction Capability service specifications. This is also
  consistent with observed GB BESS fleet performance: Modo Energy's *GB Battery Storage
  Report* (2024) reports median fleet availability of 95–97% across contracted windows.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Cycling wear cost and battery degradation
# ---------------------------------------------------------------------------

st.header("Cycling wear cost and battery degradation")
st.markdown(
    """
- Applied to imbalance arbitrage trades only: `configured cycling wear cost (£/MWh) × MWh discharged per trade`.
- Ancillary service cycling (energy delivered during frequency events) is not separately
  modelled — it is minor relative to availability payments and is typically compensated
  via the service contract.
- *Why cycling matters beyond cost:* lithium-ion cells degrade through two primary
  mechanisms that accelerate with use — SEI (solid electrolyte interphase) layer growth,
  which consumes cyclable lithium irreversibly, and lithium plating at the anode, which
  increases with deeper discharge and higher charge rates. Each MWh cycled consumes a
  small fraction of the cell's finite cycle life. The cycling wear cost parameter is a
  financial proxy for this physical degradation: more aggressive dispatch earns more
  revenue in the short run but consumes cycle life faster, reducing the asset's useful
  life and residual value. This is why the revenue/cycling tradeoff chart is the core
  output of the strategy comparison, not revenue alone. For a rigorous treatment of
  cycle-based degradation cost formulation, see
  [Xu et al. (2018)](https://arxiv.org/abs/1703.07968) and
  [Lee & Kim (2022)](https://doi.org/10.1016/j.ijepes.2021.107795).
"""
)

st.divider()

# ---------------------------------------------------------------------------
# ML price forecast model
# ---------------------------------------------------------------------------

st.header("ML price forecast model")
st.markdown(
    """
The ML strategy uses a **Random Forest regressor** to predict the 48 half-hourly APXMIDP
prices for day D using features available at the end of day D-1.

*Why Random Forest?* The feature set is tabular (lagged prices, generation mix ratios,
temporal encodings) rather than raw sequences; they require no feature scaling; they are
robust on datasets of this size; and they provide interpretable feature importances. This
choice is consistent with the electricity price forecasting literature, which finds that
tree-based methods perform competitively against deep learning approaches on short-horizon
day-ahead forecasting tasks, particularly when training data is limited
([Lago et al., 2021](https://doi.org/10.1016/j.apenergy.2021.116983);
[Weron, 2014](https://doi.org/10.1016/j.ijforecast.2014.08.008)). An LSTM was considered
but is likely overkill given the training data size and would be harder to explain. A
naive D-1 lag model sets the zero-skill baseline.

**Features used (all available at end of day D-1):**
- Same-period lagged prices: price at the same settlement period 1, 2, and 7 days prior
- Previous-day price statistics: mean, standard deviation, max, and min across all 48 periods
- Generation mix (daily, from D-1): total generation, renewable fraction, fossil fraction,
  and per-fuel breakdown (gas, wind, nuclear, hydro, etc.)
- Cyclical temporal encodings: settlement period, day-of-week, and day-of-year encoded
  as sin/cos pairs to preserve circularity (e.g. period 48 and period 1 are adjacent)
- Weekend flag

**Train/test split:** strict temporal split — training data ends before 2025-03-01
to prevent any look-ahead bias. The model never sees future prices during training.

**Known limitations:** tree-based models cannot extrapolate beyond the price ranges seen
during training; electricity price forecasting is inherently noisy; and the model improves
dispatch quality on average but does not eliminate forecast error on individual days.

Model performance metrics (RMSE, MAE) and feature importances for the currently selected
model are shown in the **Strategy Comparison** tab of the Revenue Backtester.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Known limitations
# ---------------------------------------------------------------------------

st.header("Known limitations")
st.markdown(
    """
**Not yet modelled:**
- Intraday / day-ahead market trading (APXMIDP used as a proxy; DA auction data not yet integrated)
- Balancing Mechanism (BM) direct trading
- Real-time dispatch constraints or grid connection limits

**Known approximations in the current MPC LP:**
- *Rolling horizon is not globally optimal.* A single LP solved over the full backtest
  period would yield higher revenue in theory, but the rolling approach is used because
  it reflects real operational constraints — the dispatch decision must be made before
  future prices are known.
- *LP relaxation of charge/discharge mutual exclusion.* The LP does not use binary
  variables to prohibit simultaneous charging and discharging. This is not a binding
  limitation in practice: since the objective penalises cycling, simultaneous
  charge+discharge is never optimal at a positive spread.
- *Stage 1 capacity allocation is a daily heuristic.* The FR/arbitrage MW split is
  determined once per day by comparing confirmed FR clearing prices against a
  shadow arbitrage value. A fully joint formulation would co-optimise the allocation and
  intraday dispatch simultaneously in a single LP/MIP — see
  [Swierczynski et al. (2021)](https://doi.org/10.3390/en14248365) and
  [Bai et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149).

**Battery degradation (not yet modelled):**
In practice, a BESS asset degrades over its operational life through two primary
mechanisms: calendar ageing (capacity fade even at rest) and cycle ageing (accelerated
by depth of discharge, C-rate, and temperature). A more complete model would:
- Track state-of-health (SoH) over the backtest horizon, reducing usable capacity as
  the asset ages.
- Apply a degradation-aware dispatch policy that trades off short-term revenue against
  long-term cycle-life consumption — more aggressive dispatch earns more today but
  shortens asset life and residual value.
- Incorporate chemistry-specific degradation curves (NMC, LFP, etc.), which have
  materially different cycle-life profiles.

The cycling wear cost parameter is a simplified financial proxy for this effect, but it
does not capture the compounding, path-dependent nature of real battery degradation.
This is a natural next step for a more mature model.
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

st.header("Data Sources")
st.markdown(
    """
| Dataset | Source | Coverage |
|---|---|---|
| Frequency response auction results (DC/DR/DM) | [NESO Data Portal](https://www.neso.energy/data-portal) | Sep 2021 – present |
| APXMIDP market index price | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
| System buy/sell prices (SBP/SSP) | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
| Generation by fuel type (daily) | [Elexon Insights Solution API](https://developer.data.elexon.co.uk/) | Jul 2023 – present |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# Literature & References
# ---------------------------------------------------------------------------

st.header("Literature & References")
st.markdown(
    """
**Electricity price forecasting**

- Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). Forecasting day-ahead
  electricity prices: A review of state-of-the-art algorithms, best practices and an
  open-access benchmark. *Applied Energy*, 293, 116983.
  [doi:10.1016/j.apenergy.2021.116983](https://doi.org/10.1016/j.apenergy.2021.116983)

- Weron, R. (2014). Electricity price forecasting: A review of the fundamental and
  econometric approaches. *International Journal of Forecasting*, 30(4), 1030–1081.
  [doi:10.1016/j.ijforecast.2014.08.008](https://doi.org/10.1016/j.ijforecast.2014.08.008)

**MPC dispatch engine & LP solver**

- Diamond, S., & Boyd, S. (2016). CVXPY: A Python-Embedded Modeling Language for Convex
  Optimization. *Journal of Machine Learning Research*, 17(83), 1–5.
  [jmlr.org/papers/v17/15-408](https://www.jmlr.org/papers/v17/15-408.html)

- Goulart, P., & Chen, Y. (2024). Clarabel: An interior-point solver for conic programs
  with quadratic objectives. *IEEE Transactions on Automatic Control*.
  [doi:10.1109/TAC.2024.3457633](https://doi.org/10.1109/TAC.2024.3457633)

**BESS dispatch optimisation & co-optimisation**

- Swierczynski, M., Teodorescu, R., Rasmussen, C. N., Rodriguez, P., & Vikelgaard, H.
  (2021). Co-Optimizing Battery Storage for Energy Arbitrage and Frequency Regulation in
  the GB Market. *Energies*, 14(24), 8365.
  [doi:10.3390/en14248365](https://doi.org/10.3390/en14248365)

- Bai, X., et al. (2024). Smart optimization in battery energy storage systems: An overview.
  *Energy Storage Materials*.
  [doi:10.1016/j.esm.2024.100442](https://www.sciencedirect.com/science/article/abs/pii/S0306261924015149)

- Lee, J.-O., & Kim, Y.-S. (2022). Novel battery degradation cost formulation for optimal
  scheduling of battery energy storage systems. *International Journal of Electrical Power
  & Energy Systems*, 137, 107795.
  [doi:10.1016/j.ijepes.2021.107795](https://doi.org/10.1016/j.ijepes.2021.107795)

**Battery degradation modelling**

- Xu, B., Oudalov, A., Ulbig, A., Andersson, G., & Kirschen, D. S. (2018). Modeling of
  lithium-ion battery degradation for cell life assessment. *IEEE Transactions on Smart
  Grid*, 9(2), 1131–1140. Preprint: [arXiv:1703.07968](https://arxiv.org/abs/1703.07968)

- Reniers, J. M., Mulder, G., & Howey, D. A. (2021). Economic MPC of Li-ion battery
  cyclic aging via online rainflow analysis. *Journal of Energy Storage*.
  [doi:10.1002/est2.228](https://doi.org/10.1002/est2.228)

**GB BESS market context**

- Modo Energy. (2024). *GB Battery Storage Report*. [modoenergy.com](https://modoenergy.com/research/future-of-battery-energy-storage-buildout-in-great-britain)

- Timera Energy. (2023). Battery investors confront revenue shift in 2023.
  [timera-energy.com](https://timera-energy.com/blog/battery-investors-confront-revenue-shift-in-2023/)
"""
)
