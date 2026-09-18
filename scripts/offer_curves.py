#!/usr/bin/env python3
"""
Offer curves for one day: what each block's response capacity costs in trading
==============================================================================
Prices DC High and DC Low in every EFA block of a sample day, one 5 MW step at a
time, as the trading the day-ahead plan (src/optimisation/day_ahead.py) gives up
by holding it. Each step's cost in £/MW/h is where a price-taker would offer
that slice: the curve a unit bids is this stepped supply curve.

Set beside the per-block formula (revenue_stack._shadow_arb_value_per_mw), which
charges one flat price per block for either direction, it shows what the plan
sees and the formula cannot: High and Low diverge and swap across the day, and
the first megawatts are often free while the last are dear.

Each product is priced on its own, with nothing else held, from the stated state
of energy at the start of EFA 1 and on actual prices (perfect foresight), so the
curves show the mechanism rather than any forecast's errors.

Usage:
    python scripts/offer_curves.py                     # a representative 2025 day
    python scripts/offer_curves.py --date 2025-11-12 --soc 0.5
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis.revenue_stack import (
    EFA_HOURS,
    REFERENCE_BATTERY,
    _apx_by_date,
    _efa_prices,
    _shadow_arb_value_per_mw,
)
from src.optimisation.day_ahead import plan_day

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STEP_MW = 5.0
PRODUCTS_SHOWN = ("DCH", "DCL")


def service_day_prices(apx_by_date: dict, day: pd.Timestamp) -> np.ndarray:
    """The 48 prices of a service day in delivery order, 23:00 the evening before first."""
    return np.concatenate([_efa_prices(apx_by_date, day, efa).to_numpy() for efa in range(1, 7)])


def representative_day(apx_by_date: dict, year: int) -> pd.Timestamp:
    """The day of `year` whose price shape is closest to that year's median shape."""
    candidates = {d: service_day_prices(apx_by_date, d) for d in apx_by_date if d.year == year}
    # Clock-change days, and days with gaps, do not have 48 periods to compare
    days = [d for d, shape in candidates.items() if len(shape) == 48]
    shapes = np.array([candidates[d] for d in days])
    median = np.nanmedian(shapes, axis=0)
    distance = np.nansum((shapes - median) ** 2, axis=1)
    return days[int(np.argmin(distance))]


def curves(prices: np.ndarray, soc_frac: float) -> dict:
    """{product: [[£/MW/h per step] per block]} for the plan, holding one product at a time."""
    b = REFERENCE_BATTERY
    empty = [{"prices": {}, "families": ()}] * 6

    def trading_value(fixed):
        out = plan_day(empty, b.power_mw, b.energy_mwh, b.efficiency_rt, b.cycling_cost_per_mwh,
                       soc_frac * b.energy_mwh, prices, apply_reserve=True, fixed=fixed)
        if not out["solved"]:
            raise RuntimeError(f"cannot hold {fixed} from {soc_frac:.0%}: pick another --soc")
        return out["trading_value_gbp"]

    base = trading_value([{}] * 6)
    steps = np.arange(STEP_MW, b.power_mw + 1e-9, STEP_MW)
    out = {}
    for product in PRODUCTS_SHOWN:
        per_block = []
        for block in range(6):
            previous, costs = base, []
            for mw in steps:
                fixed = [{product: mw} if k == block else {} for k in range(6)]
                value = trading_value(fixed)
                costs.append(round((previous - value) / (STEP_MW * EFA_HOURS), 2))
                previous = value
            per_block.append(costs)
        out[product] = per_block
    return out


def formula(apx_by_date: dict, day: pd.Timestamp) -> list:
    """The formula's flat price per block, in £/MW/h, the same for High and Low."""
    return [round(_shadow_arb_value_per_mw(_efa_prices(apx_by_date, day, efa), REFERENCE_BATTERY) / EFA_HOURS, 2)
            for efa in range(1, 7)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="service day (default: a representative 2025 day)")
    parser.add_argument("--soc", type=float, default=0.5, help="state of energy at the start, as a fraction")
    args = parser.parse_args()

    apx_by_date = _apx_by_date(pd.read_parquet(PROCESSED / "market_index.parquet"))
    day = pd.Timestamp(args.date).normalize() if args.date else representative_day(apx_by_date, 2025)
    prices = service_day_prices(apx_by_date, day)
    result = {
        "date": day.date().isoformat(),
        "soc_frac": args.soc,
        "step_mw": STEP_MW,
        "block_mean_price": [round(float(np.nanmean(prices[8 * k:8 * k + 8])), 1) for k in range(6)],
        "formula_gbp_per_mw_h": formula(apx_by_date, day),
        "plan_gbp_per_mw_h": curves(prices, args.soc),
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "offer_curves.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# Offer curves, {result['date']}",
        "",
        f"£/MW/h each {STEP_MW:g} MW slice would be offered at, from the trading it gives up. "
        f"Perfect foresight; state of energy {args.soc:.0%} at 23:00 the evening before; "
        "each product priced alone. The formula charges one flat price per block for either direction.",
        "",
        "| Block | Mean £/MWh | Formula | DC High, first → last 5 MW | DC Low, first → last 5 MW |",
        "|---|---|---|---|---|",
    ]
    for k in range(6):
        high = result["plan_gbp_per_mw_h"]["DCH"][k]
        low = result["plan_gbp_per_mw_h"]["DCL"][k]
        lines.append(f"| {k + 1} | {result['block_mean_price'][k]} | {result['formula_gbp_per_mw_h'][k]} | "
                     f"{' · '.join(f'{c:g}' for c in high)} | {' · '.join(f'{c:g}' for c in low)} |")
    (REPORTS / "offer_curves.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
