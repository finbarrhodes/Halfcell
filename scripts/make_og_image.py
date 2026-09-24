"""
scripts/make_og_image.py
========================
Render the Open Graph card used when a Halfcell link is shared.

Without an og:image, a pasted link unfurls as text with an empty preview panel —
and because the page declares `twitter:card: summary_large_image`, some clients
render that worse than a plain card. This draws a 1200x630 card from the live
cache, so the figure on it cannot drift away from the site it advertises.

Run from the project root:
    python scripts/make_og_image.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "site" / "static" / "og-image.png"

# Matches src/style.css
PAPER = "#FFF1E5"
INK, INK_SOFT = "#33302E", "#66605C"
TEAL, GREEN, ORANGE = "#0D7680", "#4E8A3C", "#C9400A"
RULE = "#E0D3C4"

SERIES = [("pf_mpc", "Perfect Foresight", GREEN),
          ("ml_mpc", "ML Model", TEAL),
          ("naive_mpc", "Naive (D-1)", ORANGE)]
def _cumulative(key):
    df = pd.read_parquet(CACHE / f"{key}.parquet").sort_values("month_dt")
    return pd.to_datetime(df["month_dt"]), df["net_revenue"].cumsum() / 1e6


def main() -> None:
    # 1200x630 is the size every major unfurler crops to
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(PAPER)

    fig.text(0.055, 0.845, "Halfcell", fontsize=52, color=INK,
             family="serif", weight="regular")
    fig.text(0.055, 0.775, "GB battery storage — market analysis & dispatch modelling",
             fontsize=19, color=INK_SOFT)
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.735, 0.735],
                              color=RULE, linewidth=1.2, transform=fig.transFigure))

    # Full-width chart — this card is deliberately just the headline plot,
    # not a stats dashboard: fewer numbers to go stale between deploys.
    ax = fig.add_axes([0.055, 0.20, 0.89, 0.48])
    ax.set_facecolor(PAPER)
    for key, label, colour in SERIES:
        x, y = _cumulative(key)
        ax.plot(x, y, color=colour, linewidth=2.6, label=label)

    ax.legend(frameon=False, fontsize=12, loc="upper left", labelcolor=INK_SOFT)
    # No y-axis label: the £M ticks carry it, and at preview-card size the
    # rotated text competes with the title for attention.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v:,.0f}M"))
    ax.tick_params(colors=INK_SOFT, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.text(0.055, 0.085, "halfcell.uk", fontsize=15, color=ORANGE, weight="bold")
    fig.text(0.945, 0.085, "NESO · Elexon · DESNZ open data",
             fontsize=12, color=INK_SOFT, ha="right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=PAPER, format="png")
    plt.close(fig)
    print(f"Written → {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
