#!/usr/bin/env python3
"""
Check the two claims Halfcell makes about O'Connor et al. (2025)
===============================================================
The methodology page says two things about the benchmark this project's interval
work is built on (arXiv:2502.04935): that its EnbPI and SPCI "0.1-0.9" bands are
nominally 90% intervals rather than the 80% they are compared against, and that
on the random-forest forecasts the paper publishes, the forest's own quantiles
score better than either conformal method. A claim about someone else's work does
not belong on a public page without the working, so this script is the working.

**The nominal level.** Three facts, each checkable at source:

  1. The paper defines the interval as [q_alpha, q_(1-alpha)] with confidence
     1 - 2*alpha, and says that alpha = 0.1 gives 80% (section 3.2).
  2. Its own code calls the reference implementation as
     `compute_PIs_Ensemble_online(0.1, ...)` and `(0.3, ...)`
     (Conformal Prediction Forecasting Library/EnbPI_SPCI_DAM.py).
  3. That implementation (Xu & Xie's SPCI_class.py) builds the interval from the
     residual percentiles [beta_hat, 1 - alpha + beta_hat] - a probability width of
     1 - alpha. So alpha is the *total* miscoverage: 0.1 is a 90% interval, 0.3 a
     70% one.

This script re-downloads (2) and (3) and greps them, so the reading is not taken
on trust. The quotation in (1) is in the paper's own text.

**The scores.** Coverage, mean width, pinball loss and the interval (Winkler)
score, recomputed from the published day-ahead random-forest forecasts for
quantile regression, EnbPI and SPCI. The interval score is
`(u - l) + (2/alpha)[(l - y)+ + (y - u)+]` at the *labelled* level, which is the
comparison the paper's tables make.

One honest caveat the page repeats: the paper's own Table 3 ranks the conformal
methods ahead of quantile regression (33.70 against 32.14 and 31.65 for the
random forest). None of the usual conventions reproduces those values from the
published forecasts; the ranking appears only if the miscoverage penalty is not
scaled by 1/alpha, which charges a band little for missing the price. This script
prints the variants so a reader can see what is sensitive to what.

Files come from the repository the paper links. Nothing is cached in git: the
script downloads to a scratch directory and can be re-run.

Usage:
    python scripts/verify_interval_literature.py
    python scripts/verify_interval_literature.py --cache /tmp/oconnor
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

REPORTS = ROOT / "reports"
PAPER_REPO = "https://anonymous.4open.science/api/repo/PEPF_Conformal-C0AF/file"
REFERENCE_CODE = "https://raw.githubusercontent.com/hamrel-cxu/SPCI-code/main/SPCI_class.py"
FORECASTS = {"QR": "rf_Q_DAM_1-12.csv", "EnbPI": "rf_EnbPI_DAM_1-12.csv", "SPCI": "rf_SPCI_DAM_1-12.csv"}
PAPER_CALLER = "Conformal%20Prediction%20Forecasting%20Library/EnbPI_SPCI_DAM.py"
# Table 3, random forest row, day-ahead market
PAPER_TABLE_3 = {"QR": 33.70, "EnbPI": 32.14, "SPCI": 31.65}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Halfcell literature check)"}


def fetch(url: str, target: Path) -> Path:
    """Download once into the scratch directory; keep it for re-runs."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=180) as response, open(target, "wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    return target


def nominal_level(cache: Path) -> dict:
    """What alpha means where the paper passes it, and where it is consumed."""
    caller = fetch(f"{PAPER_REPO}/{PAPER_CALLER}", cache / "EnbPI_SPCI_DAM.py").read_text(errors="replace")
    reference = fetch(REFERENCE_CODE, cache / "SPCI_class.py").read_text(errors="replace")

    calls = sorted(set(re.findall(r"compute_PIs_Ensemble_online\(\s*([0-9.]+)", caller)))
    # the reference builds [percentile(resid, beta), percentile(resid, 1 - alpha + beta)]
    construction = re.findall(r"np\.percentile\(\s*\n?\s*past_resid,\s*math\.ceil\(100 \* ([^)]+)\)", reference)
    total_miscoverage = any("1 - alpha" in piece for piece in construction)
    return {
        "paper_calls_alpha": calls,
        "reference_interval_bounds": [piece.strip() for piece in construction],
        "alpha_is_total_miscoverage": bool(total_miscoverage),
        "implied_coverage": {alpha: f"{(1 - float(alpha)) * 100:.0f}%" for alpha in calls},
        "labelled_coverage": {alpha: f"{(1 - 2 * float(alpha)) * 100:.0f}%" for alpha in calls},
    }


def bounds(path: Path, low: int = 10, high: int = 90) -> tuple:
    """Actuals and the published interval bounds, flattened over 365 days x 24 hours."""
    frame = pd.read_csv(path)
    hours = [f"EURPrices+{hour}" for hour in range(24)]
    pick = lambda suffix: frame[[f"{hour}_Forecast_{suffix}" for hour in hours]].to_numpy().ravel()
    actual = frame[hours].to_numpy().ravel()
    lower, upper = pick(low), pick(high)
    keep = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    return actual[keep], lower[keep], upper[keep]


def interval_score(actual, lower, upper, alpha: float, scale_penalty: bool = True) -> float:
    """Width plus a penalty for each price outside, the penalty scaled by 2/alpha."""
    missed = np.maximum(lower - actual, 0) + np.maximum(actual - upper, 0)
    factor = (2 / alpha) if scale_penalty else 2.0
    return float(np.mean((upper - lower) + factor * missed))


def pinball(actual, forecast, quantile: float) -> float:
    delta = actual - forecast
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1) * delta)))


def score_forecasts(cache: Path) -> dict:
    """Coverage and scores for each method, at both quantile pairs the paper reports."""
    out = {}
    for name, filename in FORECASTS.items():
        path = fetch(f"{PAPER_REPO}/Results/DAM/{filename}", cache / filename)
        row = {}
        for pair, (low, high), labelled, implied in (("0.1-0.9", (10, 90), 0.2, 0.1),
                                                     ("0.3-0.7", (30, 70), 0.6, 0.3)):
            actual, lower, upper = bounds(path, low, high)
            row[pair] = {
                "n": int(actual.size),
                "coverage": round(float(np.mean((actual >= lower) & (actual <= upper))), 3),
                "mean_width": round(float(np.mean(upper - lower)), 1),
                "interval_score_labelled": round(interval_score(actual, lower, upper, labelled), 1),
                "interval_score_implied": round(interval_score(actual, lower, upper, implied), 1),
                "interval_score_unscaled_penalty": round(
                    interval_score(actual, lower, upper, labelled, scale_penalty=False), 1),
                "pinball_lower": round(pinball(actual, lower, low / 100), 2),
                "pinball_upper": round(pinball(actual, upper, high / 100), 2),
            }
        out[name] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/oconnor-2025"),
                        help="scratch directory for the downloaded files")
    args = parser.parse_args()

    print("Claim 1 — what the nominal level really is")
    level = nominal_level(args.cache)
    print(f"  the paper calls the reference with alpha = {', '.join(level['paper_calls_alpha'])}")
    print(f"  the reference builds bounds at percentiles {level['reference_interval_bounds']}")
    print(f"  so alpha is total miscoverage: {level['alpha_is_total_miscoverage']}")
    for alpha in level["paper_calls_alpha"]:
        print(f"  alpha {alpha}: labelled {level['labelled_coverage'][alpha]}, "
              f"actually {level['implied_coverage'][alpha]}")

    print("\nClaim 2 — scores from the published random-forest day-ahead forecasts")
    scores = score_forecasts(args.cache)
    print(f"  {'method':<7}{'coverage':>10}{'width':>8}{'score@80%':>11}{'score@90%':>11}{'pinball lo/hi':>16}")
    for name, row in scores.items():
        pair = row["0.1-0.9"]
        print(f"  {name:<7}{pair['coverage']:>10.3f}{pair['mean_width']:>8.1f}"
              f"{pair['interval_score_labelled']:>11.1f}{pair['interval_score_implied']:>11.1f}"
              f"{pair['pinball_lower']:>8.2f}{pair['pinball_upper']:>8.2f}")

    print("\n  The paper's Table 3 ranks these the other way (QR 33.70, EnbPI 32.14, SPCI 31.65).")
    print("  Not reproducible from the published forecasts under the usual conventions; the")
    print("  ranking appears only with the miscoverage penalty left unscaled:")
    for name, row in scores.items():
        print(f"    {name:<7}{row['0.1-0.9']['interval_score_unscaled_penalty']:>8.1f}")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "interval_literature_check.json").write_text(json.dumps(
        {"nominal_level": level, "scores": scores, "paper_table_3_rf_dam": PAPER_TABLE_3}, indent=2) + "\n")

    lines = [
        "# Checking two claims about O'Connor et al. (2025)",
        "",
        "Reproduced by `scripts/verify_interval_literature.py` from the paper's own linked",
        "repository and from the reference implementation it calls.",
        "",
        "## The conformal bands are nominally 90% and 70%, not 80% and 40%",
        "",
        f"- The paper calls the reference implementation with alpha = {', '.join(level['paper_calls_alpha'])}",
        "  (`EnbPI_SPCI_DAM.py`), having defined the interval as `[q_alpha, q_(1-alpha)]` with",
        "  confidence `1 - 2*alpha` — so alpha = 0.1 is labelled 80%.",
        f"- The reference implementation builds the bounds at residual percentiles",
        f"  `{level['reference_interval_bounds']}`, a probability width of `1 - alpha`.",
        "- So alpha is the total miscoverage: the bands are 90% and 70% intervals, compared",
        "  against quantile regression's genuine 80% and 40%.",
        "",
        "## On its published random-forest forecasts, the forest's quantiles score better",
        "",
        "Day-ahead market, 365 days x 24 hours. Interval score is width plus `2/alpha` per unit",
        "of price outside the band, at the labelled level; lower is better.",
        "",
        "| Method | Coverage (labelled 0.80) | Mean width | Interval score |",
        "|---|---|---|---|",
    ]
    for name, row in scores.items():
        pair = row["0.1-0.9"]
        lines.append(f"| {name} | {pair['coverage']:.3f} | {pair['mean_width']:.1f} | "
                     f"{pair['interval_score_labelled']:.1f} |")
    lines += [
        "",
        "The conformal bands are narrower than quantile regression's and cover less, despite",
        "being nominally wider — the mislabelling is not what flatters them.",
        "",
        "**A caveat carried on the methodology page.** The paper's Table 3 ranks the same three",
        f"the other way for this model and market: QR {PAPER_TABLE_3['QR']}, EnbPI "
        f"{PAPER_TABLE_3['EnbPI']}, SPCI {PAPER_TABLE_3['SPCI']}. None of the usual scoring",
        "conventions reproduces those values from the published forecasts. The ranking does",
        "appear if the miscoverage penalty is left unscaled by `1/alpha`, which charges a band",
        "little for missing the price:",
        "",
        "| Method | Interval score, penalty unscaled |",
        "|---|---|",
    ]
    for name, row in scores.items():
        lines.append(f"| {name} | {row['0.1-0.9']['interval_score_unscaled_penalty']:.1f} |")
    lines += ["", "That is a reading of their table, not a claim about their code."]
    (REPORTS / "interval_literature_check.md").write_text("\n".join(lines) + "\n")
    print("\nWrote reports/interval_literature_check.md")


if __name__ == "__main__":
    main()
