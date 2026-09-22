"""
src/analysis/significance.py
============================
Error bars for the two claims Halfcell makes about forecasting: that the model
forecasts better than reusing an old day's prices, and that it earns more.

Both need the same caution. A backtest produces one path, not a sample of
independent draws: today's error resembles yesterday's, today's revenue depends
on the state of energy yesterday left behind, and a five-year window holds far
fewer effective observations than it has rows. So neither claim is tested with a
statistic that assumes independence.

**Accuracy: Diebold-Mariano.** Take the loss each forecast made on the same day -
squared error, or the error in the day's spread - and test whether the difference
has a mean of zero ([Diebold & Mariano,
1995](https://doi.org/10.1080/07350015.1995.10524599)). The test statistic is the
mean difference over a long-run standard error, computed with a
Newey-West/Bartlett kernel so that serial correlation in the differences is
carried rather than assumed away. `harvey_correction` applies the small-sample
adjustment of Harvey, Leybourne & Newbold (1997); at these sample sizes it
changes little, and it is there so the number does not depend on remembering it.

**Revenue: a moving block bootstrap.** Revenue cannot be resampled day by day,
because the days are not exchangeable: a dispatch decision carries state into the
next day. Resampling contiguous blocks preserves dependence inside each block and
only breaks it at the joins ([Künsch,
1989](https://doi.org/10.1214/aos/1176347265)); with a block long enough to cover
the memory, the interval is honest about how little independent information a
five-year backtest really holds. The default block is four weeks, comfortably
longer than the state of energy's memory, which the soft requirement resets at
every EFA block.

Neither tool decides anything on its own. They are here because the differences
Halfcell now reports - the model earns about £3k/MW/yr more than naive, and a
rejected refinement was worth £0.7k - are small enough that the first question is
whether they are distinguishable from noise at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BLOCK_DAYS = 28
DEFAULT_RESAMPLES = 10_000


def newey_west_variance(x: np.ndarray, lags: int | None = None) -> float:
    """
    Long-run variance of the mean of `x`, Bartlett-kernel weighted.

    With `lags=0` this is the plain variance of the mean. The default follows the
    usual n^(1/3) rule, which is what the Diebold-Mariano literature uses when the
    horizon does not dictate a lag.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(n ** (1 / 3))))
    centred = x - x.mean()
    total = float(centred @ centred) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        total += 2.0 * weight * float(centred[lag:] @ centred[:-lag]) / n
    return max(total, 0.0) / n


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, *, lags: int | None = None,
                    harvey_correction: bool = True) -> dict:
    """
    Test whether forecast A loses less than forecast B, on paired losses.

    Returns {"mean_difference", "t_stat", "p_value", "n", "lags"}, where the mean
    difference is loss_b - loss_a: positive means A is the better forecast. The
    p-value is two-sided from Student's t with n-1 degrees of freedom, which is the
    Harvey-Leybourne-Newbold recommendation for finite samples.
    """
    from scipy import stats

    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired losses must be the same length, got {a.shape} and {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    d = b[keep] - a[keep]
    n = len(d)
    if n < 3:
        return {"mean_difference": float("nan"), "t_stat": float("nan"),
                "p_value": float("nan"), "n": n, "lags": 0}

    used_lags = max(1, int(np.floor(n ** (1 / 3)))) if lags is None else lags
    variance = newey_west_variance(d, used_lags)
    t_stat = float(d.mean() / np.sqrt(variance)) if variance > 0 else float("nan")
    if harvey_correction and used_lags > 0:
        h = used_lags + 1
        factor = (1 + (1 - h) / n) if n > h else 1.0
        t_stat *= float(np.sqrt(max(factor, 1e-12)))
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else float("nan")
    return {"mean_difference": float(d.mean()), "t_stat": t_stat, "p_value": p_value,
            "n": n, "lags": used_lags}


def moving_block_bootstrap(values: np.ndarray, *, block: int = DEFAULT_BLOCK_DAYS,
                           resamples: int = DEFAULT_RESAMPLES, seed: int = 0) -> np.ndarray:
    """
    Resampled means of a series, drawn as overlapping contiguous blocks.

    Each resample is built by drawing ceil(n / block) starting points uniformly and
    concatenating the blocks that follow them, wrapping at the end so every
    observation is equally likely to appear.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.array([])
    block = int(min(max(1, block), n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, n_blocks))
    offsets = np.arange(block)
    # (resamples, n_blocks, block) indices, wrapped, then trimmed back to n
    index = (starts[:, :, None] + offsets[None, None, :]) % n
    return x[index.reshape(resamples, -1)[:, :n]].mean(axis=1)


def bootstrap_interval(values: np.ndarray, *, block: int = DEFAULT_BLOCK_DAYS,
                       resamples: int = DEFAULT_RESAMPLES, confidence: float = 0.95,
                       seed: int = 0, scale: float = 1.0) -> dict:
    """
    A confidence interval for the mean of a dependent series, and the share of
    resamples on the other side of zero.

    `scale` multiplies the mean and its interval, for turning a daily figure into
    an annual one (365.25) or a per-MW one, so the caller never has to rescale a
    quantile by hand.
    """
    draws = moving_block_bootstrap(values, block=block, resamples=resamples, seed=seed)
    if not draws.size:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"),
                "share_below_zero": float("nan"), "block": block, "resamples": resamples}
    tail = (1 - confidence) / 2
    observed = float(np.nanmean(np.asarray(values, dtype=float)))
    return {
        "mean": observed * scale,
        "low": float(np.quantile(draws, tail)) * scale,
        "high": float(np.quantile(draws, 1 - tail)) * scale,
        "share_below_zero": float((draws <= 0).mean()),
        "block": block,
        "resamples": resamples,
    }


def paired_daily(left: pd.DataFrame, right: pd.DataFrame, column: str = "net_revenue") -> pd.DataFrame:
    """
    Two daily revenue tables aligned on the days both cover, with their difference.

    Pairing matters: the two strategies met the same prices and the same frequency,
    so the difference removes everything they had in common and leaves the part the
    forecast is responsible for.
    """
    a = left.set_index(pd.to_datetime(left["date"]))[column]
    b = right.set_index(pd.to_datetime(right["date"]))[column]
    days = a.index.intersection(b.index)
    return pd.DataFrame({"left": a.loc[days], "right": b.loc[days],
                         "difference": a.loc[days] - b.loc[days]}).sort_index()
