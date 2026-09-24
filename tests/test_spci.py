"""
SPCI guard bands: the residual's quantiles given the residuals just before it.

The synthetic forecasts here switch every few weeks between unbiased spells and
spells where they run £40 short, the persistence SPCI exists for: a split conformal
band has one position for both, while SPCI should move once the recent errors show
the forecast has started missing. A spell that is merely noisier, with no bias, is
the case SPCI as published does not see - its forests split to predict the
residual's mean, and a noisier spell has the same mean - and a test pins that so it
stays visible. Only two settlement periods are populated, so each forest is quick
to fit.
"""
import numpy as np
import pandas as pd
import pytest

from src.analysis.intervals import band_frame, score_bands, walk_forward_bands
from src.analysis.spci import lagged_windows, residual_matrix, walk_forward_spci_bands

PERIODS = [1, 2]
REGIME_DAYS = 20


def _tables(n_days=420, start="2021-01-01", seed=0, spell="bias"):
    """
    A flat £80 forecast whose errors change every REGIME_DAYS: in alternate spells they
    are N(+40, 10) - the forecast running short - or, with spell="noise", N(0, 40)
    against N(0, 5).
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows_f, rows_a = [], []
    for k, day in enumerate(days):
        odd = (k // REGIME_DAYS) % 2 == 1
        mean, sd = ((40.0 if odd else 0.0), 10.0) if spell == "bias" else (0.0, 40.0 if odd else 5.0)
        for sp in PERIODS:
            rows_f.append({"settlementDate": day, "settlementPeriod": sp, "prediction": 80.0})
            rows_a.append({"settlementDate": day, "settlementPeriod": sp, "dataProvider": "APXMIDP",
                           "price": 80.0 + rng.normal(mean, sd)})
    return pd.DataFrame(rows_f), pd.DataFrame(rows_a)


def _split(days, start="2021-01-01", settle=10):
    """(odd-spell days, even-spell days), each judged only once its spell is `settle` days old."""
    age = lambda d: (pd.Timestamp(d) - pd.Timestamp(start)).days
    settled = [d for d in days if age(d) % REGIME_DAYS >= settle]
    return ([d for d in settled if (age(d) // REGIME_DAYS) % 2 == 1],
            [d for d in settled if (age(d) // REGIME_DAYS) % 2 == 0])


SETTINGS = dict(alpha=0.2, window=96, refit_days=40, min_days=100)


@pytest.fixture(scope="module")
def spci():
    f, a = _tables()
    return f, a, walk_forward_spci_bands(f, a, **SETTINGS)


def test_the_residual_matrix_keeps_one_row_per_calendar_day():
    f, a = _tables(n_days=10)
    f = f[pd.to_datetime(f["settlementDate"]) != pd.Timestamp("2021-01-05")]
    matrix = residual_matrix(f, a)
    assert len(matrix) == 10 and matrix.shape[1] == 48
    assert matrix.loc["2021-01-05"].isna().all()


def test_windows_end_where_they_are_told_and_pad_with_zero():
    flat = np.arange(1.0, 11.0)
    windows = lagged_windows(flat, np.array([4, 1]), 3)
    assert windows[0].tolist() == [3.0, 4.0, 5.0]
    assert windows[1].tolist() == [0.0, 1.0, 2.0]


def test_the_band_moves_when_the_recent_errors_show_a_bias(spci):
    """The point of SPCI: once the forecast starts running short, the band follows."""
    _, _, (low, high, _) = spci
    biased, plain = _split(low)
    buy_side = lambda days: np.mean([high[d][1] for d in days])
    sell_side = lambda days: np.mean([low[d][1] for d in days])
    assert buy_side(biased) > buy_side(plain) + 20
    assert sell_side(biased) > sell_side(plain) + 5       # less pessimistic about a sale


def test_a_noisier_spell_with_no_bias_goes_unnoticed():
    """
    A known limit, pinned so it stays visible: the forests split to predict the
    residual's mean, so a spell eight times noisier but unbiased gets almost the
    same band. Split conformal's trailing window is the answer to that.
    """
    f, a = _tables(spell="noise")
    low, high, _ = walk_forward_spci_bands(f, a, **SETTINGS)
    noisy, calm = _split(low)
    width = lambda days: np.mean([(high[d] - low[d])[1] for d in days])
    assert width(noisy) < 1.5 * width(calm)


def test_it_scores_better_than_one_width_for_every_regime(spci):
    f, a, (low, high, _) = spci
    split_low, split_high, _ = walk_forward_bands(f, a, alpha=0.2, group="period", min_days=100)
    days = sorted(set(low) & set(split_low))
    pick = lambda table: {d: table[d] for d in days}
    ours = score_bands(band_frame(f, a, pick(low), pick(high)), 0.2)
    theirs = score_bands(band_frame(f, a, pick(split_low), pick(split_high)), 0.2)
    assert ours["winkler"].iloc[0] < theirs["winkler"].iloc[0]


def test_days_without_enough_history_are_left_unbanded(spci):
    _, _, (low, _, folds) = spci
    assert min(low) > pd.Timestamp("2021-01-01") + pd.Timedelta(days=100)
    assert folds[0]["banded"] is False


def test_a_band_sees_nothing_newer_than_two_days_before():
    """Offers close at 14:00 the day before: that day's prices, and the day's own, are unknown."""
    f, a = _tables()
    day = pd.Timestamp("2021-10-15")
    tampered = a.copy()
    dates = pd.to_datetime(tampered["settlementDate"])
    tampered.loc[dates >= day - pd.Timedelta(days=1), "price"] += 5000.0
    base = walk_forward_spci_bands(f, a, **SETTINGS)
    after = walk_forward_spci_bands(f, tampered, **SETTINGS)
    assert base[0][day].equals(after[0][day]) and base[1][day].equals(after[1][day])


def test_the_beta_search_never_widens_the_interval():
    """β = α is one of the candidates, so the narrowest candidate is at most as wide."""
    f, a = _tables(n_days=300)
    equal = walk_forward_spci_bands(f, a, **SETTINGS)
    searched = walk_forward_spci_bands(f, a, **SETTINGS, optimise_beta=True)
    for day in list(equal[0])[:20]:
        width = lambda bands: float((bands[1][day] - bands[0][day])[1])
        assert width(searched) <= width(equal) + 1e-9


def test_alpha_outside_the_open_interval_is_refused():
    f, a = _tables(n_days=30)
    with pytest.raises(ValueError, match="alpha"):
        walk_forward_spci_bands(f, a, alpha=0.6)
