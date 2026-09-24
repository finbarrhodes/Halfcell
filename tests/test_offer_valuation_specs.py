"""
Run specs for scripts/compare_offer_valuation.py: the names are cache keys and report
rows, so a spec must keep meaning what it meant, and a new band must not collide
with an old one.
"""
import pytest

from scripts.compare_offer_valuation import parse_run, run_name


@pytest.mark.parametrize("spec, name", [
    ("ml:lp:0.5:bid:vint", "ml_lp_shrink0.5_bid_vint"),
    ("ml:lp:1:cp:0.2:bid:vint", "ml_lp_cp0.2per_bid_vint"),         # split conformal, as before
    ("ml:lp:1:cp:blk:0.2:bid:vint", "ml_lp_cp0.2blo_bid_vint"),
    ("ml:lp:1:qr:0.2:bid:vint", "ml_lp_qr0.2_bid_vint"),
    ("ml:lp:1:cqr:0.3:bid:vint", "ml_lp_cqr0.3_bid_vint"),
    ("ml:lp:0.5:spci:0.2:bid:vint", "ml_lp_shrink0.5_spci0.2_bid_vint"),
    ("ml:lp:1:spci_b:0.1:bid:vint", "ml_lp_spci_b0.1_bid_vint"),
    ("pf:lp:1:qr:0.2:vint", "pf_lp_vint"),                           # perfect foresight has no band
    ("pf:lp:1:rec:vint", "pf_lp_rec_vint"),                          # but recovery credit is dispatch
    ("ml:lp:0.5:recany:bid:vint", "ml_lp_shrink0.5_recany_bid_vint"),
])
def test_specs_name_the_run_they_describe(spec, name):
    assert run_name(*parse_run(spec)) == name


def test_a_band_method_carries_its_alpha():
    guard = parse_run("ml:lp:1:cqr:0.3:bid:vint")[7]
    assert guard == (0.3, "cqr")


@pytest.mark.parametrize("spec", [
    "ml:lp:1:qr:spci:0.2:bid:vint",      # two bands at once
    "ml:lp:1:cp:qr:0.2:bid:vint",        # split conformal and another
    "naive:lp:1:qr:0.2:bid:vint",        # quantile bands are built on the ML forecast
])
def test_ambiguous_band_specs_are_refused(spec):
    with pytest.raises(ValueError, match="one band at a time"):
        parse_run(spec)
