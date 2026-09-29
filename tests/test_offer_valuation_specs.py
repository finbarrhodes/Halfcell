"""
Run specs for scripts/compare_offer_valuation.py: the names are cache keys and report
rows, so a spec must keep meaning what it meant, and a spec for an experiment the
engine no longer runs must be refused rather than quietly run as something else.
"""
import pytest

from scripts.compare_offer_valuation import SHIPPED, parse_run, run_name


@pytest.mark.parametrize("spec, name", [
    ("ml:lp:0.5:bid:vint", "ml_lp_shrink0.5_bid_vint"),
    ("ml:lp:0.5:recany:bid:vint", "ml_lp_shrink0.5_recany_bid_vint"),
    ("ml:lp:0.5:margin:bid:vint", "ml_lp_shrink0.5_margin_bid_vint"),
    ("pf:lp:1:recany:vint", "pf_lp_recany_vint"),
    ("pf:lp:1:margin:vint", "pf_lp_margin_vint"),
    ("pf:lp:0.5:margin:vint", "pf_lp_shrink0.5_margin_vint"),   # the shrink the margin was chosen on
    ("naive:formula", "naive_formula"),
])
def test_specs_name_the_run_they_describe(spec, name):
    assert run_name(*parse_run(spec)) == name


def test_the_shipped_runs_are_specs_the_script_can_run():
    for spec in ("pf:lp:1:recany:vint", "naive:lp:0.5:recany:bid:vint", "ml:lp:0.5:recany:bid:vint"):
        assert run_name(*parse_run(spec)) in SHIPPED.values()


@pytest.mark.parametrize("spec", [
    "ml:lp:1:cqr:0.3:bid:vint",     # guard bands
    "ml:lp:0.5:rec:bid:vint",       # the loose reading of recovery credit
    "ml:lp:1:bid:vint:dyn",         # a weight fitted per day
    "ml:lp:0.5:sm5:bid:vint",       # smoothed dispatch
])
def test_set_aside_experiments_are_refused_by_name(spec):
    with pytest.raises(ValueError, match="not a setting the engine has"):
        parse_run(spec)


def test_credited_runs_are_measured_against_a_credited_ceiling():
    """Recovery credit changes dispatch for every signal, perfect foresight included."""
    from scripts.compare_offer_valuation import foresight

    def run(net):
        return {"revenue": {"all": {"net": net}}}
    rows = {"pf_lp_vint": run(100.0), "pf_lp_recany_vint": run(120.0),
            "naive_lp_shrink0.5_bid_vint": run(80.0), "ml_lp_shrink0.5_bid_vint": run(82.0),
            "naive_lp_shrink0.5_recany_bid_vint": run(90.0), "ml_lp_shrink0.5_recany_bid_vint": run(93.0)}
    assert foresight(rows, "lp_shrink0.5_bid_vint", "all") == pytest.approx(0.1)          # 2 / 20
    assert foresight(rows, "lp_shrink0.5_recany_bid_vint", "all") == pytest.approx(0.1)   # 3 / 30, not 3 / 10


def test_the_margin_is_not_asked_for_twice():
    with pytest.raises(ValueError, match="already keeps the margin"):
        parse_run("ml:lp:0.5:recany:margin:bid:vint")
