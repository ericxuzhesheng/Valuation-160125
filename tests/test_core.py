import math

import pytest

from valuation_160125.core import (
    benchmark_return,
    estimate_nav,
    numeric_cross_check,
    summarize_backtest_errors,
)


def test_benchmark_return_combines_hsi_fx_and_cash():
    result = benchmark_return(
        previous_hsi=25_858.88,
        current_hsi=25_884.43,
        previous_hkd_cny=0.92,
        current_hkd_cny=0.921,
        equity_weight=0.95,
        cash_return=0.0001,
    )

    expected_equity = (25_884.43 / 25_858.88) * (0.921 / 0.92) - 1
    expected = 0.95 * expected_equity + 0.05 * 0.0001
    assert result == pytest.approx(expected)


def test_estimate_nav_is_based_on_previous_published_nav():
    assert estimate_nav(1.5568, 0.001) == pytest.approx(1.5583568)


def test_cross_check_distinguishes_match_and_mismatch():
    assert numeric_cross_check(100.0, 100.004, absolute_tolerance=0.01).status == "match"
    assert numeric_cross_check(100.0, 100.2, absolute_tolerance=0.01).status == "mismatch"
    assert numeric_cross_check(None, 100.0).status == "missing"


def test_backtest_summary_is_auditable():
    summary = summarize_backtest_errors([0.001, -0.002, 0.003, math.nan])
    assert summary["observations"] == 3
    assert summary["mae"] == pytest.approx(0.002)
    assert summary["max_abs_error"] == pytest.approx(0.003)

