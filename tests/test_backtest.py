"""Backtest mechanics on hand-made SYNTHETIC prices."""

import math

import numpy as np
import pandas as pd
import pytest

from natgas_quant import backtest as B

DATES = pd.bdate_range("2024-03-04", periods=12)   # Mon..; synthetic
PRICES = np.array([3.00, 3.00, 3.00, 3.00, 3.06, 3.00, 3.00, 3.00, 3.00, 2.94, 3.00, 3.00])


def _events(entry_positions, z):
    n = len(entry_positions)
    return pd.DataFrame({
        "report_release_date": [DATES[max(e - 1, 0)] for e in entry_positions],
        "entry_pos": entry_positions,
        "alignment_ok": [True] * n,
        "z": z,
    })


def test_threshold_positions():
    z = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    pos = B.generate_positions(z, 1.0)
    assert pos.tolist() == [1, 1, 0, -1, -1, 0]
    assert B.generate_positions(z, 1.5).tolist() == [1, 0, 0, 0, -1, 0]
    with pytest.raises(ValueError):
        B.generate_positions(z, -1)


def test_long_trade_pnl_and_costs_match_docstring_example():
    ev = _events([3], [-2.0])                       # long at 3.00, exit next day at 3.06
    led, wk = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 1, 10.0, 5.0)
    t = led.iloc[0]
    assert t["gross_return"] == pytest.approx(0.02)
    assert t["cost"] == pytest.approx(0.003)
    assert t["net_return"] == pytest.approx(0.017)
    assert t["entry_date"] == DATES[3] and t["exit_date"] == DATES[4]
    assert wk["net_return"].iloc[0] == pytest.approx(0.017)


def test_short_trade_gains_when_price_falls():
    ev = _events([8], [2.0])                        # short at 3.00, exit at 2.94
    led, _ = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 1, 0.0, 0.0)
    assert led.iloc[0]["position"] == -1
    assert led.iloc[0]["gross_return"] == pytest.approx(0.02)


def test_zero_cost_and_no_signal_weeks():
    ev = _events([3, 6], [-2.0, 0.1])
    led, wk = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 1, 0.0, 0.0)
    assert len(led) == 1
    assert wk["status"].tolist() == ["traded", "no_signal"]
    assert wk["net_return"].iloc[1] == 0.0


def test_overlapping_positions_are_skipped():
    ev = _events([1, 3], [-2.0, -2.0])              # 5-day hold from pos 1 ends at pos 6 > pos 3
    led, wk = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 5, 0.0, 0.0)
    assert len(led) == 1
    assert wk["status"].tolist() == ["traded", "skipped_overlap"]


def test_trade_without_exit_price_is_not_taken():
    ev = _events([11], [-2.0])
    led, wk = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 1, 0.0, 0.0)
    assert led.empty and wk["status"].iloc[0] == "no_price"


def test_select_k_requires_minimum_trades():
    ev = _events([3, 8], [-2.0, 2.0])
    k, tab = B.select_k(ev, PRICES, DATES, [0.5, 1.0], 1, 0.0, 0.0, min_trades=5)
    assert k is None and len(tab) == 2
    k2, _ = B.select_k(ev, PRICES, DATES, [0.5, 3.0], 1, 0.0, 0.0, min_trades=1)
    assert k2 == 0.5


def test_random_positions_reproducible():
    idx = pd.RangeIndex(50)
    a = B.random_signal_positions(idx, (0.2, 0.6, 0.2), seed=7)
    b = B.random_signal_positions(idx, (0.2, 0.6, 0.2), seed=7)
    assert a.equals(b)
    assert set(a.unique()) <= {-1, 0, 1}


def test_performance_summary_basic_and_empty():
    ev = _events([3, 8], [-2.0, 2.0])
    led, wk = B.run_backtest(ev, PRICES, DATES, B.generate_positions(ev["z"], 1.0), 1, 0.0, 0.0)
    s = B.performance_summary(wk, led, 52)
    assert s["n_trades"] == 2 and s["hit_rate"] == 1.0
    assert s["cumulative_return_net"] == pytest.approx(1.02 * 1.02 - 1)
    assert B.performance_summary(pd.DataFrame(), pd.DataFrame(), 52)["n_trades"] == 0
