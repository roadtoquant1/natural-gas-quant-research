"""Forward-return target tests on a tiny hand-made SYNTHETIC price path."""

import math

import numpy as np
import pandas as pd
import pytest

from natgas_quant import targets as T

# Synthetic prices on consecutive business days (Mon 2024-03-04 ... )
DATES = pd.bdate_range("2024-03-04", periods=8)
PRICES = np.array([3.00, 3.10, 3.00, 3.15, 3.30, 3.20, 3.25, 3.10])


def _aligned(entry_pos, before_pos, ok=True):
    return pd.DataFrame({"entry_pos": [entry_pos], "before_pos": [before_pos], "alignment_ok": [ok],
                         "report_release_date": [DATES[max(before_pos + 1, 0)]]})


def test_forward_log_returns_match_hand_calculation():
    out = T.forward_targets(_aligned(3, 1), PRICES, DATES, [1, 3])
    r = out.iloc[0]
    assert r["price_after_release"] == 3.15
    assert r["price_before_release"] == 3.10
    assert r["fwd_ret_1d"] == pytest.approx(math.log(3.30 / 3.15))
    assert r["fwd_ret_3d"] == pytest.approx(math.log(3.25 / 3.15))
    assert r["dir_1d"] == 1.0
    assert r["reaction_ret"] == pytest.approx(math.log(3.15 / 3.10))
    assert r["exit_date_3d"] == DATES[6]


def test_realized_volatility_hand_calculation():
    out = T.forward_targets(_aligned(3, 1), PRICES, DATES, [3])
    lr = [math.log(3.30 / 3.15), math.log(3.20 / 3.30), math.log(3.25 / 3.20)]
    assert out.iloc[0]["rv_3d"] == pytest.approx(math.sqrt(sum(x * x for x in lr)))
    one = T.forward_targets(_aligned(3, 1), PRICES, DATES, [1]).iloc[0]
    assert one["rv_1d"] == pytest.approx(abs(one["fwd_ret_1d"]))


def test_horizon_past_end_of_data_is_missing():
    out = T.forward_targets(_aligned(6, 4), PRICES, DATES, [1, 3])
    assert not math.isnan(out.iloc[0]["fwd_ret_1d"])
    assert math.isnan(out.iloc[0]["fwd_ret_3d"])
    assert math.isnan(out.iloc[0]["dir_3d"])


def test_unaligned_rows_produce_no_targets():
    out = T.forward_targets(_aligned(3, 1, ok=False), PRICES, DATES, [1])
    assert out.iloc[0][["fwd_ret_1d", "price_after_release"]].isna().all()


def test_data_gap_inside_horizon_invalidates_target():
    dates = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-25"])  # 20-day hole
    out = T.forward_targets(_aligned(1, 0), np.array([3.0, 3.1, 3.5]), dates, [1])
    assert math.isnan(out.iloc[0]["fwd_ret_1d"])


def test_negative_return_direction_is_zero():
    out = T.forward_targets(_aligned(4, 2), PRICES, DATES, [1])
    assert out.iloc[0]["fwd_ret_1d"] < 0 and out.iloc[0]["dir_1d"] == 0.0


def test_overlap_with_next_release_flag():
    rel = pd.Series(pd.to_datetime(["2024-03-07", "2024-03-14"]))
    ex = pd.Series(pd.to_datetime(["2024-03-15", "2024-03-18"]))
    flag = T.flag_overlap_with_next_release(rel, ex)
    assert flag.tolist() == [True, False]
