"""Hand-checkable tests for storage-change, seasonal-expectation and price features."""

import math

import numpy as np
import pandas as pd
import pytest

from natgas_quant import features as F


def _dates(*s):
    return pd.Series(pd.to_datetime(list(s)))


# ---------------------------------------------------------------------------
# Weekly change
# ---------------------------------------------------------------------------
def test_weekly_change_simple_difference():
    d = _dates("2020-01-03", "2020-01-10", "2020-01-17")
    lv = pd.Series([3000.0, 2900.0, 2950.0])
    ch = F.weekly_change(d, lv)
    assert math.isnan(ch.iloc[0])
    assert ch.iloc[1] == -100.0
    assert ch.iloc[2] == 50.0


def test_weekly_change_not_computed_across_missing_week():
    d = _dates("2020-01-03", "2020-01-10", "2020-01-24")  # 2020-01-17 missing
    ch = F.weekly_change(d, pd.Series([3000.0, 2900.0, 2800.0]))
    assert ch.iloc[1] == -100.0
    assert math.isnan(ch.iloc[2])


def test_weekly_change_requires_sorted_dates():
    with pytest.raises(ValueError):
        F.weekly_change(_dates("2020-01-10", "2020-01-03"), pd.Series([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Seasonal expectation: the worked example from the docstring
# ---------------------------------------------------------------------------
def _example():
    # Same calendar week in 2017, 2018, 2019 with changes 70, 80, 90; current 2020 change 95.
    d = _dates("2017-06-02", "2018-06-01", "2019-05-31", "2020-06-05")
    x = pd.Series([70.0, 80.0, 90.0, 95.0])
    return d, x


def test_expanding_mean_hand_example():
    d, x = _example()
    se = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=3)
    assert se["expected"].iloc[3] == pytest.approx(80.0)
    assert se["scale"].iloc[3] == pytest.approx(10.0)
    an = F.storage_anomaly(x, se["expected"], se["scale"])
    assert an["anomaly_bcf"].iloc[3] == pytest.approx(15.0)
    assert an["z"].iloc[3] == pytest.approx(1.5)


def test_current_and_future_observations_never_enter_expectation():
    d, x = _example()
    base = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=3)
    # Add (a) a huge value one week before the current week in the SAME year and
    # (b) a huge value in a FUTURE year. Neither may change the 2020 expectation.
    d2 = _dates("2017-06-02", "2018-06-01", "2019-05-31", "2020-05-29", "2020-06-05", "2021-06-04")
    x2 = pd.Series([70.0, 80.0, 90.0, 5000.0, 95.0, -5000.0])
    se2 = F.seasonal_expectation(d2, x2, "expanding_mean", window_days=10, min_prior_years=3)
    assert se2["expected"].iloc[4] == pytest.approx(base["expected"].iloc[3])
    assert se2["scale"].iloc[4] == pytest.approx(base["scale"].iloc[3])
    # And changing the current value itself does not move its own baseline.
    x3 = x.copy()
    x3.iloc[3] = 1e6
    se3 = F.seasonal_expectation(d, x3, "expanding_mean", window_days=10, min_prior_years=3)
    assert se3["expected"].iloc[3] == pytest.approx(80.0)


def test_minimum_history_required():
    d, x = _example()
    se = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=4)
    assert math.isnan(se["expected"].iloc[3])
    assert se["n_years"].iloc[3] == 3


def test_median_and_mad_scale():
    d = _dates("2017-06-02", "2018-06-01", "2019-05-31", "2020-06-05")
    x = pd.Series([70.0, 80.0, 200.0, 95.0])
    se = F.seasonal_expectation(d, x, "expanding_median", window_days=10, min_prior_years=3)
    assert se["expected"].iloc[3] == pytest.approx(80.0)
    # |70-80|, |80-80|, |200-80| -> median 10 -> 1.4826 * 10
    assert se["scale"].iloc[3] == pytest.approx(14.826)
    # the mean version is pulled up by the outlier
    sm = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=3)
    assert sm["expected"].iloc[3] == pytest.approx(350.0 / 3)


def test_rolling_window_uses_only_recent_years():
    d = _dates("2013-06-07", "2014-06-06", "2015-06-05", "2016-06-03",
               "2017-06-02", "2018-06-01", "2019-05-31", "2020-06-05")
    x = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 70.0, 80.0, 90.0, 0.0])
    roll = F.seasonal_expectation(d, x, "rolling_mean", window_days=10, min_prior_years=3, rolling_years=3)
    expd = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=3)
    assert roll["expected"].iloc[7] == pytest.approx(80.0)
    assert expd["expected"].iloc[7] == pytest.approx((4000 + 240) / 7)


def test_seasonal_distance_wraps_new_year():
    d = pd.Timestamp("2020-01-02")
    others = pd.to_datetime(["2018-12-30", "2019-07-02"]).to_numpy()
    dist = F.seasonal_distance_days(d, others)
    assert dist[0] == pytest.approx(3.5, abs=1.0)
    assert dist[1] > 150


def test_zero_scale_gives_missing_z():
    d = _dates("2017-06-02", "2018-06-01", "2019-05-31", "2020-06-05")
    x = pd.Series([80.0, 80.0, 80.0, 95.0])
    se = F.seasonal_expectation(d, x, "expanding_mean", window_days=10, min_prior_years=3)
    assert math.isnan(se["scale"].iloc[3])


# ---------------------------------------------------------------------------
# Prior price features
# ---------------------------------------------------------------------------
def test_prior_price_features_hand_values():
    p = np.array([2.0, 2.2, 2.0, 2.5, 2.4, 2.6, 3.0])
    b = np.array([6, 5, 0, -1])
    out = F.prior_price_features(p, b, vol_days=3)
    assert out["prior_ret_1d"].iloc[0] == pytest.approx(math.log(3.0 / 2.6))
    assert out["prior_ret_5d"].iloc[0] == pytest.approx(math.log(3.0 / 2.2))
    lr = np.log(p[1:] / p[:-1])
    assert out["trailing_vol_3d"].iloc[0] == pytest.approx(np.std(lr[-3:], ddof=1))
    assert out["prior_ret_5d"].iloc[1] == pytest.approx(math.log(2.6 / 2.0))
    assert math.isnan(out["prior_ret_1d"].iloc[2])      # no earlier price
    assert out.iloc[3].isna().all()                      # invalid position


def test_lag_by_one_week_respects_gaps():
    d = _dates("2020-01-03", "2020-01-10", "2020-01-24")
    lag = F.lag_by_one_week(d, pd.Series([1.0, 2.0, 3.0]))
    assert math.isnan(lag.iloc[0])
    assert lag.iloc[1] == 1.0
    assert math.isnan(lag.iloc[2])


def test_calendar_features_withdrawal_season():
    we = _dates("2021-01-01", "2021-07-02", "2021-11-05")
    rd = we + pd.Timedelta(days=6)
    cf = F.calendar_features(we, rd)
    assert cf["withdrawal_season"].tolist() == [1, 0, 1]
    assert cf["season"].tolist() == ["DJF", "JJA", "SON"]


def test_winsorize_uses_supplied_bounds_only():
    train = pd.Series(np.arange(101, dtype=float))
    lo, hi = F.fit_winsor_bounds(train, (0.05, 0.95))
    assert (lo, hi) == (5.0, 95.0)
    w = F.winsorize(pd.Series([-100.0, 50.0, 500.0]), (lo, hi))
    assert w.tolist() == [5.0, 50.0, 95.0]
