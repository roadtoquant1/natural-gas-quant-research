"""Metric and estimator tests with hand-calculated answers (SYNTHETIC inputs)."""

import math

import numpy as np
import pandas as pd
import pytest

from natgas_quant import metrics as M
from natgas_quant import models as MD
from natgas_quant import validation as V


# ---------------------------------------------------------------------------
# Trading metrics
# ---------------------------------------------------------------------------
def test_sharpe_hand_calculation():
    r = [0.01, -0.01, 0.02, 0.0]
    mean, sd = 0.005, np.std(r, ddof=1)
    assert M.sharpe_ratio(r, 52) == pytest.approx(mean / sd * math.sqrt(52))
    assert math.isnan(M.sharpe_ratio([0.01], 52))
    assert math.isnan(M.sharpe_ratio([0.0, 0.0, 0.0], 52))


def test_max_drawdown_hand_calculation():
    assert M.max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(-0.5)
    assert M.max_drawdown([0.1, 0.1]) == 0.0
    assert M.max_drawdown([]) == 0.0
    assert M.max_drawdown([-0.1]) == pytest.approx(-0.1)


def test_annualized_return_and_vol():
    r = [0.01] * 52
    assert M.annualized_return(r, 52) == pytest.approx(1.01**52 - 1)
    assert M.annualized_volatility([0.01, -0.01], 52) == pytest.approx(np.std([0.01, -0.01], ddof=1) * math.sqrt(52))


# ---------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------
def test_oos_r2_hand_calculation():
    y = [1.0, -1.0, 2.0]
    yhat = [0.5, -0.5, 1.0]
    bench = [0.0, 0.0, 0.0]
    # SSE model = 0.25+0.25+1 = 1.5 ; SSE bench = 1+1+4 = 6
    assert M.oos_r2(y, yhat, bench) == pytest.approx(1 - 1.5 / 6)


def test_hit_rate_ignores_zero_forecasts():
    assert M.hit_rate([1.0, -1.0, 1.0, 1.0], [0.5, 0.5, 0.0, 2.0]) == pytest.approx(2 / 3)


def test_missing_values_are_dropped_pairwise():
    assert M.mse([1.0, np.nan, 3.0], [1.0, 2.0, 1.0]) == pytest.approx(2.0)


def test_newey_west_lag_rule():
    assert M.newey_west_lags(100) == 4
    assert M.newey_west_lags(500) == 5


def test_clark_west_detects_informative_forecast():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    y = 0.3 * x + rng.normal(size=600)
    res = M.clark_west(y, np.zeros_like(y), 0.3 * x)
    assert res["cw_p_one_sided"] < 0.01
    noise = M.clark_west(y, np.zeros_like(y), 0.3 * rng.normal(size=600))
    assert noise["cw_p_one_sided"] > 0.05


def test_block_bootstrap_reproducible_and_reasonable():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 400)
    draws1 = M.block_bootstrap(lambda idx: x[idx].mean(), len(x), reps=300, seed=5)
    draws2 = M.block_bootstrap(lambda idx: x[idx].mean(), len(x), reps=300, seed=5)
    assert np.array_equal(draws1, draws2)
    lo, hi = M.percentile_ci(draws1)
    assert lo < x.mean() < hi
    idx = M.moving_block_indices(10, 4, np.random.default_rng(0))
    assert len(idx) == 10 and idx.max() < 10


def test_classification_report_perfect_and_constant():
    rep = M.classification_report([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert rep["roc_auc"] == 1.0 and rep["accuracy"] == 1.0 and rep["tp"] == 2
    one_class = M.classification_report([1, 1, 1], [0.6, 0.7, 0.8])
    assert math.isnan(one_class["roc_auc"])


def test_calibration_table_shape():
    tab = M.calibration_table(np.arange(20), np.arange(20), bins=4)
    assert len(tab) == 4 and tab["n"].sum() == 20


# ---------------------------------------------------------------------------
# OLS and diagnostics
# ---------------------------------------------------------------------------
def test_ols_recovers_exact_coefficients():
    x = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = 1.0 + 2.0 * x["x"]
    res = MD.ols(y, x, cov_type="nonrobust")
    assert res.params["const"] == pytest.approx(1.0)
    assert res.params["x"] == pytest.approx(2.0)
    assert res.rsquared == pytest.approx(1.0)


def test_ols_classical_standard_error_hand_calculation():
    x = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 2.0, 4.0])
    res = MD.ols(y, x, cov_type="nonrobust")
    # slope = 0.9, intercept = 0.9 ; residuals 0.1, 0.2, -0.7, 0.4 ; SSE = 0.70
    assert res.params["x"] == pytest.approx(0.9)
    s2 = 0.70 / 2
    assert res.bse["x"] == pytest.approx(math.sqrt(s2 / 5.0))   # Sxx = 5


def test_hac_with_zero_lags_equals_hc0():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"x": rng.normal(size=200)})
    y = 0.5 * x["x"] + rng.normal(size=200) * (1 + np.abs(x["x"]))
    hac0 = MD.ols(y, x, cov_type="HAC", hac_lags=0)
    hc1 = MD.ols(y, x, cov_type="HC1")
    n, k = 200, 2
    assert hac0.bse["x"] == pytest.approx(hc1.bse["x"] * math.sqrt((n - k) / n))


def test_hac_matches_statsmodels_when_available():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(4)
    e = np.zeros(300)
    for t in range(1, 300):
        e[t] = 0.5 * e[t - 1] + rng.normal()
    x = pd.DataFrame({"x": rng.normal(size=300)})
    y = 0.2 * x["x"] + e
    ours = MD.ols(y, x, cov_type="HAC", hac_lags=5)
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    np.testing.assert_allclose(ours.params.to_numpy(), ref.params.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(ours.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-8)


def test_ljung_box_flags_autocorrelation():
    rng = np.random.default_rng(5)
    white = rng.normal(size=500)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    assert MD.ljung_box(white, 4)["lb_p"] > 0.01
    assert MD.ljung_box(ar, 4)["lb_p"] < 1e-6


def test_vif_detects_collinearity():
    rng = np.random.default_rng(6)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": a + 0.01 * rng.normal(size=200), "c": rng.normal(size=200)})
    v = MD.variance_inflation(X)
    assert v["a"] > 100 and v["c"] < 2


# ---------------------------------------------------------------------------
# Chronological validation
# ---------------------------------------------------------------------------
def test_chronological_split_never_shuffles():
    d = pd.Series(pd.date_range("2015-01-01", periods=100, freq="W"))
    lab, b = V.chronological_split(d, {"train": 0.6, "validation": 0.2, "test": 0.2})
    assert lab.iloc[:60].eq("train").all() and lab.iloc[60:80].eq("validation").all()
    assert lab.iloc[80:].eq("test").all()
    assert b["train_end"] < b["validation_start"] < b["test_start"]


def test_walk_forward_trains_only_on_past_years():
    d = pd.date_range("2015-01-02", periods=52 * 6, freq="W-FRI")
    df = pd.DataFrame({"date": d, "y": np.arange(len(d), dtype=float)})
    preds = V.walk_forward_by_year(df, "date", "y", lambda: MD.HistoricalMean(), min_train_years=2)
    assert (preds["train_end"] < preds["forecast_date"]).all()
    first = preds[preds["year"] == preds["year"].min()]
    assert first["n_train"].iloc[0] == (df["date"].dt.year < first["year"].iloc[0]).sum()


def test_frozen_spec_detects_tampering():
    import json
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "frozen.json"
    V.save_frozen({"k": 1.0}, p)
    assert V.load_frozen(p)["spec"]["k"] == 1.0
    payload = json.loads(p.read_text())
    payload["spec"]["k"] = 0.5
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        V.load_frozen(p)


def test_report_renders_pending_sections_without_results():
    """With no result tables, the report must say 'pending' rather than show numbers."""
    import copy
    import tempfile
    from pathlib import Path

    from natgas_quant.config import load_config
    from natgas_quant.reporting import write_report

    cfg = copy.deepcopy(load_config())
    tmp = Path(tempfile.mkdtemp())
    cfg["paths"].update({"tables": str(tmp / "tables"), "figures": str(tmp / "figures"),
                         "processed": str(tmp / "processed"), "report": str(tmp / "report.md")})
    text = write_report(cfg).read_text()
    assert "## 20. Conclusion" in text and "Pending" in text
    assert "H0 **is" not in text   # no verdict without results
