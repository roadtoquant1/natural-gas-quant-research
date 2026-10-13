"""Forecast, classification, volatility and trading metrics, plus bootstrap tools.

Key ideas for beginners
-----------------------
*Out-of-sample R^2* compares a model's squared forecast errors with those of a
simple benchmark (here: the historical average return known at the time):

    R2_oos = 1 - sum (y - yhat_model)^2 / sum (y - yhat_benchmark)^2

Positive means the model beat the benchmark; negative means you would have
done better ignoring the model. For daily returns even +0.5% can matter, and
values below zero are common.

*Information coefficient (IC)*: here, the Spearman rank correlation between
forecasts and realized returns. It asks "were higher forecasts followed by
higher returns?" without caring about scale.

*Block bootstrap*: to get a confidence interval we resample the data many times
and recompute the statistic. Weekly market data can be correlated over time,
so we resample *blocks* of consecutive observations, which keeps short-range
dependence inside each block. Plain resampling of single points would
understate uncertainty if dependence exists.

*Clark-West test*: when the model nests the benchmark (our models contain an
intercept, the benchmark is the intercept alone), the bigger model's forecasts
are noisier just from estimating extra coefficients. Clark and West (2007)
adjust for that; a small one-sided p-value says the model's forecasts are
genuinely better than the benchmark's.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_pairs(*arrays: Sequence[float]) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        if len(a) != len(arrs[0]):
            raise ValueError("arrays must have equal length")
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrs]


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb HAC lag length: floor(4 * (n/100)^(2/9))."""
    if n <= 0:
        return 0
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def default_block_length(n: int) -> int:
    """Rule-of-thumb block length for the moving-block bootstrap: ceil(n^(1/3))."""
    return max(1, int(math.ceil(n ** (1.0 / 3.0))))


# ---------------------------------------------------------------------------
# Point forecasts
# ---------------------------------------------------------------------------
def mse(y: Sequence[float], yhat: Sequence[float]) -> float:
    y, yhat = _clean_pairs(y, yhat)
    return float(np.mean((y - yhat) ** 2)) if len(y) else float("nan")


def rmse(y: Sequence[float], yhat: Sequence[float]) -> float:
    return math.sqrt(mse(y, yhat))


def mae(y: Sequence[float], yhat: Sequence[float]) -> float:
    y, yhat = _clean_pairs(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")


def oos_r2(y: Sequence[float], yhat: Sequence[float], ybench: Sequence[float]) -> float:
    """1 - SSE(model)/SSE(benchmark), using rows where all three are present."""
    y, yhat, yb = _clean_pairs(y, yhat, ybench)
    denom = float(np.sum((y - yb) ** 2))
    if len(y) == 0 or denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / denom


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = _clean_pairs(x, y)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(x, y)[0])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = _clean_pairs(x, y)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def information_coefficient(yhat: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between forecasts and outcomes."""
    return spearman(yhat, y)


def hit_rate(y: Sequence[float], yhat: Sequence[float]) -> float:
    """Share of cases where sign(forecast) == sign(outcome), ignoring zero forecasts/outcomes."""
    y, yhat = _clean_pairs(y, yhat)
    m = (yhat != 0) & (y != 0)
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(y[m]) == np.sign(yhat[m])))


def hac_mean_test(x: Sequence[float], lags: int | None = None) -> dict[str, float]:
    """Mean of x with a Newey-West standard error and two-sided t-test of mean = 0."""
    (v,) = _clean_pairs(x)
    n = len(v)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    L = newey_west_lags(n) if lags is None else lags
    u = v - v.mean()
    s = float(np.sum(u * u)) / n
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        s += 2.0 * w * float(np.sum(u[l:] * u[:-l])) / n
    se = math.sqrt(max(s, 0.0) / n)
    t = v.mean() / se if se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), df=n - 1) if se > 0 else float("nan")
    return {"mean": float(v.mean()), "se": se, "t": float(t), "p": float(p), "n": n}


def clark_west(
    y: Sequence[float], f_bench: Sequence[float], f_model: Sequence[float], lags: int | None = None
) -> dict[str, float]:
    """Clark-West (2007) MSPE-adjusted test; H0: model does not beat the nested benchmark.

    f_adj = (y - f_bench)^2 - [ (y - f_model)^2 - (f_bench - f_model)^2 ]
    One-sided p-value from a HAC t-test that mean(f_adj) > 0.
    """
    y, fb, fm = _clean_pairs(y, f_bench, f_model)
    adj = (y - fb) ** 2 - ((y - fm) ** 2 - (fb - fm) ** 2)
    res = hac_mean_test(adj, lags)
    p_one = stats.t.sf(res["t"], df=max(res["n"] - 1, 1)) if not math.isnan(res["t"]) else float("nan")
    return {"cw_mean": res["mean"], "cw_t": res["t"], "cw_p_one_sided": float(p_one), "n": res["n"]}


def forecast_report(y: Sequence[float], yhat: Sequence[float], ybench: Sequence[float]) -> dict[str, float]:
    """All return-forecast metrics used in the report, in one dict."""
    y_, yh_, yb_ = _clean_pairs(y, yhat, ybench)
    cw = clark_west(y_, yb_, yh_) if len(y_) >= 3 else {"cw_t": float("nan"), "cw_p_one_sided": float("nan")}
    return {
        "n": len(y_),
        "mse": mse(y_, yh_),
        "mae": mae(y_, yh_),
        "oos_r2_vs_hist_mean": oos_r2(y_, yh_, yb_),
        "oos_r2_vs_zero": oos_r2(y_, yh_, np.zeros_like(y_)),
        "pearson": pearson(yh_, y_),
        "spearman_ic": information_coefficient(yh_, y_),
        "hit_rate": hit_rate(y_, yh_),
        "clark_west_t": cw["cw_t"],
        "clark_west_p": cw["cw_p_one_sided"],
    }


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one moving-block bootstrap sample of length n."""
    if n <= 0:
        return np.array([], dtype=int)
    block = max(1, min(block, n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return idx


def block_bootstrap(
    statistic: Callable[[np.ndarray], float],
    n: int,
    reps: int = 2000,
    block: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap distribution of ``statistic(indices)`` using moving blocks.

    ``statistic`` receives an index array into your (time-ordered) data, so it
    can compute anything: a mean, a group mean, a difference in R^2...
    """
    rng = np.random.default_rng(seed)
    b = default_block_length(n) if block is None else block
    out = np.empty(reps)
    for r in range(reps):
        out[r] = statistic(moving_block_indices(n, b, rng))
    return out


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    d = np.asarray(draws, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_report(y: Sequence[float], prob: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, balanced accuracy, ROC-AUC, log loss, Brier score, confusion counts."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score,
    )

    y_, p_ = _clean_pairs(y, prob)
    if len(y_) == 0:
        return {"n": 0}
    yi = y_.astype(int)
    pred = (p_ >= threshold).astype(int)
    pc = np.clip(p_, 1e-6, 1 - 1e-6)
    both = len(np.unique(yi)) == 2
    tn, fp, fn, tp = confusion_matrix(yi, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(yi)),
        "positive_rate": float(yi.mean()),
        "accuracy": float(accuracy_score(yi, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yi, pred)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(yi, p_)) if both else float("nan"),
        "log_loss": float(log_loss(yi, pc, labels=[0, 1])),
        "brier": float(brier_score_loss(yi, p_)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_table(y: Sequence[float], score: Sequence[float], bins: int = 5) -> pd.DataFrame:
    """Mean predicted value vs mean outcome within quantile buckets of the prediction.

    Works for probabilities (direction) and for predicted volatility.
    """
    y_, s_ = _clean_pairs(y, score)
    if len(y_) < bins:
        return pd.DataFrame(columns=["bucket", "n", "mean_prediction", "mean_outcome"])
    df = pd.DataFrame({"y": y_, "s": s_})
    df["bucket"] = pd.qcut(df["s"].rank(method="first"), bins, labels=False) + 1
    g = df.groupby("bucket").agg(n=("y", "size"), mean_prediction=("s", "mean"), mean_outcome=("y", "mean"))
    return g.reset_index()


def volatility_report(y: Sequence[float], yhat: Sequence[float]) -> dict[str, float]:
    y_, yh_ = _clean_pairs(y, yhat)
    return {"n": len(y_), "mae": mae(y_, yh_), "rmse": rmse(y_, yh_),
            "pearson": pearson(yh_, y_), "spearman": spearman(yh_, y_)}


# ---------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------
def sharpe_ratio(period_returns: Sequence[float], periods_per_year: float) -> float:
    """Annualized Sharpe ratio with a zero risk-free rate.

    mean(r)/std(r) * sqrt(periods_per_year). Undefined (NaN) with fewer than
    two observations or zero variance.
    """
    (r,) = _clean_pairs(period_returns)
    if len(r) < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(r) / sd * math.sqrt(periods_per_year))


def equity_curve(period_returns: Sequence[float], start: float = 1.0) -> np.ndarray:
    """Compounded wealth path: start * prod(1 + r)."""
    (r,) = _clean_pairs(period_returns)
    return start * np.cumprod(1.0 + r)


def max_drawdown(period_returns: Sequence[float]) -> float:
    """Largest peak-to-trough fall of the compounded equity curve (negative number, or 0).

    The starting wealth (1.0) counts as the first peak.
    """
    eq = np.concatenate([[1.0], equity_curve(period_returns)])
    if len(eq) == 1:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    return float(np.min(eq / peaks - 1.0))


def annualized_return(period_returns: Sequence[float], periods_per_year: float) -> float:
    (r,) = _clean_pairs(period_returns)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(period_returns: Sequence[float], periods_per_year: float) -> float:
    (r,) = _clean_pairs(period_returns)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))
