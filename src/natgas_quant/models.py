"""Statistical models, from simplest to most flexible.

Fitting vs prediction vs inference
----------------------------------
* **Fitting** estimates coefficients from data (e.g. beta in r = a + beta*Z).
* **Inference** asks how sure we are about those coefficients: standard
  errors, confidence intervals, p-values. It is about the *past sample*.
* **Prediction** uses fitted coefficients on data the model has not seen.
  A coefficient can be "significant" in-sample yet useless for prediction.
* **Backtesting** turns predictions into hypothetical trades with costs.

Why Newey-West (HAC) standard errors
------------------------------------
Ordinary least squares (OLS) standard errors assume every error term has the
same variance and is unrelated to every other one. Financial returns violate
both: volatility comes in clusters (heteroskedasticity) and, when holding
windows overlap (e.g. 5-day returns measured every week), neighbouring errors
are correlated (autocorrelation). Under these conditions ordinary standard
errors are usually too small, making effects look more certain than they are.
Newey-West estimates the variance in a way that stays valid under both
problems, up to a chosen number of lags.

OLS with HAC errors is implemented here directly in NumPy so the formula is
visible; ``tests/test_metrics.py`` cross-checks it against statsmodels when
statsmodels is installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import newey_west_lags


# ---------------------------------------------------------------------------
# OLS with HAC / HC1 / classical covariance
# ---------------------------------------------------------------------------
@dataclass
class OLSResult:
    """Container for an OLS fit (mirrors the parts of statsmodels we report)."""

    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    rsquared: float
    rsquared_adj: float
    nobs: int
    df_resid: int
    cov_type: str
    hac_lags: int | None
    resid: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)
    X: np.ndarray = field(repr=False)

    def summary_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.bse,
                "t": self.tvalues,
                "p_value": self.pvalues,
                "ci_low": self.conf_int["low"],
                "ci_high": self.conf_int["high"],
            }
        )
        return df

    def info(self) -> dict[str, Any]:
        return {
            "nobs": self.nobs,
            "r_squared": self.rsquared,
            "adj_r_squared": self.rsquared_adj,
            "cov_type": self.cov_type,
            "hac_lags": self.hac_lags,
        }


def _hac_meat(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West 'meat': sum of weighted autocovariances of x_t * e_t (Bartlett kernel)."""
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    return S


def ols(
    y: Sequence[float] | pd.Series,
    X: pd.DataFrame,
    add_constant: bool = True,
    cov_type: str = "HAC",
    hac_lags: int | None = None,
    alpha: float = 0.05,
) -> OLSResult:
    """Ordinary least squares with a choice of covariance estimator.

    Rows with any missing value are dropped. ``cov_type`` is ``"HAC"``
    (Newey-West, Bartlett kernel, no small-sample correction), ``"HC1"``
    (heteroskedasticity-robust) or ``"nonrobust"`` (classical iid).
    p-values and intervals use a t distribution with n - k degrees of freedom,
    which is slightly more conservative than the normal approximation.
    """
    Xd = X.copy()
    if add_constant:
        Xd.insert(0, "const", 1.0)
    yv = pd.Series(np.asarray(y, dtype=float), index=Xd.index)
    data = pd.concat([yv.rename("__y__"), Xd], axis=1).dropna()
    names = list(Xd.columns)
    Y = data["__y__"].to_numpy(dtype=float)
    M = data[names].to_numpy(dtype=float)
    n, k = M.shape
    if n <= k:
        raise ValueError(f"Not enough observations ({n}) for {k} parameters")
    XtX_inv = np.linalg.pinv(M.T @ M)
    beta = XtX_inv @ M.T @ Y
    fitted = M @ beta
    e = Y - fitted
    df_resid = n - k
    if cov_type == "nonrobust":
        sigma2 = float(e @ e) / df_resid
        cov = sigma2 * XtX_inv
        lags = None
    elif cov_type == "HC1":
        meat = (M * e[:, None]).T @ (M * e[:, None])
        cov = XtX_inv @ meat @ XtX_inv * n / df_resid
        lags = None
    elif cov_type == "HAC":
        lags = newey_west_lags(n) if hac_lags is None else int(hac_lags)
        cov = XtX_inv @ _hac_meat(M, e, lags) @ XtX_inv
    else:
        raise ValueError("cov_type must be HAC, HC1 or nonrobust")
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=df_resid)
    q = stats.t.ppf(1 - alpha / 2, df=df_resid)
    ss_tot = float(np.sum((Y - Y.mean()) ** 2)) if add_constant else float(np.sum(Y**2))
    r2 = 1.0 - float(e @ e) / ss_tot if ss_tot > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / df_resid if add_constant else float("nan")
    idx = pd.Index(names)
    return OLSResult(
        params=pd.Series(beta, idx),
        bse=pd.Series(se, idx),
        tvalues=pd.Series(t, idx),
        pvalues=pd.Series(p, idx),
        conf_int=pd.DataFrame({"low": beta - q * se, "high": beta + q * se}, index=idx),
        rsquared=r2,
        rsquared_adj=r2_adj,
        nobs=n,
        df_resid=df_resid,
        cov_type=cov_type,
        hac_lags=lags,
        resid=e,
        fitted=fitted,
        X=M,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def ljung_box(resid: Sequence[float], lags: int = 4) -> dict[str, float]:
    """Ljung-Box Q test for autocorrelation in residuals (H0: no autocorrelation)."""
    e = np.asarray(resid, dtype=float)
    e = e[~np.isnan(e)] - np.nanmean(e)
    n = len(e)
    denom = float(e @ e)
    q = 0.0
    for l in range(1, lags + 1):
        rho = float(e[l:] @ e[:-l]) / denom
        q += rho**2 / (n - l)
    q *= n * (n + 2)
    return {"lb_stat": q, "lb_p": float(stats.chi2.sf(q, df=lags)), "lags": lags}


def breusch_pagan(resid: Sequence[float], X: np.ndarray) -> dict[str, float]:
    """Breusch-Pagan LM test (H0: constant residual variance).

    Regress squared residuals on the regressors; LM = n * R^2 ~ chi2(k-1).
    ``X`` must include the constant column.
    """
    e2 = np.asarray(resid, dtype=float) ** 2
    M = np.asarray(X, dtype=float)
    beta = np.linalg.pinv(M.T @ M) @ M.T @ e2
    fit = M @ beta
    r2 = 1 - np.sum((e2 - fit) ** 2) / np.sum((e2 - e2.mean()) ** 2)
    k = M.shape[1]
    lm = len(e2) * r2
    return {"bp_stat": float(lm), "bp_p": float(stats.chi2.sf(lm, df=max(k - 1, 1)))}


def jarque_bera(resid: Sequence[float]) -> dict[str, float]:
    """Normality test for residuals (financial returns usually fail it: fat tails)."""
    e = np.asarray(resid, dtype=float)
    res = stats.jarque_bera(e[~np.isnan(e)])
    return {"jb_stat": float(res.statistic), "jb_p": float(res.pvalue),
            "skew": float(stats.skew(e)), "excess_kurtosis": float(stats.kurtosis(e))}


def durbin_watson(resid: Sequence[float]) -> float:
    e = np.asarray(resid, dtype=float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e**2))


def variance_inflation(X: pd.DataFrame) -> pd.Series:
    """VIF for each column (no constant in X). Values above ~5-10 signal multicollinearity."""
    Xc = X.dropna().to_numpy(dtype=float)
    out = {}
    for j, name in enumerate(X.columns):
        others = np.delete(Xc, j, axis=1)
        A = np.column_stack([np.ones(len(Xc)), others])
        beta = np.linalg.pinv(A.T @ A) @ A.T @ Xc[:, j]
        resid = Xc[:, j] - A @ beta
        ss = np.sum((Xc[:, j] - Xc[:, j].mean()) ** 2)
        r2 = 1 - np.sum(resid**2) / ss if ss > 0 else float("nan")
        out[name] = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
    return pd.Series(out, name="vif")


def influence(res: OLSResult) -> pd.DataFrame:
    """Leverage and Cook's distance for each observation used in the fit.

    Cook's distance > 4/n is a common flag for observations that move the
    coefficients a lot.
    """
    M, e = res.X, res.resid
    n, k = M.shape
    H = M @ np.linalg.pinv(M.T @ M)
    lev = np.einsum("ij,ij->i", H, M)
    s2 = float(e @ e) / (n - k)
    cooks = (e**2 / (k * s2)) * lev / (1 - lev) ** 2
    return pd.DataFrame({"leverage": lev, "cooks_distance": cooks, "resid": e})


def diagnostics(res: OLSResult, lb_lags: int = 4) -> dict[str, float]:
    """Bundle of residual diagnostics for one regression."""
    d: dict[str, float] = {}
    d.update(ljung_box(res.resid, lb_lags))
    d.update(breusch_pagan(res.resid, res.X))
    d.update(jarque_bera(res.resid))
    d["durbin_watson"] = durbin_watson(res.resid)
    inf = influence(res)
    d["n_high_cooks"] = int((inf["cooks_distance"] > 4 / res.nobs).sum())
    d["max_cooks"] = float(inf["cooks_distance"].max())
    return d


# ---------------------------------------------------------------------------
# Forecasting models with a common fit/predict interface
# ---------------------------------------------------------------------------
class ZeroForecast:
    """Model 0a: always predict a zero return."""

    name = "zero"

    def fit(self, train: pd.DataFrame, target: str) -> "ZeroForecast":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(test))


class HistoricalMean:
    """Model 0b: predict the average target observed in the training data."""

    name = "historical_mean"

    def fit(self, train: pd.DataFrame, target: str) -> "HistoricalMean":
        self.mean_ = float(train[target].mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.full(len(test), self.mean_)


class SeasonalMean:
    """Model 0d: average target by storage season (withdrawal vs injection) in training data."""

    name = "seasonal_mean"

    def __init__(self, season_col: str = "withdrawal_season") -> None:
        self.season_col = season_col

    def fit(self, train: pd.DataFrame, target: str) -> "SeasonalMean":
        self.overall_ = float(train[target].mean())
        self.means_ = train.groupby(self.season_col)[target].mean().to_dict()
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test[self.season_col].map(self.means_).fillna(self.overall_).to_numpy(dtype=float)


class OLSForecast:
    """Models 1-2: linear regression used for forecasting."""

    def __init__(self, features: list[str], name: str = "ols") -> None:
        self.features = features
        self.name = name

    def fit(self, train: pd.DataFrame, target: str) -> "OLSForecast":
        self.res_ = ols(train[target], train[self.features], cov_type="HAC")
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = test[self.features].to_numpy(dtype=float)
        p = self.res_.params
        return p["const"] + X @ p[self.features].to_numpy()


class SklearnForecast:
    """Wrap a scikit-learn estimator; hyperparameters tuned with TimeSeriesSplit on training rows only.

    ``TimeSeriesSplit`` makes every inner validation fold come *after* its
    inner training fold, so tuning never peeks forward. Scaling sits inside the
    pipeline, so it is fitted on the inner training folds only.
    """

    def __init__(self, name: str, features: list[str], estimator: Any, grid: dict[str, list[Any]],
                 n_splits: int = 5, scoring: str = "neg_mean_squared_error", scale: bool = True) -> None:
        self.name = name
        self.features = features
        self.estimator = estimator
        self.grid = grid
        self.n_splits = n_splits
        self.scoring = scoring
        self.scale = scale

    def _pipeline(self) -> Any:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = [("scale", StandardScaler())] if self.scale else []
        steps.append(("model", self.estimator))
        return Pipeline(steps)

    def fit(self, train: pd.DataFrame, target: str) -> "SklearnForecast":
        from sklearn.base import clone
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        X = train[self.features].to_numpy(dtype=float)
        y = train[target].to_numpy(dtype=float)
        if self.grid:
            gs = GridSearchCV(
                self._pipeline(),
                {f"model__{k}": v for k, v in self.grid.items()},
                cv=TimeSeriesSplit(n_splits=self.n_splits),
                scoring=self.scoring,
            )
            gs.fit(X, y)
            self.model_ = gs.best_estimator_
            self.best_params_ = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        else:
            self.model_ = clone(self._pipeline()).fit(X, y)
            self.best_params_ = {}
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = test[self.features].to_numpy(dtype=float)
        if hasattr(self.model_, "predict_proba") and self.scoring != "neg_mean_squared_error":
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)


class MajorityClass:
    """Model 0c (direction): predict the training-sample frequency of up moves."""

    name = "majority_class"

    def fit(self, train: pd.DataFrame, target: str) -> "MajorityClass":
        self.p_ = float(train[target].mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.full(len(test), self.p_)


def make_return_models(cfg_protocol: dict, seed: int) -> dict[str, Any]:
    """Factories for every return model compared in the study, in order of complexity."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Lasso, Ridge

    f = cfg_protocol["model_features"]
    return {
        "zero": lambda: ZeroForecast(),
        "historical_mean": lambda: HistoricalMean(),
        "seasonal_mean": lambda: SeasonalMean(),
        "m1_ols_z": lambda: OLSForecast(f["m1"], "m1_ols_z"),
        "m2_ols_controls": lambda: OLSForecast(f["m2"], "m2_ols_controls"),
        "m4_ridge": lambda: SklearnForecast("m4_ridge", f["wide"], Ridge(),
                                            {"alpha": list(np.logspace(-2, 4, 13))}),
        "m4_lasso": lambda: SklearnForecast("m4_lasso", f["wide"], Lasso(max_iter=50000),
                                            {"alpha": list(np.logspace(-6, -1, 11))}),
        "m5_gbm": lambda: SklearnForecast(
            "m5_gbm", f["wide"],
            GradientBoostingRegressor(max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed),
            {"n_estimators": [50, 100, 200], "min_samples_leaf": [20, 40]},
            scale=False,
        ),
    }


def make_direction_models(cfg_protocol: dict) -> dict[str, Any]:
    """Logistic regression (Model 3) and its majority-class baseline."""
    from sklearn.linear_model import LogisticRegression

    f = cfg_protocol["model_features"]
    return {
        "majority_class": lambda: MajorityClass(),
        "m3_logit_z": lambda: SklearnForecast("m3_logit_z", f["m1"], LogisticRegression(C=1e6, max_iter=1000),
                                              {}, scoring="neg_log_loss"),
        "m3_logit_controls": lambda: SklearnForecast("m3_logit_controls", f["m2"],
                                                     LogisticRegression(C=1e6, max_iter=1000),
                                                     {}, scoring="neg_log_loss"),
    }


class TrailingVolBaseline:
    """Volatility baseline: trailing daily vol scaled to the horizon, sqrt(h) * sigma_20d."""

    name = "trailing_vol_scaled"

    def __init__(self, horizon: int, vol_col: str = "trailing_vol_20d") -> None:
        self.h = horizon
        self.vol_col = vol_col

    def fit(self, train: pd.DataFrame, target: str) -> "TrailingVolBaseline":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return math.sqrt(self.h) * test[self.vol_col].to_numpy(dtype=float)


def make_volatility_models(cfg_protocol: dict, horizon: int) -> dict[str, Any]:
    f = cfg_protocol["volatility_features"]
    controls = [c for c in f if c != "abs_z"]
    return {
        "historical_mean": lambda: HistoricalMean(),
        "trailing_vol_scaled": lambda: TrailingVolBaseline(horizon),
        "vol_ols_controls_only": lambda: OLSForecast(controls, "vol_ols_controls_only"),
        "vol_ols_abs_z": lambda: OLSForecast(f, "vol_ols_abs_z"),
    }
