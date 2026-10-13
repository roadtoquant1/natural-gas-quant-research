"""Orchestrates Stages 5-11 in two strictly separated phases.

Phase "development"
    Uses ONLY the train and validation periods. Produces the data audit,
    exploratory analysis, event study, in-sample inference, validation-period
    forecasts, robustness checks, and chooses the backtest threshold k and the
    secondary forecasting model. Then it FREEZES the specification
    (``reports/tables/frozen_spec.json``).

Phase "final"
    Refuses to run without the frozen specification. Refits the frozen models
    on train+validation, evaluates the untouched test period once, runs the
    annual walk-forward over the full sample, the test-period backtest, and
    applies the pre-registered evidence rule. Every run is logged in
    ``reports/tables/test_evaluation_log.csv``.

All outputs are CSV/JSON tables in ``reports/tables`` and PNG figures in
``reports/figures``; :mod:`natgas_quant.reporting` turns them into the report.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import backtest as BT
from . import event_study as ES
from . import features as F
from . import metrics as M
from . import models as MD
from . import plotting as P
from . import validation as V
from .config import resolve_path
from .dataset import modeling_sample

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

RETURN_TARGETS = ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
class Outputs:
    """Small helper that writes tables/figures and remembers figure captions."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.tables = resolve_path(cfg["paths"]["tables"])
        self.figures = resolve_path(cfg["paths"]["figures"])
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        cap = self.tables / "figure_captions.json"
        self.captions: dict[str, str] = json.loads(cap.read_text()) if cap.exists() else {}

    def table(self, df: pd.DataFrame, name: str, index: bool = False) -> None:
        df.to_csv(self.tables / f"{name}.csv", index=index)

    def json(self, obj: Any, name: str) -> None:
        (self.tables / f"{name}.json").write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")

    def fig(self, result: tuple[Path, str]) -> None:
        path, caption = result
        self.captions[path.name] = caption
        (self.tables / "figure_captions.json").write_text(json.dumps(self.captions, indent=2), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def load_processed(cfg: dict[str, Any], suffix: str = "") -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    proc = resolve_path(cfg["paths"]["processed"])
    ev = pd.read_csv(proc / f"events{suffix}.csv", low_memory=False)
    for c in [c for c in ev.columns if c.endswith("_date") or c.startswith("exit_date_")]:
        ev[c] = pd.to_datetime(ev[c])
    px = pd.read_csv(proc / "prices_valid.csv", parse_dates=["date"])
    return ev, px["price"].to_numpy(dtype=float), pd.DatetimeIndex(px["date"])


def assign_split(ms: pd.DataFrame, cfg: dict[str, Any], out: Outputs) -> tuple[pd.DataFrame, dict[str, str]]:
    """Use frozen split boundaries if they exist; otherwise create and freeze them."""
    path = out.tables / "frozen_split.json"
    if path.exists():
        bounds = V.load_frozen(path)["spec"]
        log.info("Reusing frozen split boundaries from %s", path.name)
    else:
        _, bounds = V.chronological_split(ms["report_release_date"], cfg["protocol"]["split_fractions"])
        V.save_frozen(bounds, path)
    ms = ms.copy()
    ms["split"] = V.labels_from_bounds(ms["report_release_date"], bounds).to_numpy()
    return ms, bounds


def _complete(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.dropna(subset=cols)


# ---------------------------------------------------------------------------
# Stage 5: audit + EDA
# ---------------------------------------------------------------------------
def data_audit(ev: pd.DataFrame, ms: pd.DataFrame, dev: pd.DataFrame, cfg: dict[str, Any],
               out: Outputs, prices_raw: pd.DataFrame | None) -> None:
    rows = [
        ("storage weeks (all)", len(ev)),
        ("first week-ending", ev["storage_week_ending_date"].min().date()),
        ("last week-ending", ev["storage_week_ending_date"].max().date()),
        ("weeks with storage change", int(ev["storage_change_bcf"].notna().sum())),
        ("weeks with primary anomaly z", int(ev["z"].notna().sum())),
        ("weeks aligned to prices", int(ev["alignment_ok"].sum())),
        ("weeks with release date uncertain", int(ev["release_date_uncertain"].sum())),
        ("weeks using verified release dates", int((ev["release_date_source"] == "verified_schedule").sum())),
        ("duplicate entry dates (aligned weeks)", int(ev["entry_date_duplicated"].sum())),
        ("modeling sample (z and fwd_ret_1d present)", len(ms)),
        ("development sample (train+validation)", len(dev)),
        ("5-day windows overlapping next release", int(ev["overlaps_next_release_5d"].sum())),
    ]
    out.table(pd.DataFrame(rows, columns=["item", "value"]), "audit_coverage")
    miss = ev[["storage_level_bcf", "storage_change_bcf", "z", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d",
               "prior_ret_5d", "trailing_vol_20d", "z_lag1"]].isna().sum().rename("missing").reset_index()
    out.table(miss.rename(columns={"index": "column"}), "audit_missing")
    stats_cols = ["storage_level_bcf", "storage_change_bcf", "anomaly_bcf", "z", "fwd_ret_1d", "fwd_ret_3d",
                  "fwd_ret_5d", "rv_3d", "prior_ret_5d", "trailing_vol_20d"]
    out.table(dev[stats_cols].describe().T.reset_index().rename(columns={"index": "variable"}), "summary_stats_dev")

    ev_plot = ev.dropna(subset=["storage_level_bcf"])
    out.fig(P.storage_level(ev_plot, out.figures / "fig01_storage_level.png"))
    out.fig(P.weekly_change(ev_plot, out.figures / "fig02_weekly_change.png"))
    out.fig(P.seasonal_change(ev_plot, out.figures / "fig03_seasonal_change.png"))
    if prices_raw is not None:
        out.fig(P.price_history(prices_raw, out.figures / "fig04_henry_hub_price.png"))
    out.fig(P.histogram(dev["fwd_ret_1d"], out.figures / "fig05_return_distribution.png",
                        "Distribution of 1-day forward log returns", "Log return", pct=True,
                        caption="One-day forward log returns from the first eligible post-release price (development sample)."))
    out.fig(P.histogram(dev["z"], out.figures / "fig06_anomaly_distribution.png",
                        "Distribution of the standardized storage anomaly", "Z (unitless)",
                        caption="Point-in-time standardized storage anomaly, expanding-mean method (development sample)."))
    out.fig(P.scatter_anomaly_return(dev, "z", "fwd_ret_1d", out.figures / "fig07_scatter_z_return.png",
                                     "Each dot is one release week (development sample); the line joins decile averages."))

    # Quintiles (descriptive; cut points use the whole development sample)
    d = dev.dropna(subset=["z", "fwd_ret_1d"]).copy()
    d["z_quintile"] = pd.qcut(d["z"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    q = ES.group_summary(d, "z_quintile", "fwd_ret_1d", reps=cfg["protocol"]["bootstrap_reps"],
                         seed=cfg["project"]["random_seed"])
    out.table(q, "eda_quintile_returns")
    out.fig(P.bucket_bars(q, out.figures / "fig08_quintile_returns.png",
                          "Mean 1-day forward return by anomaly quintile", "Mean log return",
                          "Q1 = most negative anomalies (less gas than normal), Q5 = most positive. Error bars: 95% "
                          "block-bootstrap intervals. Quintile cut points use the whole development sample, so this "
                          "is descriptive, not a tradable rule.", colors=P.DIVERGING_5))
    d2 = dev.dropna(subset=["abs_z", "rv_3d"]).copy()
    d2["absz_quintile"] = pd.qcut(d2["abs_z"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    qv = ES.group_summary(d2, "absz_quintile", "rv_3d", reps=cfg["protocol"]["bootstrap_reps"],
                          seed=cfg["project"]["random_seed"])
    out.table(qv, "eda_absz_quintile_rv3d")
    out.fig(P.bucket_bars(qv, out.figures / "fig09_absz_quintile_vol.png",
                          "Mean 3-day realized volatility by |Z| quintile", "Realized volatility (log units)",
                          "Q1 = smallest absolute anomalies, Q5 = largest. Error bars: 95% block-bootstrap intervals."))
    corr_cols = ["z", "anomaly_bcf", "z_lag1", "prior_ret_1d", "prior_ret_5d", "trailing_vol_20d",
                 "fwd_ret_1d", "fwd_ret_3d", "rv_3d", "withdrawal_season"]
    corr = dev[corr_cols].corr()
    out.table(corr.reset_index().rename(columns={"index": "variable"}), "eda_correlation")
    out.fig(P.correlation_heatmap(corr, out.figures / "fig10_correlation.png",
                                  "Pearson correlations among features and targets in the development sample. "
                                  "Correlation is not causation."))
    counts = ms.groupby(["year", "storage_season"]).size().unstack(fill_value=0)
    out.table(counts.reset_index(), "sample_counts_year_season")
    out.fig(P.sample_counts(counts, out.figures / "fig11_sample_counts.png"))

    # Flags
    flags = []
    n_ext = int((dev["z"].abs() > 3).sum())
    flags.append(("Extreme anomalies |Z| > 3 (dev)", n_ext))
    top = dev["fwd_ret_1d"].abs().nlargest(5)
    flags.append(("Largest |1-day return| (dev)", ", ".join(f"{v:.3f}" for v in top)))
    share = float(top.pow(2).sum() / dev["fwd_ret_1d"].pow(2).sum())
    flags.append(("Share of squared 1-day returns from top 5 weeks", f"{share:.1%}"))
    flags.append(("Event groups with n < 30", "see event-study tables (column small_sample)"))
    for k, v in flags:
        log.info("Flag: %s = %s", k, v)
    out.table(pd.DataFrame(flags, columns=["flag", "value"]), "eda_flags")


# ---------------------------------------------------------------------------
# Stage 6: event study
# ---------------------------------------------------------------------------
def event_study(df: pd.DataFrame, prices: np.ndarray, cfg: dict[str, Any], out: Outputs, tag: str) -> None:
    pr = cfg["protocol"]
    reps, seed = pr["bootstrap_reps"], cfg["project"]["random_seed"]
    d = df.copy()
    d["group"] = ES.assign_groups(d["z"], pr["event_bins"], pr["event_labels"])
    parts = []
    for tgt in RETURN_TARGETS + ["abs_ret_1d", "rv_3d", "reaction_ret"]:
        s = ES.group_summary(d, "group", tgt, reps=reps, seed=seed)
        s.insert(0, "target", tgt)
        parts.append(s)
    tab = pd.concat(parts, ignore_index=True)
    out.table(tab, f"event_groups_{tag}")
    s1 = tab[tab["target"] == "fwd_ret_1d"]
    out.fig(P.bucket_bars(s1, out.figures / f"fig12_event_groups_{tag}.png",
                          f"Mean 1-day forward return by anomaly group ({tag} sample)", "Mean log return",
                          "Groups use fixed thresholds on point-in-time Z (-1.5, -0.5, +0.5, +1.5). Error bars: 95% "
                          "block-bootstrap intervals. H1 predicts bars falling from left to right.",
                          colors=P.DIVERGING_5))
    by_season = ES.group_summary_by(d, "storage_season", "group", "fwd_ret_1d", reps=reps, seed=seed)
    out.table(by_season, f"event_groups_by_season_{tag}")
    mid = d["report_release_date"].min() + (d["report_release_date"].max() - d["report_release_date"].min()) / 2
    d["subperiod"] = np.where(d["report_release_date"] < mid, f"before {mid.date()}", f"from {mid.date()}")
    out.table(ES.group_summary_by(d, "subperiod", "group", "fwd_ret_1d", reps=reps, seed=seed),
              f"event_groups_by_subperiod_{tag}")
    diffs = []
    for tgt in RETURN_TARGETS:
        r = ES.extreme_minus_extreme(d, "group", tgt, "large_negative", "large_positive", reps=reps, seed=seed)
        diffs.append({"target": tgt, **r})
    out.table(pd.DataFrame(diffs), f"event_extreme_difference_{tag}")
    car = ES.cumulative_return_paths(d, prices)
    if not car.empty:
        car = car.merge(d[["group"]].astype(str), left_on="event", right_index=True)
        out.fig(P.car_paths(car, out.figures / f"fig13_car_paths_{tag}.png"))


# ---------------------------------------------------------------------------
# Stage 7: in-sample inference
# ---------------------------------------------------------------------------
def inference(dev: pd.DataFrame, cfg: dict[str, Any], out: Outputs) -> dict[str, Any]:
    pr = cfg["protocol"]
    f = pr["model_features"]
    res = {}
    d1 = _complete(dev, ["fwd_ret_1d", "z"])
    m1 = MD.ols(d1["fwd_ret_1d"], d1[["z"]], cov_type="HAC")
    m1_iid = MD.ols(d1["fwd_ret_1d"], d1[["z"]], cov_type="nonrobust")
    t = m1.summary_frame().assign(se_type="Newey-West HAC")
    t2 = m1_iid.summary_frame().assign(se_type="classical iid (for comparison only)")
    out.table(pd.concat([t, t2]).reset_index().rename(columns={"index": "term"}), "m1_dev")
    info = m1.info() | {"target": "fwd_ret_1d", "feature": "z"}
    out.json(info, "m1_dev_info")
    res["m1"] = m1
    diag = MD.diagnostics(m1)
    d2 = _complete(dev, ["fwd_ret_1d"] + f["m2"])
    m2 = MD.ols(d2["fwd_ret_1d"], d2[f["m2"]], cov_type="HAC")
    out.table(m2.summary_frame().reset_index().rename(columns={"index": "term"}), "m2_dev")
    out.json(m2.info(), "m2_dev_info")
    diag2 = MD.diagnostics(m2)
    out.table(pd.DataFrame([{"model": "m1", **diag}, {"model": "m2", **diag2}]), "diagnostics_dev")
    out.table(MD.variance_inflation(d2[f["m2"]]).reset_index().rename(columns={"index": "variable"}), "vif_dev")
    out.fig(P.residual_diagnostics(m1.fitted, m1.resid, out.figures / "fig14_m1_residuals.png"))

    # Influence: drop high-Cook's observations and re-estimate
    infl = MD.influence(m1)
    keep = infl["cooks_distance"].to_numpy() <= 4 / m1.nobs
    m1_trim = MD.ols(d1["fwd_ret_1d"][keep], d1[["z"]][keep], cov_type="HAC")
    out.json({"n_dropped": int((~keep).sum()), "beta_full": float(m1.params["z"]),
              "beta_trimmed": float(m1_trim.params["z"]), "p_trimmed": float(m1_trim.pvalues["z"])},
             "m1_influence_dev")

    # Coefficient stability: expanding window by year
    rows = []
    for y in sorted(d1["year"].unique())[2:]:
        sub = d1[d1["year"] <= y]
        r = MD.ols(sub["fwd_ret_1d"], sub[["z"]], cov_type="HAC")
        rows.append({"end_year": y, "n": r.nobs, "coef": r.params["z"], "ci_low": r.conf_int.loc["z", "low"],
                     "ci_high": r.conf_int.loc["z", "high"], "p": r.pvalues["z"]})
    stab = pd.DataFrame(rows)
    out.table(stab, "m1_coefficient_stability_dev")
    if not stab.empty:
        out.fig(P.coefficient_stability(stab, out.figures / "fig15_m1_stability.png", "Z"))

    # Secondary hypothesis: |Z| and volatility, controlling for trailing volatility
    vf = pr["volatility_features"]
    dv = _complete(dev, ["rv_3d"] + vf)
    mv = MD.ols(dv["rv_3d"], dv[vf], cov_type="HAC")
    out.table(mv.summary_frame().reset_index().rename(columns={"index": "term"}), "vol_model_dev")
    out.json(mv.info() | {"target": "rv_3d"}, "vol_model_dev_info")
    res["vol"] = mv
    return res


# ---------------------------------------------------------------------------
# Stages 8-9: fixed-split validation and walk-forward
# ---------------------------------------------------------------------------
def _fit_predict(models: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, target: str) -> pd.DataFrame:
    preds = {}
    fitted = {}
    for name, factory in models.items():
        m = factory().fit(train, target)
        preds[name] = m.predict(test)
        fitted[name] = m
    df = pd.DataFrame(preds, index=test.index)
    df.attrs["fitted"] = fitted
    return df


def evaluate_returns(train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any], target: str = "fwd_ret_1d",
                     in_sample: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit every return model on ``train`` and score it on ``test``.

    ``in_sample=True`` is only for reporting the training-period fit itself.
    """
    cols = cfg["protocol"]["model_features"]["wide"] + [target]
    tr, te = _complete(train, cols), _complete(test, cols)
    if not in_sample:
        V.assert_chronological(tr["report_release_date"], te["report_release_date"])
    models = MD.make_return_models(cfg["protocol"], cfg["project"]["random_seed"])
    preds = _fit_predict(models, tr, te, target)
    bench = preds["historical_mean"]
    rows = [{"model": n, **M.forecast_report(te[target], preds[n], bench)} for n in preds.columns]
    preds.insert(0, "forecast_date", te["report_release_date"])
    preds.insert(1, "y", te[target])
    return pd.DataFrame(rows), preds, preds.attrs["fitted"]


def evaluate_direction(train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = cfg["protocol"]["model_features"]["m2"] + ["dir_1d"]
    tr, te = _complete(train, cols), _complete(test, cols)
    models = MD.make_direction_models(cfg["protocol"])
    preds = _fit_predict(models, tr, te, "dir_1d")
    rows = [{"model": n, **M.classification_report(te["dir_1d"], preds[n])} for n in preds.columns]
    calib = M.calibration_table(te["dir_1d"], preds["m3_logit_z"], bins=5)
    return pd.DataFrame(rows), calib


def evaluate_volatility(train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any], horizon: int = 3
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    tgt = f"rv_{horizon}d"
    cols = cfg["protocol"]["volatility_features"] + [tgt]
    tr, te = _complete(train, cols), _complete(test, cols)
    models = MD.make_volatility_models(cfg["protocol"], horizon)
    preds = _fit_predict(models, tr, te, tgt)
    rows = [{"model": n, **M.volatility_report(te[tgt], preds[n]),
             "oos_r2_vs_hist_mean": M.oos_r2(te[tgt], preds[n], preds["historical_mean"])} for n in preds.columns]
    calib = M.calibration_table(te[tgt], preds["vol_ols_abs_z"], bins=5)
    return pd.DataFrame(rows), calib


def walk_forward(df: pd.DataFrame, cfg: dict[str, Any], last_date: pd.Timestamp | None = None
                 ) -> pd.DataFrame:
    """Annual expanding-window forecasts for every return model (rows up to ``last_date``)."""
    cols = cfg["protocol"]["model_features"]["wide"]
    d = df if last_date is None else df[df["report_release_date"] < last_date]
    d = _complete(d, cols + ["fwd_ret_1d"])
    models = MD.make_return_models(cfg["protocol"], cfg["project"]["random_seed"])
    parts = []
    for name, factory in models.items():
        p = V.walk_forward_by_year(d, "report_release_date", "fwd_ret_1d", factory,
                                   min_train_years=cfg["protocol"]["walk_forward_min_train_years"],
                                   required_cols=cols)
        if not p.empty:
            p["model"] = name
            p["storage_season"] = d.loc[p.index, "storage_season"].to_numpy()
            parts.append(p)
    return pd.concat(parts) if parts else pd.DataFrame()


def walk_forward_metrics(wf: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    if wf.empty:
        return pd.DataFrame()
    bench = wf[wf["model"] == "historical_mean"][["forecast_date", "yhat"]].rename(columns={"yhat": "bench"})
    m = wf.merge(bench, on="forecast_date", how="left")
    keys = ["model"] + ([by] if by else [])
    rows = []
    for k, g in m.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        rows.append({**dict(zip(keys, k)), **M.forecast_report(g["y"], g["yhat"], g["bench"])})
    return pd.DataFrame(rows)


def permutation_importance_table(fitted: dict[str, Any], test: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    feats = cfg["protocol"]["model_features"]["wide"]
    te = _complete(test, feats + ["fwd_ret_1d"])
    rows = []
    for name in ("m4_ridge", "m5_gbm"):
        if name not in fitted:
            continue
        pi = permutation_importance(fitted[name].model_, te[feats].to_numpy(), te["fwd_ret_1d"].to_numpy(),
                                    scoring="neg_mean_squared_error", n_repeats=50,
                                    random_state=cfg["project"]["random_seed"])
        for j, f in enumerate(feats):
            rows.append({"model": name, "feature": f, "mse_increase_mean": pi.importances_mean[j],
                         "mse_increase_std": pi.importances_std[j]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 10: robustness
# ---------------------------------------------------------------------------
def _beta_row(name: str, d: pd.DataFrame, x: str, y: str, expected_sign: int = -1) -> dict[str, Any]:
    """HAC slope of y on x. ``expected_sign`` is -1 for H1 (returns) and +1 for the volatility hypothesis."""
    d = _complete(d, [x, y])
    if len(d) < 30:
        return {"variant": name, "n": len(d), "note": "too few observations"}
    r = MD.ols(d[y], d[[x]], cov_type="HAC")
    sd = float(d[x].std())
    return {"variant": name, "feature": x, "target": y, "n": r.nobs, "beta": r.params[x], "se_hac": r.bse[x],
            "p_value": r.pvalues[x], "beta_per_1sd": r.params[x] * sd,
            "expected_sign": "negative" if expected_sign < 0 else "positive",
            "sign_as_hypothesized": bool(np.sign(r.params[x]) == expected_sign),
            "significant_5pct": bool(r.pvalues[x] < 0.05)}


def robustness(dev: pd.DataFrame, dev_lag2: pd.DataFrame | None, cfg: dict[str, Any], out: Outputs,
               train_mask: pd.Series, all_weeks: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate the primary slope under alternative, pre-listed choices.

    ``all_weeks`` is the full storage history; alternative seasonal baselines
    are rebuilt on it (each week still uses only EARLIER weeks), then the
    development rows are selected.
    """
    rows = [_beta_row("baseline: z expanding mean, 1-day", dev, "z", "fwd_ret_1d")]
    for m in cfg["features"]["methods"]:
        if m != cfg["features"]["primary_method"]:
            rows.append(_beta_row(f"seasonal method: {m}", dev, f"z_{m}", "fwd_ret_1d"))
    # Alternative comparable-week window (recomputed, still point-in-time)
    hist = all_weeks.sort_values("storage_week_ending_date")
    for wd in (17, 24):
        se = F.seasonal_expectation(hist["storage_week_ending_date"], hist["storage_change_bcf"], "expanding_mean",
                                    window_days=wd, min_prior_years=cfg["features"]["min_prior_years"],
                                    same_year_exclusion_days=cfg["features"]["same_year_exclusion_days"])
        z_alt = F.storage_anomaly(hist["storage_change_bcf"], se["expected"], se["scale"])["z"]
        zmap = pd.Series(z_alt.to_numpy(), index=hist["storage_week_ending_date"].to_numpy())
        tmp = dev.assign(z_alt=dev["storage_week_ending_date"].map(zmap).to_numpy())
        rows.append(_beta_row(f"comparable window +/-{wd} days", tmp, "z_alt", "fwd_ret_1d"))
    rows.append(_beta_row("raw anomaly (Bcf) instead of z", dev, "anomaly_bcf", "fwd_ret_1d"))
    for t in ("fwd_ret_3d", "fwd_ret_5d"):
        rows.append(_beta_row(f"target {t}", dev, "z", t))
    for s, g in dev.groupby("storage_season"):
        rows.append(_beta_row(f"season: {s}", g, "z", "fwd_ret_1d"))
    mid = dev["report_release_date"].median()
    rows.append(_beta_row(f"early half (< {mid.date()})", dev[dev["report_release_date"] < mid], "z", "fwd_ret_1d"))
    rows.append(_beta_row(f"late half (>= {mid.date()})", dev[dev["report_release_date"] >= mid], "z", "fwd_ret_1d"))
    rows.append(_beta_row("exclude |z| > 3", dev[dev["z"].abs() <= 3], "z", "fwd_ret_1d"))
    q = dev["fwd_ret_1d"].quantile([0.01, 0.99])
    rows.append(_beta_row("exclude top/bottom 1% returns", dev[dev["fwd_ret_1d"].between(q.iloc[0], q.iloc[1])],
                          "z", "fwd_ret_1d"))
    bounds = F.fit_winsor_bounds(dev.loc[train_mask, "z"], tuple(cfg["features"]["winsor_quantiles"]))
    rows.append(_beta_row("winsorized z (bounds from train only)", dev.assign(z_w=F.winsorize(dev["z"], bounds)),
                          "z_w", "fwd_ret_1d"))
    rows.append(_beta_row("exclude weeks with uncertain release date", dev[~dev["release_date_uncertain"]],
                          "z", "fwd_ret_1d"))
    if dev_lag2 is not None:
        rows.append(_beta_row("entry on 2nd price after release", dev_lag2, "z", "fwd_ret_1d"))
    for label, cut in (("before 2022 (pre Europe gas crisis / Freeport outage)", "2022-01-01"),):
        rows.append(_beta_row(f"subsample {label}", dev[dev["report_release_date"] < cut], "z", "fwd_ret_1d"))
        rows.append(_beta_row(f"subsample from {cut}", dev[dev["report_release_date"] >= cut], "z", "fwd_ret_1d"))
    rows.append(_beta_row("vol: |z| -> rv_3d (no controls)", dev, "abs_z", "rv_3d", expected_sign=+1))
    # Models with/without prior returns and seasonal controls
    f = cfg["protocol"]["model_features"]["m2"]
    for label, feats in (("M2 without prior returns", [c for c in f if not c.startswith("prior_ret")]),
                         ("M2 without seasonal control", [c for c in f if c != "withdrawal_season"]),
                         ("M2 full", f)):
        d = _complete(dev, feats + ["fwd_ret_1d"])
        r = MD.ols(d["fwd_ret_1d"], d[feats], cov_type="HAC")
        rows.append({"variant": label, "feature": "z", "target": "fwd_ret_1d", "n": r.nobs, "beta": r.params["z"],
                     "se_hac": r.bse["z"], "p_value": r.pvalues["z"], "expected_sign": "negative",
                     "sign_as_hypothesized": bool(r.params["z"] < 0),
                     "significant_5pct": bool(r.pvalues["z"] < 0.05)})
    rows.append({"variant": "spot vs tradable futures proxy", "note": "not run: optional futures data not enabled"})
    tab = pd.DataFrame(rows)
    out.table(tab, "robustness_dev")
    return tab


# ---------------------------------------------------------------------------
# Stage 11: backtest helpers
# ---------------------------------------------------------------------------
def _bt_params(cfg: dict[str, Any]) -> dict[str, Any]:
    b = cfg["backtest"]
    return {"holding_days": b["holding_days"], "one_way_cost_bps": b["one_way_cost_bps"],
            "slippage_bps": b["slippage_bps"]}


def backtest_period(events: pd.DataFrame, prices: np.ndarray, pdates: pd.DatetimeIndex, k: float,
                    cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Strategy plus benchmarks on one period."""
    b = cfg["backtest"]
    ppy = b["periods_per_year"]
    kw = _bt_params(cfg)
    pos = BT.generate_positions(events["z"], k)
    led, wk = BT.run_backtest(events, prices, pdates, pos, **kw)
    res = {"strategy": BT.performance_summary(wk, led, ppy)}
    led_l, wk_l = BT.run_backtest(events, prices, pdates, pd.Series(1, index=events.index), **kw)
    res["always_long"] = BT.performance_summary(wk_l, led_l, ppy)
    res["no_trade"] = BT.performance_summary(wk.assign(gross_return=0.0, net_return=0.0, position=0),
                                             led.iloc[0:0], ppy)
    n = max(len(pos), 1)
    probs = (float((pos == 1).sum()) / n, float((pos == 0).sum()) / n, float((pos == -1).sum()) / n)
    if probs[1] == 1.0:
        probs = (0.05, 0.90, 0.05)
    rp = BT.random_signal_positions(events.index, probs, seed)
    led_r, wk_r = BT.run_backtest(events, prices, pdates, rp, **kw)
    res["random_signal_seed"] = BT.performance_summary(wk_r, led_r, ppy)
    dist = BT.random_benchmark_distribution(events, prices, pdates, probs, b["random_benchmark_draws"], seed, **kw,
                                            periods_per_year=ppy)
    strat = res["strategy"]["mean_net_return_per_week"]
    res["random_placebo_share_beating_strategy"] = float((dist["mean_net_return_per_week"] >= strat).mean())
    res["_ledger"], res["_weekly"], res["_weekly_long"], res["_weekly_random"] = led, wk, wk_l, wk_r
    return res


def _curves(res: dict[str, Any]) -> dict[str, pd.Series]:
    def eq(wk: pd.DataFrame, col: str) -> pd.Series:
        return pd.Series(np.cumprod(1 + wk[col].to_numpy()), index=pd.to_datetime(wk["report_release_date"]))
    return {
        "Strategy (net)": eq(res["_weekly"], "net_return"),
        "Strategy (gross)": eq(res["_weekly"], "gross_return"),
        "Always long (net)": eq(res["_weekly_long"], "net_return"),
        "Random signal (net)": eq(res["_weekly_random"], "net_return"),
    }


def _summaries_table(res: dict[str, Any], period: str) -> pd.DataFrame:
    rows = []
    for name in ("strategy", "always_long", "random_signal_seed", "no_trade"):
        rows.append({"period": period, "portfolio": name, **res[name]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
def run_development(cfg: dict[str, Any]) -> dict[str, Any]:
    out = Outputs(cfg)
    ev, prices, pdates = load_processed(cfg)
    proc = resolve_path(cfg["paths"]["processed"])
    lag2_path = proc / "events_lag2.csv"
    ms = modeling_sample(ev, "z", cfg["protocol"]["primary_target"])
    ms, bounds = assign_split(ms, cfg, out)
    dev = ms[ms["split"].isin(["train", "validation"])].copy()
    train, val = ms[ms["split"] == "train"], ms[ms["split"] == "validation"]
    out.json(bounds | {"n_train": len(train), "n_validation": len(val), "n_test": int((ms["split"] == "test").sum())},
             "split_summary")
    raw_px = pd.read_csv(proc / "prices_all.csv", parse_dates=["date"]) if (proc / "prices_all.csv").exists() else None

    data_audit(ev, ms, dev, cfg, out, raw_px)
    event_study(dev, prices, cfg, out, "dev")
    inference(dev, cfg, out)

    tab_ret, preds_val, fitted = evaluate_returns(train, val, cfg)
    out.table(tab_ret, "validation_return_models")
    out.table(preds_val, "predictions_validation")
    out.table(permutation_importance_table(fitted, val, cfg), "permutation_importance_validation")
    tab_dir, calib = evaluate_direction(train, val, cfg)
    out.table(tab_dir, "validation_direction_models")
    out.table(calib, "validation_direction_calibration")
    tab_vol, vcal = evaluate_volatility(train, val, cfg)
    out.table(tab_vol, "validation_volatility_models")
    out.table(vcal, "validation_volatility_calibration")
    # Training-period (in-sample) fit, reported separately for contrast with validation
    tab_tr, _, _ = evaluate_returns(train, train, cfg, in_sample=True)
    out.table(tab_tr, "train_in_sample_return_models")
    wf_dev = walk_forward(dev, cfg)
    out.table(wf_dev, "predictions_walkforward_dev")
    out.table(walk_forward_metrics(wf_dev), "walkforward_dev_metrics")

    dev_lag2 = None
    if lag2_path.exists():
        e2, _, _ = load_processed(cfg, suffix="_lag2")
        e2 = e2[(e2["report_release_date"] < pd.Timestamp(bounds["test_start"]))]
        dev_lag2 = modeling_sample(e2, "z", "fwd_ret_1d")
    robustness(dev, dev_lag2, cfg, out, dev["split"] == "train", ev)

    # Choose the secondary model by validation MSE (baselines included)
    best = tab_ret.sort_values("mse").iloc[0]["model"]
    spec = {
        "primary_model": "m1_ols_z",
        "primary_feature": cfg["protocol"]["primary_feature"],
        "primary_target": cfg["protocol"]["primary_target"],
        "secondary_model_lowest_validation_mse": best,
        "direction_model": "m3_logit_z",
        "volatility_model": "vol_ols_abs_z",
        "split": bounds,
        "model_features": cfg["protocol"]["model_features"],
        "volatility_features": cfg["protocol"]["volatility_features"],
    }
    h = V.save_frozen(spec, out.tables / "frozen_spec.json")
    log.info("Frozen model specification %s (secondary=%s)", h, best)
    return {"spec_hash": h, "secondary": best}


def run_final(cfg: dict[str, Any], note: str = "") -> dict[str, Any]:
    out = Outputs(cfg)
    frozen = V.load_frozen(out.tables / "frozen_spec.json")
    V.load_frozen(out.tables / "frozen_backtest.json")  # everything must be frozen before the test is touched
    spec = frozen["spec"]
    logbook = V.log_test_evaluation(out.tables / "test_evaluation_log.csv", frozen["spec_hash"], note, "models")
    logbook = logbook[logbook["kind"] == "models"]
    if logbook["spec_hash"].nunique() > 1:
        log.warning("The test set has been evaluated under %d different specifications; this is disclosed in the report.",
                    logbook["spec_hash"].nunique())
    ev, prices, pdates = load_processed(cfg)
    ms = modeling_sample(ev, "z", spec["primary_target"])
    ms["split"] = V.labels_from_bounds(ms["report_release_date"], spec["split"]).to_numpy()
    dev = ms[ms["split"].isin(["train", "validation"])]
    test = ms[ms["split"] == "test"]
    alpha = cfg["protocol"]["hypothesis_test_alpha"]

    tab_ret, preds, _ = evaluate_returns(dev, test, cfg)
    out.table(tab_ret, "test_return_models")
    out.table(preds, "predictions_test")
    tab_dir, calib = evaluate_direction(dev, test, cfg)
    out.table(tab_dir, "test_direction_models")
    out.table(calib, "test_direction_calibration")
    if not calib.empty:
        out.fig(P.calibration_plot(calib, out.figures / "fig17_direction_calibration_test.png",
                                   "Direction model calibration (test)", "Mean predicted P(up)",
                                   "Observed share of up days", "Logistic model on Z, refitted on train+validation; "
                                   "quintile buckets of predicted probability.", pct=True))
    tab_vol, vcal = evaluate_volatility(dev, test, cfg)
    out.table(tab_vol, "test_volatility_models")
    out.table(vcal, "test_volatility_calibration")
    if not vcal.empty:
        out.fig(P.calibration_plot(vcal, out.figures / "fig18_vol_calibration_test.png",
                                   "Volatility model calibration (test)", "Mean predicted 3-day realized vol",
                                   "Mean realized 3-day vol", "Buckets of predicted volatility (quintiles)."))

    wf = walk_forward(ms, cfg)
    out.table(wf, "predictions_walkforward_full")
    out.table(walk_forward_metrics(wf), "walkforward_full_metrics")
    wf_y = wf.assign(year=pd.to_datetime(wf["forecast_date"]).dt.year) if not wf.empty else wf
    out.table(walk_forward_metrics(wf_y, "year"), "walkforward_full_by_year")
    out.table(walk_forward_metrics(wf, "storage_season"), "walkforward_full_by_season")

    event_study(test, prices, cfg, out, "test")
    event_study(ms, prices, cfg, out, "full")

    # Pre-registered evidence rule
    m1_dev = pd.read_csv(out.tables / "m1_dev.csv")
    row = m1_dev[(m1_dev["term"] == "z") & (m1_dev["se_type"] == "Newey-West HAC")].iloc[0]
    m1_test = tab_ret.set_index("model").loc["m1_ols_z"]
    c1 = bool(row["p_value"] < alpha and row["coef"] < 0)
    c2 = bool(m1_test["oos_r2_vs_hist_mean"] > 0)
    c3 = bool(m1_test["clark_west_p"] < alpha)
    vol_dev = pd.read_csv(out.tables / "vol_model_dev.csv").set_index("term")
    vol_test = tab_vol.set_index("model")
    verdict = {
        "criterion_1_dev_beta_negative_and_significant": c1,
        "criterion_2_test_oos_r2_positive": c2,
        "criterion_3_test_clark_west_significant": c3,
        "H0_rejected_for_returns": c1 and c2 and c3,
        "dev_beta_z": float(row["coef"]), "dev_p_hac": float(row["p_value"]),
        "test_oos_r2": float(m1_test["oos_r2_vs_hist_mean"]), "test_clark_west_p": float(m1_test["clark_west_p"]),
        "vol_dev_beta_abs_z": float(vol_dev.loc["abs_z", "coef"]), "vol_dev_p_abs_z": float(vol_dev.loc["abs_z", "p_value"]),
        "vol_test_mae_with_abs_z": float(vol_test.loc["vol_ols_abs_z", "mae"]),
        "vol_test_mae_controls_only": float(vol_test.loc["vol_ols_controls_only", "mae"]),
        "spec_hash": frozen["spec_hash"],
        "test_evaluations_logged": int(len(logbook)),
        "distinct_specs_evaluated_on_test": int(logbook["spec_hash"].nunique()),
    }
    out.json(verdict, "verdict")
    return verdict


def _periods(cfg: dict[str, Any], spec_split: dict[str, str]) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    ev, prices, pdates = load_processed(cfg)
    ev = ev[ev["alignment_ok"] & ev["z"].notna()].copy()
    ev["split"] = V.labels_from_bounds(ev["report_release_date"], spec_split).to_numpy()
    return ev, prices, pdates


def run_backtest_development(cfg: dict[str, Any]) -> dict[str, Any]:
    """Choose k on the validation period only, report train/validation, freeze k."""
    out = Outputs(cfg)
    split = V.load_frozen(out.tables / "frozen_split.json")["spec"]
    ev, prices, pdates = _periods(cfg, split)
    b = cfg["backtest"]
    val = ev[ev["split"] == "validation"]
    k, ktab = BT.select_k(val, prices, pdates, b["k_grid"], b["holding_days"], b["one_way_cost_bps"],
                          b["slippage_bps"], b["min_trades_for_selection"], b["periods_per_year"])
    out.table(ktab, "backtest_k_selection_validation")
    if k is not None:
        rows = []
        for name in ("train", "validation"):
            part = ev[ev["split"] == name]
            r = backtest_period(part, prices, pdates, k, cfg, cfg["project"]["random_seed"])
            rows.append(_summaries_table(r, name))
            if name == "validation":
                out.fig(P.equity_curves(_curves(r), out.figures / "fig16_backtest_validation.png",
                                        f"Economic-signal simulation, validation period (k = {k:g})",
                                        "Spot-price simulation, not an executable strategy. Costs: "
                                        f"{b['one_way_cost_bps']:g} bps + {b['slippage_bps']:g} bps slippage per side; "
                                        f"holding {b['holding_days']} trading day(s)."))
        out.table(pd.concat(rows), "backtest_dev")
    h = V.save_frozen({"k": k, **_bt_params(cfg), "selection_metric": b["selection_metric"],
                       "split": split}, out.tables / "frozen_backtest.json")
    log.info("Frozen backtest specification %s (k=%s)", h, k)
    return {"k": k, "spec_hash": h}


def run_backtest_final(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the frozen threshold rule once on the untouched test period."""
    out = Outputs(cfg)
    frozen = V.load_frozen(out.tables / "frozen_backtest.json")
    spec = frozen["spec"]
    V.log_test_evaluation(out.tables / "test_evaluation_log.csv", frozen["spec_hash"], "", "backtest")
    ev, prices, pdates = _periods(cfg, spec["split"])
    test = ev[ev["split"] == "test"]
    k = spec["k"]
    summary: dict[str, Any] = {"k": k}
    if k is None:
        summary["note"] = "No k met the minimum-trade rule in validation; no strategy was tested."
    else:
        cfg_bt = dict(cfg)
        cfg_bt["backtest"] = {**cfg["backtest"], "holding_days": spec["holding_days"],
                              "one_way_cost_bps": spec["one_way_cost_bps"], "slippage_bps": spec["slippage_bps"]}
        bt = backtest_period(test, prices, pdates, k, cfg_bt, cfg["project"]["random_seed"])
        out.table(_summaries_table(bt, "test"), "backtest_test")
        out.table(bt["_ledger"], "backtest_trades_test")
        out.fig(P.equity_curves(_curves(bt), out.figures / "fig19_backtest_test.png",
                                f"Economic-signal simulation, test period (frozen k = {k:g})",
                                "Spot-price simulation on the untouched test period, not an executable strategy."))
        # Cost sensitivity: same trades, different assumed costs
        sens = []
        for c in (0.0, 5.0, 10.0, 25.0, 50.0):
            led, wk = BT.run_backtest(test, prices, pdates, BT.generate_positions(test["z"], k),
                                      spec["holding_days"], c, spec["slippage_bps"])
            s = BT.performance_summary(wk, led, cfg["backtest"]["periods_per_year"])
            sens.append({"one_way_cost_bps": c, "slippage_bps": spec["slippage_bps"],
                         "mean_net_return_per_week": s["mean_net_return_per_week"], "sharpe_net": s["sharpe_net"],
                         "cumulative_return_net": s["cumulative_return_net"]})
        out.table(pd.DataFrame(sens), "backtest_cost_sensitivity_test")
        summary.update({
            "strategy": bt["strategy"], "always_long": bt["always_long"],
            "random_signal_seed": bt["random_signal_seed"],
            "random_placebo_share_beating_strategy": bt["random_placebo_share_beating_strategy"],
        })
    out.json(summary, "backtest_test_summary")
    return summary
