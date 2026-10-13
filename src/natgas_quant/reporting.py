"""Assemble ``reports/research_report.md`` from computed tables.

The report has fixed methodological text (written before any results) and
result sections that are filled ONLY from files in ``reports/tables``. If a
table does not exist yet, the section says the result is pending; no number
in the report is typed by hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve_path

PENDING = "_Pending: this result has not been computed yet. Run the commands in Section 19._"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4g}"
    return str(v)


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return PENDING
    d = df[[c for c in (cols or df.columns) if c in df.columns]].head(max_rows)
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    more = f"\n\n_{len(df) - max_rows} more rows in the CSV._" if len(df) > max_rows else ""
    return "\n".join([head, sep, *body]) + more


class Tables:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.dir = resolve_path(cfg["paths"]["tables"])
        self.fig_dir = resolve_path(cfg["paths"]["figures"])
        cap = self.dir / "figure_captions.json"
        self.captions = json.loads(cap.read_text()) if cap.exists() else {}

    def csv(self, name: str) -> pd.DataFrame | None:
        p = self.dir / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None

    def json(self, name: str) -> Any:
        p = self.dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def text(self, name: str) -> str | None:
        p = self.dir / name
        return p.read_text() if p.exists() else None

    def table(self, name: str, cols: list[str] | None = None, max_rows: int = 60) -> str:
        return md_table(self.csv(name), cols, max_rows)

    def fig(self, filename: str) -> str:
        p = self.fig_dir / filename
        if not p.exists():
            return ""
        cap = self.captions.get(filename, "")
        return f"![{filename}](figures/{filename})\n\n*Figure: {cap}*\n"


# ---------------------------------------------------------------------------
# Fixed text
# ---------------------------------------------------------------------------
PROTOCOL = """\
This protocol was written before any real data were downloaded or examined. Its
parameters are stored in `configs/research_config.yaml` (block `protocol`).

| Item | Pre-registered choice |
|---|---|
| Primary hypothesis (H1) | A larger positive storage anomaly is associated with a lower subsequent natural-gas return; a larger negative anomaly with a higher subsequent return. |
| Null hypothesis (H0) | After accounting for seasonality and information timing, storage anomalies have no stable relationship with subsequent natural-gas returns. |
| Secondary hypothesis | The absolute anomaly is associated with subsequent realized volatility, even if it does not predict direction. |
| Primary feature | `z`: standardized anomaly, expanding same-time-of-year mean and standard deviation from previous years only (window +/-10 days, at least 5 prior years). |
| Primary target | `fwd_ret_1d`: log return from the first price dated after the release day to the next valid price. |
| Secondary targets | 3- and 5-day log returns, direction of the 1-day return, absolute 1-day return, 3-day realized volatility (`rv_3d`, primary volatility target). |
| Primary test | Univariate OLS `fwd_ret_1d = b0 + b1 z`, Newey-West (HAC) errors, development sample (train+validation). One-sided expectation b1 < 0. |
| Validation | Chronological 60/20/20 split (train / validation / untouched test) plus annual expanding-window forecasts. No random splits. |
| Evaluation metrics | MSE, MAE, out-of-sample R^2 vs historical mean, Clark-West test, Pearson/Spearman correlation (IC), hit rate; classification and volatility metrics as listed in Section 12. |
| Evidence rule for rejecting H0 | All three: (1) development b1 < 0 with HAC p < 0.05; (2) test-period OOS R^2 > 0; (3) test-period Clark-West one-sided p < 0.05. |
| Backtest | Threshold rule on z; k chosen on validation only by mean net return per week (>= 10 trades); 1-day hold; 10 bps cost + 5 bps slippage per side. |
| Significance level | 5%. Many secondary tests are run; they are descriptive and not corrected for multiple comparisons, so isolated "significant" secondary results should be treated with caution. |
"""

RATIONALE = """\
Natural gas is stored underground in summer, when demand is low, and withdrawn
in winter for heating. Each Thursday EIA reports how much working gas was in
storage at the end of the previous week. Because the physical market balances
through storage, a week with more injection (or less withdrawal) than usual
suggests supply was looser or demand weaker than normal, which, other things
equal, should push prices down; the opposite should push prices up. That is
H1.

Two caveats are built into the design. First, traders do not compare the EIA
number with a five-year average; they compare it with their own forecasts
(weather-driven models, analyst surveys). Much of a "seasonal anomaly" is
therefore predictable from weather before the release and may already be in
the price. What we measure is a **deviation from seasonal expectation**, not a
**market surprise**; a surprise requires a pre-release consensus forecast, which
this project does not have. Second, prices react within minutes; daily spot
data cannot capture that, and our conservative timing deliberately excludes the
release day. A weak result here is therefore consistent with an efficient
market, and is a legitimate finding.
"""

TIMING = """\
| Step | Convention | Why |
|---|---|---|
| Storage observation | Week ending Friday (`storage_week_ending_date`) | EIA's survey date. |
| Public release | Thursday 10:30 a.m. ET, six days later | Standard schedule. |
| Holiday weeks | Dates verified from EIA's published schedule are used where available (`data/raw/manual/eia_wngsr_verified_release_dates.csv`). Otherwise, if a federal holiday falls on Tuesday-Thursday of the release week, the **later** possible date (next business day after Thursday) is assumed and the week is flagged `release_date_uncertain`. Late-December weeks are also flagged. | EIA has moved holiday releases both earlier (Wednesday noon) and later (Friday, and once a Monday). Assuming too late only costs a day; assuming too early would leak future information. EIA's 2025-26 schedule lists no change for Monday holidays, so Monday holidays are not treated as shifts (an inference, not a documented rule). |
| Price before release | Last valid daily price dated strictly before the release day | Used only to describe the market reaction; never a trade price. |
| Entry (first eligible price) | First valid daily price dated strictly after the release day | The release-day spot price averages trades from before and after 10:30 a.m., so trading at it after seeing the report would be look-ahead. Robustness: second price after release. |
| Prior-return features | Measured up to the price before the release | Conservative: excludes the mixed release-day price. |
| Forward return | `ln(P_(entry+h) / P_entry)` over h valid prices | Uses only prices the strategy could have traded. |

Whether FRED/EIA date the daily spot price by trade date or by delivery (flow)
date is not stated in EIA's definitions. The absence of weekend-dated rows is
consistent with trade dates, but this is an inference; the two-day entry lag in
the robustness table covers the alternative.
"""

FEATURES = """\
**Weekly change.** `Delta_S_t = S_t - S_(t-1)` (Bcf), computed only when the
previous row is exactly seven days earlier.

**Seasonal expectation.** For week t, the comparable observations are weekly
changes from **earlier years** whose date lies within +/-10 days of the same
calendar day (a circular "time of year" distance, so late December and early
January are neighbours). Observations less than 180 days old are excluded, so
the current week, later weeks, and neighbouring weeks of the current year never
enter their own baseline. At least five distinct earlier years are required.

* Method A (expanding): all earlier years. Mean with sample standard deviation
  (primary), or median with 1.4826 x median absolute deviation (robust).
* Method B (rolling): only the five most recent earlier years (the industry's
  "five-year average" idea). Faster to adapt to structural change, noisier.

**Anomaly and standardized anomaly.** `A_t = Delta_S_t - Expected_t` (Bcf);
`Z_t = A_t / sigma_t`.

*Worked example (also a unit test).* Comparable past changes 70, 80, 90 Bcf ->
mean 80, standard deviation 10. Current change 95 -> A = 15 Bcf, Z = 1.5: an
injection 1.5 typical deviations larger than normal, bearish under H1.

**Log return.** `r_(t,h) = ln(P_(t+h)/P_t)`. Example: 3.00 -> 3.15 gives
ln(1.05) = 0.0488 (4.88%). Log returns add across days.

**Realized volatility.** `rv_h = sqrt(sum of squared daily log returns over the
h days after entry)`. Daily returns +0.02, -0.03, +0.01 give rv_3 = 0.0374.

**Other features (all known before entry).** Month, meteorological season,
ISO week, withdrawal-season indicator (Nov-Mar), sine/cosine of time of year,
prior 1- and 5-day returns and 20-day trailing volatility measured up to the
pre-release price, last week's z (`z_lag1`).
"""

LIMITATIONS = """\
* **Spot price is not directly tradable.** Henry Hub spot is a volume-weighted
  index of physical next-day deals. Backtests are economic-signal simulations.
* **Daily resolution.** The market reacts to the 10:30 a.m. release within
  minutes; daily data cannot measure that and our timing skips the release day.
  Any information absorbed on the release day is invisible here by design.
* **No consensus forecast.** Seasonal anomalies are not market surprises. Much
  of an anomaly is predictable from weather already known before the release.
* **Current-vintage data.** EIA revises storage estimates (including
  reclassifications between base and working gas). Weekly changes computed from
  today's levels can differ from the net change printed on release day. FRED
  serves latest-vintage prices.
* **Release dates before 2025 are rule-based**, not verified week by week.
  Unknown ad-hoc closures or delays (e.g. government shutdowns) could make an
  assumed release date too early; such weeks may be mis-timed.
* **Sample size.** Weekly data since 2010, minus five years of history for the
  baseline, leaves roughly 600 release weeks; extreme-anomaly groups contain
  far fewer. Fat-tailed returns make averages sensitive to a handful of weeks.
* **Structural change.** Shale growth, LNG exports since 2016, the 2021 Texas
  freeze, and the 2022 European crisis changed price dynamics; a relationship
  estimated on one regime may not carry over.
* **Multiple comparisons.** Many secondary tests are reported; some will look
  significant by chance.
"""

EXTENSIONS = """\
* Add a consensus storage forecast (e.g. survey medians) to measure true market
  surprises, then compare with the seasonal anomaly.
* Add NOAA heating/cooling degree days to model the weather-predictable part of
  storage changes and isolate the unpredictable part.
* Use front-month NYMEX futures settlement prices (optionally downloaded by
  `scripts/download_data.py --include-optional`) as a tradable proxy. Futures
  settle at 2:30 p.m. ET, after the release, but returns spanning contract rolls
  must be removed, and settlement-price execution is itself an assumption.
* Build a point-in-time vintage history from ALFRED and archived EIA reports.
* Storage level relative to the five-year range; production, LNG exports,
  sector consumption as controls.
"""


# ---------------------------------------------------------------------------
# Dynamic sections
# ---------------------------------------------------------------------------
def _verdict_text(t: Tables, cfg: dict[str, Any]) -> tuple[str, str, str]:
    """Return (executive summary, negative findings, conclusion) text."""
    b = cfg["backtest"]
    round_trip = 2 * (b["one_way_cost_bps"] + b["slippage_bps"]) / 1e4
    v = t.json("verdict")
    bt = t.json("backtest_test_summary")
    m1 = t.csv("m1_dev")
    rob = t.csv("robustness_dev")
    neg: list[str] = []
    if m1 is not None:
        r = m1[(m1["term"] == "z") & (m1["se_type"] == "Newey-West HAC")].iloc[0]
        if not (r["coef"] < 0 and r["p_value"] < 0.05):
            neg.append(f"Development sample: the z coefficient was {r['coef']:.5f} (HAC p = {r['p_value']:.3f}); "
                       "H1's predicted negative, significant relationship was not found.")
    if rob is not None and "significant_5pct" in rob.columns:
        rr = rob.dropna(subset=["beta"])
        rr = rr[rr["expected_sign"] == "negative"]
        share = rr["sign_as_hypothesized"].astype(bool).mean() if len(rr) else float("nan")
        sig = rr["significant_5pct"].astype(bool).sum()
        neg.append(f"Robustness: {share:.0%} of {len(rr)} return variants had the H1 (negative) sign; {sig} were significant at 5% "
                   "(uncorrected for multiple testing).")
    if v is None:
        exec_txt = ("The development phase may have run, but the untouched test period has not been evaluated, "
                    "so no conclusion is drawn yet. " + (PENDING if m1 is None else ""))
        concl = PENDING
        return exec_txt, ("\n".join(f"* {x}" for x in neg) if neg else PENDING), concl
    if not v["criterion_2_test_oos_r2_positive"]:
        neg.append(f"Test period: out-of-sample R^2 of the primary model was {v['test_oos_r2']:.4f} "
                   "(not better than the historical mean).")
    if not v["criterion_3_test_clark_west_significant"]:
        neg.append(f"Test period: Clark-West p = {v['test_clark_west_p']:.3f}; no significant forecast improvement.")
    if v["vol_test_mae_with_abs_z"] >= v["vol_test_mae_controls_only"]:
        neg.append("Volatility: adding |z| did not reduce test-period MAE relative to trailing-volatility controls.")
    strat = (bt or {}).get("strategy") or {}
    if strat and not (strat.get("mean_net_return_per_week", 0) > 0):
        neg.append("Backtest: the frozen threshold rule did not earn a positive mean net return in the test period.")

    reject = v["H0_rejected_for_returns"]
    exec_txt = (
        f"Under the pre-registered evidence rule, H0 **{'is' if reject else 'is not'} rejected** for returns. "
        f"Development-sample slope on z: {v['dev_beta_z']:.5f} (HAC p = {v['dev_p_hac']:.3f}). "
        f"Test-period out-of-sample R^2 vs the historical mean: {v['test_oos_r2']:.4f}; "
        f"Clark-West p = {v['test_clark_west_p']:.3f}. "
        f"For volatility (hypothesized sign: positive), the development slope on |z| was "
        f"{v['vol_dev_beta_abs_z']:.5f} (HAC p = {v['vol_dev_p_abs_z']:.3f}), controlling for trailing volatility."
    )
    vol_supported = (v["vol_dev_beta_abs_z"] > 0 and v["vol_dev_p_abs_z"] < 0.05
                     and v["vol_test_mae_with_abs_z"] < v["vol_test_mae_controls_only"])
    effect = abs(v["dev_beta_z"])
    if v["dev_beta_z"] >= 0:
        econ = "No: the estimated slope does not even have the hypothesized (negative) sign."
    elif effect < round_trip:
        econ = (f"Not on its own: a one-standard-deviation anomaly shifts the expected next-day return by "
                f"{effect:.4f}, below the assumed round-trip cost of {round_trip:.4f}; only anomalies beyond "
                f"about {round_trip / effect:.1f} standard deviations would cover costs on average.")
    else:
        econ = (f"Potentially: a one-standard-deviation anomaly shifts the expected next-day return by "
                f"{effect:.4f}, above the assumed round-trip cost of {round_trip:.4f}. Statistical and "
                "out-of-sample evidence above determines whether this is believable.")
    survive = None
    if strat:
        placebo = bt.get("random_placebo_share_beating_strategy")
        survive = strat.get("mean_net_return_per_week", 0) > 0 and placebo is not None and placebo < 0.05
        exec_txt += (f" The spot-price simulation earned a mean net return of "
                     f"{strat['mean_net_return_per_week']:.5f} per week in the test period "
                     f"({placebo:.0%} of random-signal placebos did at least as well).")
    wf = t.csv("walkforward_full_by_year")
    stable = "not computed"
    if wf is not None and not wf.empty:
        w = wf[wf["model"] == "m1_ols_z"]
        stable = f"{(w['oos_r2_vs_hist_mean'] > 0).mean():.0%} of {len(w)} walk-forward years had positive OOS R^2"
    concl = "\n".join([
        "_These answers are generated mechanically from the pre-registered rules and the tables above; "
        "they should be read together with the limitations._",
        "",
        f"* **Did storage anomalies contain statistically reliable information about returns?** "
        f"{'Yes, by the pre-registered rule.' if reject else 'No, not by the pre-registered rule.'}",
        f"* **Was the relationship stable out of sample?** Test OOS R^2 = {v['test_oos_r2']:.4f}; {stable}.",
        f"* **Was it economically meaningful?** {econ}",
        f"* **Did it survive assumed costs?** "
        + ("Not tested (no threshold met the minimum-trade rule)." if survive is None
           else ("Yes, in this simulation." if survive else "No.")),
        f"* **Volatility (secondary hypothesis):** {'supported' if vol_supported else 'not supported'} "
        f"(requires a positive, significant development slope on |z| AND lower test MAE than controls alone: "
        f"{v['vol_test_mae_with_abs_z']:.5f} with |z| vs {v['vol_test_mae_controls_only']:.5f} without).",
        "* **Supported:** only the statements above that passed their tests. "
        "**Not supported:** any claim of causality, of a market 'surprise' effect, or of an executable profitable "
        "strategy; spot-price simulations do not establish tradable profits.",
        f"* Test-set evaluations logged: {v['test_evaluations_logged']} "
        f"({v['distinct_specs_evaluated_on_test']} distinct model specification(s)).",
    ])
    return exec_txt, "\n".join(f"* {x}" for x in neg) if neg else "* None recorded.", concl


def write_report(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    t = Tables(cfg)
    exec_txt, neg_txt, concl = _verdict_text(t, cfg)
    split = t.json("split_summary")
    manifest_p = resolve_path(cfg["paths"]["processed"]) / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else None
    verified = pd.read_csv(resolve_path(cfg["release"]["verified_release_file"]))

    S: list[str] = []
    a = S.append
    a("# From Physical Fundamentals to Prices: Do U.S. Natural-Gas Storage Anomalies Predict Henry Hub Returns or Volatility?\n")
    a("_Educational research project. Not financial advice. Generated by `natgas_quant.reporting`; "
      "every number below comes from `reports/tables`._\n")

    a("## 1. Executive summary\n")
    a(exec_txt + "\n")

    a("## Pre-registered research protocol\n")
    a(PROTOCOL)
    a("\n**Protocol deviations:** none recorded. (Any change to a primary setting after seeing results must be "
      "listed here with its reason.)\n")
    log = t.csv("test_evaluation_log")
    a("\n**Test-set evaluation log:**\n\n" + (md_table(log) if log is not None else "_The test set has not been evaluated._") + "\n")

    a("## 2. Research question\n")
    a("Do unusually large weekly changes in U.S. working natural-gas storage contain statistically significant "
      "and out-of-sample information about subsequent Henry Hub natural-gas returns or volatility?\n")

    a("## 3. Economic and physical rationale\n")
    a(RATIONALE)

    a("## 4. Data sources\n")
    a("| Dataset | Source | Identifier | Units | Frequency |\n|---|---|---|---|---|\n"
      f"| Working gas in storage, Lower 48 | EIA API v2, route `{cfg['sources']['eia_storage']['route']}` | "
      f"`{cfg['sources']['eia_storage']['series_id']}` | billion cubic feet | weekly (week ending Friday) |\n"
      f"| Henry Hub spot price | FRED (republishes EIA `RNGWHHD`) | `{cfg['sources']['fred_henry_hub']['series_id']}` | "
      "US$ per million Btu | daily (business days) |\n"
      "| Release schedule exceptions | EIA, https://ir.eia.gov/ngs/schedule.html | manual file | dates | holiday weeks 2025-26 |\n")
    if manifest:
        rows = []
        for name, info in manifest["inputs"].items():
            m = info["metadata"] if isinstance(info["metadata"], dict) else {}
            rows.append({"file": name, "downloaded_at_utc": m.get("downloaded_at_utc"), "first": m.get("first_period"),
                         "last": m.get("last_period"), "rows": m.get("row_count"), "sha256": info["sha256"][:12]})
        a("\nDownloaded inputs used for this report:\n\n" + md_table(pd.DataFrame(rows)) + "\n")
    else:
        a("\n" + PENDING + "\n")
    val_md = t.text("data_validation.md")
    a("\n### Validation checks\n\n" + (val_md or PENDING) + "\n")

    a("## 5. Data dictionary\n")
    a("See `data/README.md` for every column, its unit, and the time at which it is known.\n")

    a("## 6. Timestamp and release-date methodology\n")
    a(TIMING)
    a("\nVerified release dates on file:\n\n" + md_table(verified[["storage_week_ending_date", "report_release_date",
                                                                    "report_release_time_et", "reason"]]) + "\n")
    al = t.csv("release_alignment_table")
    if al is not None:
        src = al["release_date_source"].value_counts().rename_axis("release_date_source").reset_index(name="weeks")
        a("\nRelease-date sources across the sample:\n\n" + md_table(src) + "\n")
        hol = al[al["release_date_source"] != "rule_nominal"].tail(8)
        a("\nExample holiday / verified weeks (full table: `reports/tables/release_alignment_table.csv`):\n\n"
          + md_table(hol) + "\n")

    a("## 7. Mathematical feature definitions\n")
    a(FEATURES)

    a("## 8. Exploratory analysis\n")
    a("Relationships between features and outcomes use the **development sample only** (train + validation); "
      "the test period stays unseen until the final phase.\n")
    a("\n### Coverage\n\n" + t.table("audit_coverage") + "\n")
    a("\n### Missing values\n\n" + t.table("audit_missing") + "\n")
    a("\n### Summary statistics (development sample)\n\n" + t.table("summary_stats_dev") + "\n")
    for f in ("fig01_storage_level.png", "fig02_weekly_change.png", "fig03_seasonal_change.png",
              "fig04_henry_hub_price.png", "fig05_return_distribution.png", "fig06_anomaly_distribution.png",
              "fig07_scatter_z_return.png", "fig08_quintile_returns.png", "fig09_absz_quintile_vol.png",
              "fig10_correlation.png", "fig11_sample_counts.png"):
        a(t.fig(f))
    a("\n### Quintile table: 1-day returns by z quintile\n\n" + t.table("eda_quintile_returns") + "\n")
    a("\n### Quintile table: 3-day realized volatility by |z| quintile\n\n" + t.table("eda_absz_quintile_rv3d") + "\n")
    a("\n### Flags\n\n" + t.table("eda_flags") + "\n")

    a("## 9. Event-study results\n")
    a("Development sample, fixed z thresholds (-1.5, -0.5, +0.5, +1.5). `boot_ci_*` are 95% moving-block "
      "bootstrap intervals; `small_sample` marks groups with fewer than 30 weeks.\n")
    a("\n" + t.table("event_groups_dev") + "\n")
    a(t.fig("fig12_event_groups_dev.png"))
    a(t.fig("fig13_car_paths_dev.png"))
    a("\nLarge-negative minus large-positive mean return (H1 predicts > 0):\n\n" + t.table("event_extreme_difference_dev") + "\n")
    a("\nBy storage season:\n\n" + t.table("event_groups_by_season_dev") + "\n")
    a("\nBy subperiod:\n\n" + t.table("event_groups_by_subperiod_dev") + "\n")
    if t.csv("event_groups_test") is not None:
        a("\nTest period:\n\n" + t.table("event_groups_test") + "\n" + t.fig("fig12_event_groups_test.png"))
        a("\nFull sample (descriptive):\n\n" + t.table("event_groups_full") + "\n")

    a("## 10. Statistical models\n")
    a("**Model 1 (univariate OLS), development sample.** Newey-West errors are the basis for inference; classical "
      "errors are shown only to illustrate how much they can differ.\n\n" + t.table("m1_dev") + "\n")
    info = t.json("m1_dev_info")
    if info:
        a(f"\nn = {info['nobs']}, R^2 = {_fmt(info['r_squared'])}, adjusted R^2 = {_fmt(info['adj_r_squared'])}, "
          f"HAC lags = {info['hac_lags']}.\n")
    a("\n**Model 2 (multivariate OLS).**\n\n" + t.table("m2_dev") + "\n")
    info2 = t.json("m2_dev_info")
    if info2:
        a(f"\nn = {info2['nobs']}, R^2 = {_fmt(info2['r_squared'])}, adjusted R^2 = {_fmt(info2['adj_r_squared'])}.\n")
    a("\n**Residual diagnostics** (Ljung-Box: autocorrelation; Breusch-Pagan: heteroskedasticity; Jarque-Bera: "
      "normality; Cook's distance: influential weeks).\n\n" + t.table("diagnostics_dev") + "\n")
    a("\n**Variance inflation factors (Model 2).**\n\n" + t.table("vif_dev") + "\n")
    infl = t.json("m1_influence_dev")
    if infl:
        a(f"\n**Influence check.** Dropping {infl['n_dropped']} high-influence weeks changes the z slope from "
          f"{_fmt(infl['beta_full'])} to {_fmt(infl['beta_trimmed'])} (p = {_fmt(infl['p_trimmed'])}).\n")
    a(t.fig("fig14_m1_residuals.png"))
    a(t.fig("fig15_m1_stability.png"))
    a("\n**Secondary hypothesis: |z| and 3-day realized volatility** (controls: trailing volatility, season).\n\n"
      + t.table("vol_model_dev") + "\n")

    a("## 11. Validation design\n")
    if split:
        a(md_table(pd.DataFrame([split])) + "\n")
    else:
        a(PENDING + "\n")
    a("\nModels are fitted on train and compared on validation; hyperparameters for ridge, lasso and gradient "
      "boosting are tuned with `TimeSeriesSplit` inside the training rows only, with scaling inside the pipeline. "
      "The test period is evaluated once, after `frozen_spec.json` and `frozen_backtest.json` are written.\n")

    a("## 12. Out-of-sample results\n")
    a("\n**Training period (in-sample fit, for contrast).**\n\n" + t.table("train_in_sample_return_models") + "\n")
    a("\n**Validation period, return models.** Benchmark for OOS R^2 is the training-period mean.\n\n"
      + t.table("validation_return_models") + "\n")
    a("\n**Validation period, direction models.**\n\n" + t.table("validation_direction_models") + "\n")
    a("\n**Validation period, volatility models (target rv_3d).**\n\n" + t.table("validation_volatility_models") + "\n")
    a("\n**Permutation importance (validation).** Increase in MSE when a feature is shuffled; not evidence of "
      "causality.\n\n" + t.table("permutation_importance_validation") + "\n")
    a("\n**Annual walk-forward, development years.**\n\n" + t.table("walkforward_dev_metrics") + "\n")
    if t.csv("test_return_models") is not None:
        a("\n**Test period (untouched until the final phase), return models.**\n\n" + t.table("test_return_models") + "\n")
        a("\n**Test period, direction models.**\n\n" + t.table("test_direction_models") + "\n")
        a(t.fig("fig17_direction_calibration_test.png"))
        a("\n**Test period, volatility models.**\n\n" + t.table("test_volatility_models") + "\n")
        a("\nVolatility calibration (test):\n\n" + t.table("test_volatility_calibration") + "\n")
        a(t.fig("fig18_vol_calibration_test.png"))
        a("\n**Annual walk-forward, full sample.**\n\n" + t.table("walkforward_full_metrics") + "\n")
        wfy = t.csv("walkforward_full_by_year")
        a("\nBy year, primary model (all models in the CSV):\n\n" + md_table(
            wfy[wfy["model"] == "m1_ols_z"] if wfy is not None and not wfy.empty else None) + "\n")
        a("\nBy storage season:\n\n" + t.table("walkforward_full_by_season") + "\n")
    else:
        a("\n**Test period:** " + PENDING + "\n")

    a("## 13. Backtest methodology\n")
    b = cfg["backtest"]
    a(f"Rule: long if z <= -k, short if z >= +k, flat otherwise. Entry at the first eligible post-release price; "
      f"exit {b['holding_days']} valid price(s) later; no overlapping positions. Costs per side: "
      f"{b['one_way_cost_bps']:g} bps commission-like cost + {b['slippage_bps']:g} bps slippage (both configurable). "
      f"k is chosen from {b['k_grid']} on the validation period only. Sharpe ratio = mean/std of weekly event "
      f"returns (0 when flat) x sqrt({b['periods_per_year']}), zero risk-free rate. Benchmarks: no trade, always "
      f"long, a fixed-seed random signal with the same activity level, and {b['random_benchmark_draws']} random "
      "placebo draws.\n\n**This is an economic-signal simulation on the Henry Hub spot index, which is not directly "
      "tradable. It is not an executable strategy.** A futures or ETF implementation would face roll yield, "
      "contract expiry, fees, tracking error and different timing.\n")

    a("## 14. Backtest results\n")
    a("\nThreshold selection (validation only):\n\n" + t.table("backtest_k_selection_validation") + "\n")
    a("\nTrain and validation periods:\n\n" + t.table("backtest_dev") + "\n")
    a(t.fig("fig16_backtest_validation.png"))
    if t.csv("backtest_test") is not None:
        a("\nTest period (frozen k):\n\n" + t.table("backtest_test") + "\n")
        a(t.fig("fig19_backtest_test.png"))
        a("\nCost sensitivity (same trades):\n\n" + t.table("backtest_cost_sensitivity_test") + "\n")
        bt = t.json("backtest_test_summary") or {}
        if "random_placebo_share_beating_strategy" in bt:
            a(f"\nShare of random placebo strategies at least as good as the rule: "
              f"{bt['random_placebo_share_beating_strategy']:.1%}.\n")
    else:
        a("\n**Test period:** " + PENDING + "\n")

    a("## 15. Robustness checks\n")
    a("Every variant is reported, not only the favourable ones. Development sample; HAC errors.\n\n"
      + t.table("robustness_dev") + "\n")

    a("## 16. Limitations\n")
    a(LIMITATIONS)

    a("## 17. Negative findings\n")
    a(neg_txt + "\n")

    a("## 18. Potential extensions\n")
    a(EXTENSIONS)

    a("## 19. Reproducibility instructions\n")
    a("```bash\npython -m venv .venv && source .venv/bin/activate\npip install -e \".[dev]\"\n"
      "git init && git add -A && git commit -m \"Pre-registered protocol, before data\"\n"
      "cp .env.example .env            # then put your EIA key in .env\npytest\n"
      "python scripts/download_data.py\npython scripts/build_dataset.py\n"
      "python scripts/run_analysis.py --phase development\npython scripts/run_backtest.py --phase development\n"
      "python scripts/run_analysis.py --phase final --note \"first test evaluation\"\n"
      "python scripts/run_backtest.py --phase final\n```\n")

    a("## 20. Conclusion\n")
    a(concl + "\n")

    path = resolve_path(cfg["paths"]["report"])
    path.write_text("\n".join(S), encoding="utf-8")
    return path
