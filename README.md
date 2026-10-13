# From Physical Fundamentals to Prices

**Do U.S. natural-gas storage anomalies predict Henry Hub returns or volatility?**

An educational, reproducible quantitative-research pipeline. It is **not financial advice**, and a negative
result is an acceptable, reportable outcome.

> **Status:** complete. Run on EIA and FRED data through October 2026, with a single pre-registered
> evaluation of the untouched test period (June 2024 – October 2026). Full write-up:
> [`reports/research_report.md`](reports/research_report.md).



## 1. Project overview

**What is natural-gas storage?** The U.S. produces gas at a fairly steady rate, but uses far more in winter
(heating) than in summer. The gap is bridged by pumping gas into underground storage in spring and summer
(*injections*) and drawing it out in winter (*withdrawals*). Every Thursday at 10:30 a.m. Eastern, the
U.S. Energy Information Administration (EIA) publishes how much *working gas* (gas that can be withdrawn) was in
storage at the end of the previous week, in **billion cubic feet (Bcf)**.

**What is Henry Hub?** A pipeline hub in Louisiana whose daily cash ("spot") price, in **US dollars per million
British thermal units ($/MMBtu)**, is the main U.S. natural-gas benchmark.

**The idea being tested.** If storage rises by much more than is normal for that time of year, the market is
probably looser than expected (more supply or less demand), which should push prices down. If it rises by much
less than normal, prices should go up. We call the difference from normal a **storage anomaly**. We test whether
anomalies are followed by price moves that someone could have exploited *after* the report was public.

**Why this is harder than it sounds.**

* The price reacts within minutes of 10:30 a.m.; we only have **daily** prices, and we refuse to pretend we could
  trade at a price that was set before the report.
* Traders compare the number with *their own forecasts*, not with a seasonal average. Our anomaly is a
  **deviation from seasonal expectation**, not a **market surprise** (that would need a pre-release consensus
  forecast, which we don't have).
* With about 600 usable weeks and noisy daily returns, chance patterns are easy to find. The design guards
  against fooling ourselves.

### Key terms

| Term | Meaning |
|---|---|
| Log return | `ln(P_later / P_earlier)`. 3.00 to 3.15 is ln(1.05) = 0.0488, about +4.9%. Log returns add up over days. |
| Look-ahead bias | Using information in a historical test that was not available at the time. The most common way a backtest lies. |
| Point-in-time | A value computed only from information available on that date. |
| Seasonal expectation | What a weekly storage change "normally" is at that time of year, estimated from earlier years only. |
| Standardized anomaly (Z) | (actual change - expected change) / typical deviation. Z = 2 means "two typical deviations above normal". |
| Realized volatility | Square root of the sum of squared daily log returns over a window; how much the price moved, regardless of direction. |
| OLS regression | Fits a straight line `return = b0 + b1 * Z`. b1 is the slope. |
| Standard error, p-value | How uncertain the slope is; the p-value is the chance of a slope at least this far from zero if the true slope were zero. |
| Newey-West (HAC) errors | Standard errors that stay valid when returns have changing volatility and are correlated over time. |
| Out-of-sample (OOS) | Data the model did not see while being fitted. The only honest test of prediction. |
| OOS R² | 1 - (model squared errors / benchmark squared errors). Above 0 means the model beat the benchmark. |
| Backtest | A simulation of what a trading rule would have done historically, including costs. |
| Basis point (bp) | 0.01%. A 10 bp cost on $10,000 is $10. |
| Sharpe ratio | Average return divided by its volatility, scaled to a year. Higher means steadier. |
| Drawdown | Fall from a previous peak in account value. |

---

## 2. Research question and pre-registered protocol

**Research question.** Do unusually large weekly changes in U.S. working natural-gas storage contain
statistically significant and out-of-sample information about subsequent Henry Hub natural-gas returns or
volatility?

This protocol was written **before any real data were examined**. Its parameters live in
`configs/research_config.yaml`.

| Item | Choice |
|---|---|
| **H1 (primary)** | A larger positive storage anomaly is associated with a lower subsequent return; a larger negative anomaly with a higher subsequent return. |
| **H0 (null)** | After accounting for seasonality and information timing, storage anomalies have no stable relationship with subsequent returns. |
| **Secondary hypothesis** | The absolute anomaly is associated with subsequent realized volatility, even if it does not predict direction. |
| Primary feature | `z`: anomaly standardized by the mean and standard deviation of same-time-of-year changes in **earlier years only** (+/-10 days, at least 5 years). |
| Primary target | `fwd_ret_1d`: log return from the first price dated **after** the release day to the next price. |
| Secondary targets | 3- and 5-day returns, direction, absolute return, 3-day realized volatility (`rv_3d`). |
| Validation | Chronological 60% train / 20% validation / 20% untouched test, plus year-by-year expanding-window forecasts. Never a random split. |
| Metrics | MSE, MAE, OOS R² vs historical mean, Clark-West test, Pearson/Spearman correlation, hit rate; accuracy, balanced accuracy, ROC-AUC, log loss, Brier score; volatility MAE/RMSE/correlation/calibration. |
| **Decision rule for rejecting H0** | All three: (1) slope on z negative with HAC p < 0.05 in train+validation; (2) test OOS R² > 0; (3) test Clark-West p < 0.05. |
| Backtest | Long if z <= -k, short if z >= +k; k chosen on validation only; 1-day hold; 10 bp cost + 5 bp slippage per side. |

**Assumptions** (each chosen to be conservative): entry only at a price dated strictly after the release day;
holiday weeks assume the *later* possible release; prior-return features stop at the last pre-release price;
seasonal baselines never include the current or any later week.

**Known data limitations:** current-vintage (revised) data; release dates before 2025 are rule-based, not
verified week by week; spot prices are not directly tradable; daily resolution; no consensus forecast; weekly
storage data begin in 2010 so the usable sample starts around 2015. Details in the report, Section 16.

---

## 3. Data sources

| Dataset | Source | Identifier | Units | Frequency |
|---|---|---|---|---|
| Working gas in underground storage, Total Lower 48 | EIA Open Data API v2, route `natural-gas/stor/wkly` | `NW2_EPG0_SWO_R48_BCF` (verified against the live facet listing at download time) | Bcf | weekly, week ending Friday, since 2010 |
| Henry Hub natural-gas spot price | FRED (republishes EIA `RNGWHHD`, sourced from Refinitiv/LSEG) | `DHHNGSP` | $/MMBtu | daily, business days, since 1997 |
| Storage-report holiday schedule | EIA, <https://ir.eia.gov/ngs/schedule.html> | `data/raw/manual/eia_wngsr_verified_release_dates.csv` | dates | 2025-2026 exceptions |

Every download records its URL, query (API key redacted), series, units, frequency, timestamp, date range, row
count and SHA-256 hash in a `.meta.json` file next to the raw file. See `data/README.md` for the data dictionary.

---

## 4. Repository structure

```
natural-gas-quant-research/
├── configs/research_config.yaml   every tunable choice + the pre-registered protocol
├── data/
│   ├── raw/                       untouched downloads (+ .meta.json); raw/manual = hand-verified release dates
│   ├── interim/                   parsed and validated tables
│   ├── processed/                 one row per storage release (events.csv), price series, manifest
│   └── README.md                  data dictionary
├── notebooks/                     01 data audit, 02 exploratory analysis, 03 model review (read results; no logic)
├── src/natgas_quant/              the library (pipeline order):
│   ├── config.py                  load YAML, resolve paths
│   ├── data_download.py           retries, API error checks, caching, vintage archive
│   ├── data_validation.py         parsing, units, duplicates, gaps, revisions
│   ├── release_calendar.py        week-ending date -> release date -> first tradable price
│   ├── features.py                weekly change, seasonal expectation, anomaly, z, price features
│   ├── targets.py                 forward returns, direction, realized volatility
│   ├── dataset.py                 assembles the event table, with a hard look-ahead guard
│   ├── event_study.py             grouped returns with block-bootstrap intervals
│   ├── models.py                  OLS + Newey-West, diagnostics, baselines, logistic, ridge/lasso, boosting
│   ├── validation.py              chronological splits, walk-forward, frozen specifications, test log
│   ├── metrics.py                 forecast, classification, trading metrics, bootstrap, Clark-West
│   ├── backtest.py                threshold strategy, costs, benchmarks
│   ├── plotting.py                figures
│   ├── analysis.py                runs Stages 5-11 in a development phase and a final phase
│   └── reporting.py               writes reports/research_report.md from computed tables only
├── tests/                         pytest suite (synthetic, hand-checkable data only)
├── reports/                       figures/, tables/, research_report.md
└── scripts/                       download_data, build_dataset, run_analysis, run_backtest
```

---

## 5. Installation

Requires Python 3.11 or newer.

```bash
git clone <your-repo-url> natural-gas-quant-research   # or unzip the delivered archive
cd natural-gas-quant-research
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Environment variables

```bash
cp .env.example .env
# edit .env and set EIA_API_KEY=<your key>   (free: https://www.eia.gov/opendata/register.php)
```

`.env` is in `.gitignore`; keys are read with `python-dotenv` and never written to disk or logs. FRED's CSV
endpoint needs no key.

---

## 6. Running the project

Run these **in order**. The order is part of the method: the test period stays untouched until everything
is frozen.

```bash
git init && git add -A && git commit -m "Pre-registered protocol, before data"   # 0. timestamp the protocol
pytest                                                    # 1. all tests should pass
python scripts/download_data.py                           # 2. raw data (cached; --force to refresh)
python scripts/build_dataset.py                           # 3. validate + build point-in-time event table
python scripts/run_analysis.py --phase development        # 4. audit, EDA, event study, models (train+validation only)
python scripts/run_backtest.py --phase development        # 5. choose k on validation, freeze it
python scripts/run_analysis.py --phase final --note "first test evaluation"   # 6. ONE look at the test period
python scripts/run_backtest.py --phase final              # 7. frozen rule on the test period
```

After step 3, read the printed sample weeks: the `first_eligible_price_date` must always be after the
`report_release_date`. After step 4, read `reports/research_report.md` and the notebooks **before** step 6.
If you change anything after step 6, record it under "Protocol deviations" in the report; the test-evaluation
log will show it anyway.

`python scripts/run_analysis.py --phase report` rebuilds the report from existing tables.

---

## 7. Main results

**Storage anomalies did not reliably predict Henry Hub returns or volatility.**

* **Primary test (2015–2024 development data):** slope on the standardized anomaly +0.0013,
  Newey–West p = 0.71, R² = 0.0002. None of 23 robustness variants was significant.
* **Untouched test period (2024–2026):** a small out-of-sample gain appeared, but it came from the
  *opposite* sign to the hypothesis and disappeared when five extreme weeks were removed.
  Under the pre-registered rule, H0 is not rejected.
* **Volatility:** the size of the anomaly added nothing beyond recent volatility and season.
* **Backtest (spot-price simulation, costs included):** −38.5% over 23 trades; 80% of random-signal
  placebos did at least as well.
* **Exploratory lead:** short-term price reversal (prior 5-day return) improved forecasts out of
  sample (walk-forward R² +3.8%), unrelated to storage. It needs its own pre-registered study.

A negative result is consistent with markets pricing storage news before or within minutes of the
10:30 a.m. release, which daily data cannot capture.
---

## 8. How this was verified (synthetic data only)

Because EIA and FRED were unreachable from the build environment, the full pipeline was exercised on **synthetic
data written in the exact EIA/FRED file formats** (never committed, never used for conclusions):

* **Null scenario** (prices are a random walk unrelated to storage): slope on z = -0.0008, p = 0.54; test OOS
  R² < 0; H0 not rejected; the strategy lost money after costs. The pipeline does not invent signal.
* **Planted scenario** (next-day return shifted by -0.02 x z): estimated slope -0.0208; H0 rejected; event
  groups fall monotonically. The pipeline finds a real effect with the right sign and size.
* **Mutation tests**: deliberately introducing look-ahead bugs (entering on the release day, letting the current
  week into its own baseline, measuring prior returns after the release, charging costs one-way) each made the
  test suite fail.

---

## 9. Important limitations

* Henry Hub spot is **not directly tradable**; the backtest is an economic-signal simulation.
* Daily data cannot capture the minutes-scale reaction to the 10:30 a.m. release; we skip the release day.
* Seasonal anomalies are **not market surprises**; much of a storage change is predictable from weather.
* Data are **current vintage**; EIA revises storage, so today's numbers can differ from what traders saw.
* Release dates before 2025 are inferred by rule; unknown delays could mis-time some weeks.
* About 600 usable weeks; extreme-anomaly groups are small; returns are fat-tailed.
* Many secondary tests are reported without multiple-comparison correction.

## 10. Common beginner mistakes this project avoids

1. Aligning prices to the storage **week-ending** date (6 days before anyone knew the number).
2. Trading at the **release-day** daily price, which partly predates the report.
3. Computing the seasonal "normal" with the **whole sample** (including future years).
4. **Random** train/test splits on time series.
5. Fitting the **scaler** or tuning hyperparameters on data that includes the test period.
6. Looking at the test set, tweaking, and looking again.
7. Reporting only the best-looking specification.
8. Ignoring transaction costs, or confusing statistical with economic significance.

---

**Disclaimer.** Educational project. Nothing here is investment advice or a recommendation to trade. Historical
performance, especially in a simulation on a non-tradable index, says nothing reliable about future results.
