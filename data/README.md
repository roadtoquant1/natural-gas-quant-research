# Data dictionary

All dates are calendar dates without time zone unless stated. "Known at" says when a value could first have been
used in a decision; features must be known **before entry**, targets are realized **after entry**.

## Raw files (`data/raw/`, never edited)

| File | Created by | Content |
|---|---|---|
| `eia_storage_weekly.json` (+ `.meta.json`) | `download_data.py` | Untouched EIA API v2 response pages for series `NW2_EPG0_SWO_R48_BCF` |
| `eia_metadata/natural-gas_stor_wkly_route.json` | `download_data.py` | Route metadata (facets, frequencies, columns) used to verify the series |
| `eia_metadata/natural-gas_stor_wkly_facet_series.json` | `download_data.py` | Every series value the route offers; our series must appear here |
| `fred_DHHNGSP.csv` (+ `.meta.json`) | `download_data.py` | FRED CSV exactly as served |
| `archive/<UTC timestamp>/` | `download_data.py --force` | Previous vintages, kept to measure revisions |
| `manual/eia_wngsr_verified_release_dates.csv` | hand-entered from EIA's schedule page | Holiday-week release dates and times (committed to git) |

The `.meta.json` sidecars record: source, URL, query (API key redacted), series id, units, frequency,
`downloaded_at_utc`, first/last period, row count, SHA-256 of the raw file, and a vintage note.

## Source series

| Series | Definition | Units | Frequency | Geography | Notes |
|---|---|---|---|---|---|
| `NW2_EPG0_SWO_R48_BCF` | Weekly working gas in underground storage | billion cubic feet (Bcf) | weekly, as of Friday | Total Lower 48 States | Storage **level**; weekly change is computed here. EIA weekly history in this series starts in 2010. Values are current vintage. |
| `DHHNGSP` | Henry Hub natural-gas spot price | US$ per million Btu | daily (business days) | Henry Hub, Louisiana | FRED republication of EIA `RNGWHHD` (Refinitiv/LSEG). Missing values (`.` or empty) are holidays/no report. Trade vs flow dating is not documented; see report Section 6. |

## Processed event table (`data/processed/events.csv`, one row per storage week)

| Column | Meaning | Unit | Known at |
|---|---|---|---|
| `storage_week_ending_date` | Friday the storage level refers to | date | release |
| `nominal_release_date` | week-ending + 6 days (Thursday) | date | - |
| `report_release_date` | release date used (verified or conservative rule) | date | - |
| `report_release_timestamp_et` | date + time; "(time assumed)" if not the standard 10:30 or verified | text | - |
| `release_date_source` | `verified_schedule`, `rule_nominal`, `rule_holiday_assume_later` | text | - |
| `release_date_uncertain` | True for rule-shifted holiday weeks and late-December weeks | bool | - |
| `storage_level_bcf` | S_t | Bcf | release |
| `storage_change_bcf` | S_t - S_(t-1), only if the previous week is 7 days earlier | Bcf | release |
| `price_before_release_date`, `price_before_release` | last valid price dated before the release day | date, $/MMBtu | before release |
| `first_eligible_price_date`, `price_after_release` | first valid price dated after the release day (entry) | date, $/MMBtu | entry |
| `alignment_ok` | both prices exist within 7 calendar days of the release | bool | - |
| `entry_gap_days` | calendar days from release to entry | days | - |
| `expected_<method>`, `scale_<method>`, `n_years_<method>` | seasonal expectation, its scale, and years of history used | Bcf, Bcf, count | release |
| `anomaly_<method>`, `z_<method>` | anomaly and standardized anomaly for each method (`expanding_mean`, `expanding_median`, `rolling_mean`, `rolling_median`) | Bcf, unitless | release |
| `expected_change_bcf`, `anomaly_bcf`, `z`, `abs_z` | primary-method (expanding mean) versions | Bcf, unitless | release |
| `z_lag1` | previous week's z | unitless | release |
| `prior_ret_1d`, `prior_ret_5d` | log returns ending at the pre-release price | log return | before release |
| `trailing_vol_20d` | std. dev. of the 20 daily log returns ending at the pre-release price | daily log units | before release |
| `month`, `season`, `week_of_year`, `year` | calendar of the release / week-ending date | - | release |
| `withdrawal_season`, `storage_season` | 1 / "withdrawal (Nov-Mar)" vs 0 / "injection (Apr-Oct)" | - | release |
| `woy_sin`, `woy_cos` | time of year on a circle | - | release |
| `reaction_ret` | ln(price_after / price_before); spans the release; **diagnostic only, not tradable** | log return | entry |
| `exit_date_{h}d` | h-th valid price after entry (h = 1, 3, 5) | date | target |
| `fwd_ret_{h}d` | ln(P_(entry+h) / P_entry) | log return | target |
| `dir_{h}d` | 1 if fwd return > 0 else 0 | 0/1 | target |
| `abs_ret_{h}d` | absolute forward return | log return | target |
| `rv_{h}d` | sqrt(sum of squared daily log returns over the h days after entry) | log units | target |
| `overlaps_next_release_{h}d` | holding window reaches the next week's release (always for 5-day) | bool | - |
| `entry_date_duplicated` | two releases share an entry date (holiday clusters) | bool | - |

`events_lag2.csv` has the same columns with entry on the **second** valid price after the release
(robustness). `prices_valid.csv` holds the non-missing prices whose positions the event table references;
`prices_all.csv` keeps missing days; `manifest.json` stores the SHA-256 of the raw inputs used.

## Vintage warning

EIA revises storage estimates, and the API returns only the latest values. Weekly changes computed from
today's levels can differ from the "net change" market participants saw on release day (which also included
reclassifications between base and working gas). Re-downloading with `--force` archives the previous file and
`build_dataset.py` reports any revised weeks in `reports/tables/storage_revisions_vs_previous_vintage.csv`.
