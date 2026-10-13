"""natgas_quant: an educational research pipeline asking whether U.S. natural-gas
storage anomalies carry information about subsequent Henry Hub returns or
volatility.

Module map (in pipeline order):

- config            load configs/research_config.yaml and resolve paths
- data_download     fetch raw EIA / FRED data, save untouched copies + metadata
- data_validation   parse raw files, check types, gaps, duplicates, units
- release_calendar  map each storage week to its public release date and to
                    the first price that could be traded after the release
- features          weekly change, seasonal expectation, anomaly, z-score,
                    calendar and prior-price features (all point-in-time)
- targets           forward log returns, direction, realized volatility
- event_study       grouped forward returns with block-bootstrap intervals
- models            OLS with Newey-West errors, diagnostics, baselines,
                    logistic, ridge/lasso, shallow gradient boosting
- validation        chronological splits, walk-forward, frozen specifications
- metrics           forecast, classification and trading metrics
- backtest          threshold strategy simulation with costs and benchmarks
- plotting          publication-style figures
- reporting         assemble reports/research_report.md from computed tables
"""

__version__ = "0.1.0"
