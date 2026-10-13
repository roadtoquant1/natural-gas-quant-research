"""SYNTHETIC DATA FOR TESTS ONLY.

Nothing in this file is real market or EIA data. The generators below produce
small, seeded, clearly artificial series so that pipeline behaviour (alignment,
no look-ahead, feature arithmetic) can be checked automatically. They must
never be used to draw research conclusions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC - test data, not real observations"


def synthetic_storage(start: str = "2010-01-01", years: int = 8, seed: int = 1) -> pd.DataFrame:
    """Weekly Friday 'storage levels' with a seasonal injection/withdrawal cycle."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=52 * years + 1, freq="W-FRI")
    doy = dates.dayofyear.to_numpy()
    seasonal = -90 * np.cos(2 * np.pi * (doy - 20) / 365.25)  # withdrawals in winter, injections in summer
    change = seasonal + rng.normal(0, 15, len(dates))
    level = 2500 + np.cumsum(change)
    return pd.DataFrame({"week_ending_date": dates, "storage_level_bcf": level,
                         "units": "BCF", "series": "SYNTHETIC", "label": SYNTHETIC_LABEL})


def synthetic_prices(start: str = "2009-12-01", end: str = "2018-03-31", seed: int = 2) -> pd.DataFrame:
    """Business-day 'prices' following a random walk; federal holidays set to NaN."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    lr = rng.normal(0, 0.03, len(dates))
    price = 3.0 * np.exp(np.cumsum(lr))
    df = pd.DataFrame({"date": dates, "price": price})
    hol = USFederalHolidayCalendar().holidays(start, end)
    df.loc[df["date"].isin(hol), "price"] = np.nan
    df["label"] = SYNTHETIC_LABEL
    return df


def make_test_config() -> dict:
    """A configuration matching the research defaults, used by integration tests."""
    return {
        "release": {
            "nominal_offset_days": 6,
            "nominal_time_et": "10:30",
            "flag_year_end_window": ["12-20", "01-05"],
            "entry_lag_trading_days": 1,
            "max_entry_gap_calendar_days": 7,
        },
        "features": {
            "seasonal_window_days": 10,
            "min_prior_years": 3,
            "same_year_exclusion_days": 180,
            "rolling_years": 3,
            "primary_method": "expanding_mean",
            "methods": ["expanding_mean", "rolling_median"],
            "trailing_vol_days": 20,
        },
        "targets": {"horizons": [1, 3, 5]},
    }
