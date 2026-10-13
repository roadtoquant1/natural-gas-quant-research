"""Point-in-time features.

Every function here obeys one rule: **a value for week t may use only
information that was public before the decision for week t was made.**

Storage anomaly, in plain language
----------------------------------
Storage normally rises in spring/summer (injections: gas is pumped into
underground storage) and falls in winter (withdrawals: gas is taken out for
heating). So a +80 Bcf change is ordinary in June and extraordinary in
January. To find *unusual* weeks we compare each change with what is normal
for that time of year, using **only earlier years**:

    Delta_S_t          = S_t - S_(t-1)                     (weekly change, Bcf)
    Expected_Delta_S_t = mean (or median) of Delta_S in comparable past weeks
    A_t                = Delta_S_t - Expected_Delta_S_t     (anomaly, Bcf)
    Z_t                = A_t / sigma_t                      (standardized anomaly)

"Comparable past weeks" are weeks from *previous years* whose date falls within
``window_days`` of the same calendar day. sigma_t is the standard deviation
(or a robust equivalent) of those same past observations.

Numerical example (see tests/test_features.py): if the comparable past
changes are 70, 80 and 90 Bcf, the mean is 80 and the sample standard deviation
is 10. A current change of 95 gives A = 15 Bcf and Z = 1.5, i.e. the build was
1.5 "typical deviations" larger than normal, which, other things equal, means more
gas than expected (bearish for price under H1).

Expanding versus rolling
------------------------
*Expanding* (Method A) uses every earlier year: more data, steadier estimate,
but slow to adapt if the market changes (e.g. growth of LNG exports).
*Rolling* (Method B) uses only the last N years (the industry's "five-year
average" idea): adapts faster, noisier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
MAD_TO_SIGMA = 1.4826  # makes the median absolute deviation comparable to a std for normal data


# ---------------------------------------------------------------------------
# Weekly change
# ---------------------------------------------------------------------------
def weekly_change(dates: pd.Series, levels: pd.Series) -> pd.Series:
    """Delta_S_t = S_t - S_(t-1), only when the previous row is exactly 7 days earlier.

    If a week is missing in the source, the change across the gap would mix two
    weeks, so it is left as NaN instead.
    """
    d = pd.to_datetime(dates).reset_index(drop=True)
    s = pd.Series(levels, dtype=float).reset_index(drop=True)
    if not d.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending")
    contiguous = d.diff().dt.days.eq(7)
    change = s.diff().where(contiguous)
    change.index = pd.Series(levels).index
    return change


# ---------------------------------------------------------------------------
# Seasonal expectation
# ---------------------------------------------------------------------------
def seasonal_distance_days(d: pd.Timestamp, others: np.ndarray) -> np.ndarray:
    """Distance in days between ``d`` and each date in ``others`` *ignoring the year*.

    Uses the elapsed days modulo one average year, so Dec 30 and Jan 2 are 3
    days apart and leap years need no special handling.
    """
    elapsed = (d.to_datetime64() - others) / np.timedelta64(1, "D")
    r = np.mod(elapsed, DAYS_PER_YEAR)
    return np.minimum(r, DAYS_PER_YEAR - r)


def seasonal_expectation(
    dates: pd.Series,
    changes: pd.Series,
    method: str = "expanding_mean",
    window_days: float = 10,
    min_prior_years: int = 5,
    same_year_exclusion_days: float = 180,
    rolling_years: int = 5,
) -> pd.DataFrame:
    """Point-in-time seasonal expectation and scale for each weekly change.

    Parameters
    ----------
    method:
        ``expanding_mean``, ``expanding_median``, ``rolling_mean`` or
        ``rolling_median``. *mean* pairs with the sample standard deviation;
        *median* pairs with 1.4826 x median absolute deviation, which resists
        extreme weeks.
    window_days:
        Comparable observations must be within this many days of the same
        calendar day in an earlier year.
    min_prior_years:
        Minimum number of distinct earlier years with a comparable observation.
        Below that, the expectation is NaN (no anomaly is produced).
    same_year_exclusion_days:
        Comparable observations must be more than this many days older than the
        current week, so neighbouring weeks of the *current* year (and the
        current week itself) never enter their own baseline.
    rolling_years:
        For ``rolling_*`` methods, only the most recent N earlier years are used.

    Returns
    -------
    DataFrame with ``expected, scale, n_obs, n_years`` aligned to ``changes``.
    """
    valid_methods = {"expanding_mean", "expanding_median", "rolling_mean", "rolling_median"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}")
    rolling = method.startswith("rolling")
    use_median = method.endswith("median")
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    x = pd.Series(changes, dtype=float).to_numpy()
    n = len(x)
    expected = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    n_obs = np.zeros(n, dtype=int)
    n_years = np.zeros(n, dtype=int)
    for i in range(n):
        di = pd.Timestamp(d[i])
        age = (d[i] - d) / np.timedelta64(1, "D")          # >0 means earlier
        cand = (age > same_year_exclusion_days) & ~np.isnan(x)
        if not cand.any():
            continue
        dist = seasonal_distance_days(di, d)
        cand &= dist <= window_days
        years_back = np.rint(age / DAYS_PER_YEAR).astype(int)
        if rolling:
            cand &= (years_back >= 1) & (years_back <= rolling_years)
        if not cand.any():
            continue
        vals = x[cand]
        yrs = np.unique(years_back[cand])
        n_obs[i] = len(vals)
        n_years[i] = len(yrs)
        if len(yrs) < min_prior_years or len(vals) < 2:
            continue
        if use_median:
            med = float(np.median(vals))
            expected[i] = med
            scale[i] = MAD_TO_SIGMA * float(np.median(np.abs(vals - med)))
        else:
            expected[i] = float(np.mean(vals))
            scale[i] = float(np.std(vals, ddof=1))
    scale[scale <= 0] = np.nan
    idx = pd.Series(changes).index
    return pd.DataFrame({"expected": expected, "scale": scale, "n_obs": n_obs, "n_years": n_years}, index=idx)


def storage_anomaly(changes: pd.Series, expected: pd.Series, scale: pd.Series) -> pd.DataFrame:
    """Return the raw anomaly A_t (Bcf) and standardized anomaly Z_t."""
    a = pd.Series(changes, dtype=float) - pd.Series(expected, dtype=float)
    z = a / pd.Series(scale, dtype=float)
    return pd.DataFrame({"anomaly_bcf": a, "z": z})


def add_anomaly_features(df: pd.DataFrame, cfg_features: dict) -> pd.DataFrame:
    """Add expectation/anomaly/z columns for every configured method.

    Expects columns ``storage_week_ending_date`` and ``storage_change_bcf``.
    Adds ``expected_<m>, scale_<m>, anomaly_<m>, z_<m>, n_years_<m>``, plus the
    primary-method aliases ``expected_change_bcf, anomaly_bcf, z``.
    """
    out = df.copy()
    for m in cfg_features["methods"]:
        se = seasonal_expectation(
            out["storage_week_ending_date"],
            out["storage_change_bcf"],
            method=m,
            window_days=cfg_features["seasonal_window_days"],
            min_prior_years=cfg_features["min_prior_years"],
            same_year_exclusion_days=cfg_features["same_year_exclusion_days"],
            rolling_years=cfg_features["rolling_years"],
        )
        an = storage_anomaly(out["storage_change_bcf"], se["expected"], se["scale"])
        out[f"expected_{m}"] = se["expected"]
        out[f"scale_{m}"] = se["scale"]
        out[f"n_years_{m}"] = se["n_years"]
        out[f"anomaly_{m}"] = an["anomaly_bcf"]
        out[f"z_{m}"] = an["z"]
    p = cfg_features["primary_method"]
    out["expected_change_bcf"] = out[f"expected_{p}"]
    out["anomaly_bcf"] = out[f"anomaly_{p}"]
    out["z"] = out[f"z_{p}"]
    out["abs_z"] = out["z"].abs()
    return out


def lag_by_one_week(dates: pd.Series, values: pd.Series) -> pd.Series:
    """Previous week's value, only if the previous row is exactly 7 days earlier."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    v = pd.Series(values, dtype=float).reset_index(drop=True)
    lagged = v.shift(1).where(d.diff().dt.days.eq(7))
    lagged.index = pd.Series(values).index
    return lagged


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def calendar_features(week_ending: pd.Series, release_date: pd.Series) -> pd.DataFrame:
    """Month, meteorological season, ISO week, storage season, and smooth seasonality.

    ``withdrawal_season`` = 1 for releases in November..March, the months in
    which U.S. storage typically falls (the "winter" half used in robustness
    checks); 0 for April..October (injection season, "summer").
    ``woy_sin/woy_cos`` encode the time of year as a point on a circle so that
    late December and early January are close together.
    """
    we = pd.to_datetime(pd.Series(week_ending))
    rd = pd.to_datetime(pd.Series(release_date))
    month = rd.dt.month
    doy = we.dt.dayofyear
    out = pd.DataFrame(index=pd.Series(week_ending).index)
    out["month"] = month.to_numpy()
    out["season"] = month.map(SEASON_OF_MONTH).to_numpy()
    out["week_of_year"] = we.dt.isocalendar().week.astype(int).to_numpy()
    out["withdrawal_season"] = month.isin([11, 12, 1, 2, 3]).astype(int).to_numpy()
    out["storage_season"] = np.where(out["withdrawal_season"] == 1, "withdrawal (Nov-Mar)", "injection (Apr-Oct)")
    out["woy_sin"] = np.sin(2 * np.pi * doy / DAYS_PER_YEAR).to_numpy()
    out["woy_cos"] = np.cos(2 * np.pi * doy / DAYS_PER_YEAR).to_numpy()
    out["year"] = rd.dt.year.to_numpy()
    return out


# ---------------------------------------------------------------------------
# Prior-price features (known before entry)
# ---------------------------------------------------------------------------
def prior_price_features(prices: np.ndarray, before_pos: np.ndarray, vol_days: int = 20) -> pd.DataFrame:
    """Returns and volatility measured up to the last price *before* the release.

    Using the pre-release price (not the release-day price) is deliberately
    conservative: the release-day spot price partly reflects trades made after
    the report.

    prior_ret_1d     = ln(P_b / P_(b-1))
    prior_ret_5d     = ln(P_b / P_(b-5))
    trailing_vol_20d = sample std of the 20 daily log returns ending at b
                       (daily units, not annualized)
    where b is the position of the last valid price before the release.
    """
    p = np.asarray(prices, dtype=float)
    lr = np.full(len(p), np.nan)
    lr[1:] = np.log(p[1:] / p[:-1])
    b = np.asarray(before_pos, dtype=int)
    r1 = np.full(len(b), np.nan)
    r5 = np.full(len(b), np.nan)
    vol = np.full(len(b), np.nan)
    for i, bi in enumerate(b):
        if bi < 0:
            continue
        if bi >= 1:
            r1[i] = np.log(p[bi] / p[bi - 1])
        if bi >= 5:
            r5[i] = np.log(p[bi] / p[bi - 5])
        if bi >= vol_days:
            vol[i] = np.std(lr[bi - vol_days + 1 : bi + 1], ddof=1)
    return pd.DataFrame({"prior_ret_1d": r1, "prior_ret_5d": r5, f"trailing_vol_{vol_days}d": vol})


# ---------------------------------------------------------------------------
# Winsorization (robustness only)
# ---------------------------------------------------------------------------
def fit_winsor_bounds(values: pd.Series, quantiles: tuple[float, float]) -> tuple[float, float]:
    """Quantile bounds estimated on the data you pass (pass training data only)."""
    v = pd.Series(values, dtype=float).dropna()
    return float(v.quantile(quantiles[0])), float(v.quantile(quantiles[1]))


def winsorize(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    """Clip values to fixed bounds. Never the default: robustness test only."""
    lo, hi = bounds
    return pd.Series(values, dtype=float).clip(lower=lo, upper=hi)
