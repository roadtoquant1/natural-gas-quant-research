"""Forward-looking targets: what happened to the price *after* we could act.

Log return
----------
    r_(t,h) = ln(P_(t+h) / P_t)

where t is the first eligible (post-release) price date and t+h is the h-th
valid price date after it. Log returns add across days
(ln(P2/P0) = ln(P1/P0) + ln(P2/P1)) and are close to simple percentage returns
when small: ln(3.15/3.00) = 0.0488, versus a simple return of 5.0%.

Realized volatility over h days
-------------------------------
    rv_h = sqrt( sum_{j=1..h} (ln(P_(t+j)/P_(t+j-1)))^2 )

It measures how much the price moved, regardless of direction. For h = 1,
rv_1 = |r_(t,1)|. Example: daily log returns +0.02, -0.03, +0.01 give
rv_3 = sqrt(0.0004 + 0.0009 + 0.0001) = 0.0374.

Direction
---------
    dir_h = 1 if r_(t,h) > 0 else 0     (NaN when the return is missing)

Announcement reaction (diagnostic only, NOT tradable)
-----------------------------------------------------
    reaction = ln(P_entry / P_before)

spans the release, so it shows whether the market moved when the report came
out. Because P_before is observed *before* the report, nobody could trade at it
after seeing the number; it is never used as a strategy return.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def max_calendar_days_for_horizon(h: int) -> int:
    """Largest believable calendar span for h trading days (weekends + a holiday)."""
    return math.ceil(h * 7 / 5) + 4


def forward_targets(
    aligned: pd.DataFrame,
    prices: np.ndarray,
    price_dates: pd.DatetimeIndex,
    horizons: list[int],
) -> pd.DataFrame:
    """Compute forward returns, direction, absolute return and realized vol.

    Parameters
    ----------
    aligned:
        Output of :func:`natgas_quant.release_calendar.align_releases_to_prices`
        (needs ``entry_pos, before_pos, alignment_ok``).
    prices, price_dates:
        Valid prices and their dates (same positions as used for alignment).
    horizons:
        Trading-day horizons, e.g. ``[1, 3, 5]``.
    """
    p = np.asarray(prices, dtype=float)
    pdates = pd.DatetimeIndex(price_dates)
    n = len(p)
    lr = np.full(n, np.nan)
    lr[1:] = np.log(p[1:] / p[:-1])
    e = aligned["entry_pos"].to_numpy(dtype=int)
    b = aligned["before_pos"].to_numpy(dtype=int)
    ok = aligned["alignment_ok"].to_numpy(dtype=bool)
    out = pd.DataFrame(index=aligned.index)
    out["price_before_release"] = np.where(ok & (b >= 0), p[np.clip(b, 0, n - 1)], np.nan)
    out["price_after_release"] = np.where(ok & (e >= 0), p[np.clip(e, 0, n - 1)], np.nan)
    out["reaction_ret"] = np.log(out["price_after_release"] / out["price_before_release"])
    for h in horizons:
        exit_pos = e + h
        valid = ok & (e >= 0) & (exit_pos < n)
        exit_dates = pd.Series(pd.NaT, index=aligned.index, dtype="datetime64[ns]")
        exit_dates[valid] = pdates.values[exit_pos[valid]]
        span = (exit_dates - pd.Series(pdates.values[np.clip(e, 0, n - 1)], index=aligned.index)).dt.days
        valid &= (span <= max_calendar_days_for_horizon(h)).to_numpy()
        fwd = np.full(len(e), np.nan)
        rv = np.full(len(e), np.nan)
        for i in np.flatnonzero(valid):
            fwd[i] = math.log(p[e[i] + h] / p[e[i]])
            seg = lr[e[i] + 1 : e[i] + h + 1]
            rv[i] = math.sqrt(float(np.sum(seg**2)))
        out[f"exit_date_{h}d"] = exit_dates.where(valid)
        out[f"fwd_ret_{h}d"] = fwd
        out[f"abs_ret_{h}d"] = np.abs(fwd)
        out[f"dir_{h}d"] = np.where(np.isnan(fwd), np.nan, (fwd > 0).astype(float))
        out[f"rv_{h}d"] = rv
    return out


def flag_overlap_with_next_release(
    release_dates: pd.Series, exit_dates: pd.Series
) -> pd.Series:
    """True if a holding window ends on/after the *next* week's release date.

    Such windows contain the next report too, so their returns are not driven
    by one report alone, and consecutive targets overlap.
    """
    rd = pd.to_datetime(pd.Series(release_dates)).reset_index(drop=True)
    ex = pd.to_datetime(pd.Series(exit_dates)).reset_index(drop=True)
    nxt = rd.shift(-1)
    flag = (ex >= nxt) & ex.notna() & nxt.notna()
    flag.index = pd.Series(release_dates).index
    return flag
