"""Map each storage week to the date the public could first see it, and to the
first price that could have been traded afterwards.

Why this matters
----------------
EIA measures storage for a week ending on a **Friday** but publishes it the
following **Thursday at 10:30 a.m. ET** (six days later). Lining prices up
with the Friday week-ending date would let the model "know" the number almost
a week before anyone could. That is *look-ahead bias*: using information in a
historical test that was not available at the time.

Conventions used (all configurable in ``research_config.yaml``)
---------------------------------------------------------------
1. ``report_release_date``:
   * if the week is listed in ``data/raw/manual/eia_wngsr_verified_release_dates.csv``
     (dates copied from EIA's published schedule), that date is used;
   * otherwise Thursday = week-ending + 6 days;
   * if a federal holiday falls on the Tuesday, Wednesday or Thursday of the
     release week, EIA has moved the release in the past, sometimes earlier,
     sometimes later. We assume the **later** possibility (next business day
     after Thursday) and flag the week as uncertain. Assuming too late only
     wastes a day; assuming too early would leak the future.
   * Monday holidays do not move the date: EIA's 2025-26 schedule lists no
     exception for Monday holidays. This is an inference from the schedule,
     not a documented rule, and is recorded in the report.
   * late-December/early-January weeks are flagged uncertain because ad-hoc
     federal closures are not in the holiday calendar.
2. ``first_eligible_price_date``: the first date **strictly after** the
   release date that has a valid price (the N-th such date if
   ``entry_lag_trading_days`` = N). A daily spot price on the release day mixes
   trades from before and after 10:30 a.m., so it is never used as an entry.
3. ``price_before_release``: the last valid price dated **strictly before** the
   release date. It is used only to describe the market's reaction, never as
   a tradable entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def federal_holidays(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """U.S. federal holidays (observed dates) between ``start`` and ``end``."""
    return USFederalHolidayCalendar().holidays(start=start, end=end)


def next_business_day(day: pd.Timestamp, holidays: pd.DatetimeIndex) -> pd.Timestamp:
    """First weekday strictly after ``day`` that is not a federal holiday."""
    d = day + pd.Timedelta(days=1)
    hol = set(holidays)
    while d.dayofweek >= 5 or d in hol:
        d += pd.Timedelta(days=1)
    return d


def _in_year_end_window(day: pd.Timestamp, window: tuple[str, str] | None) -> bool:
    if not window:
        return False
    start_md, end_md = window
    md = day.strftime("%m-%d")
    if start_md <= end_md:
        return start_md <= md <= end_md
    return md >= start_md or md <= end_md  # window wraps across New Year


def rule_based_release(
    week_ending: pd.Timestamp,
    holidays: pd.DatetimeIndex,
    nominal_offset_days: int = 6,
    year_end_window: tuple[str, str] | None = ("12-20", "01-05"),
) -> tuple[pd.Timestamp, str, bool, str]:
    """Conservative release date for one week when no verified date exists.

    Returns ``(release_date, source, uncertain, note)``.
    """
    nominal = week_ending + pd.Timedelta(days=nominal_offset_days)
    hol = set(holidays)
    # Tuesday..Thursday of the release week (nominal is Thursday)
    window = [nominal - pd.Timedelta(days=k) for k in (2, 1, 0)]
    hits = [d for d in window if d in hol]
    if hits:
        assumed = next_business_day(nominal, holidays)
        note = (
            f"holiday on {', '.join(str(h.date()) for h in hits)}; "
            f"assumed later release {assumed.date()}"
        )
        return assumed, "rule_holiday_assume_later", True, note
    if nominal.dayofweek != 3:
        # Only possible if week_ending is not a Friday; keep but flag.
        return nominal, "rule_nominal_non_thursday", True, "week-ending date is not a Friday"
    if _in_year_end_window(nominal, year_end_window):
        return nominal, "rule_nominal", True, "year-end window: ad-hoc closures possible"
    return nominal, "rule_nominal", False, ""


def load_verified_releases(path: str | Path | None) -> pd.DataFrame:
    """Load hand-verified release dates; empty frame if the file is absent."""
    cols = ["storage_week_ending_date", "report_release_date", "report_release_time_et"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Verified release file lacks columns {missing}")
    df["storage_week_ending_date"] = pd.to_datetime(df["storage_week_ending_date"])
    df["report_release_date"] = pd.to_datetime(df["report_release_date"])
    if df["storage_week_ending_date"].duplicated().any():
        raise ValueError("Duplicate week-ending dates in verified release file")
    bad = df["report_release_date"] <= df["storage_week_ending_date"]
    if bad.any():
        raise ValueError("A verified release date is not after its week-ending date")
    return df


def build_release_calendar(
    week_endings: Iterable[pd.Timestamp],
    verified: pd.DataFrame | None = None,
    nominal_offset_days: int = 6,
    nominal_time_et: str = "10:30",
    year_end_window: tuple[str, str] | None = ("12-20", "01-05"),
) -> pd.DataFrame:
    """One row per storage week with its (verified or conservatively assumed) release.

    Columns: ``storage_week_ending_date, nominal_release_date,
    report_release_date, report_release_timestamp_et, release_time_standard_or_verified,
    release_date_source, release_date_uncertain, release_note``.
    """
    we = pd.DatetimeIndex(pd.to_datetime(list(week_endings))).sort_values()
    if we.empty:
        return pd.DataFrame(
            columns=[
                "storage_week_ending_date", "nominal_release_date", "report_release_date",
                "report_release_timestamp_et", "release_time_standard_or_verified", "release_date_source",
                "release_date_uncertain", "release_note",
            ]
        )
    holidays = federal_holidays(we.min() - pd.Timedelta(days=7), we.max() + pd.Timedelta(days=21))
    vmap: dict[pd.Timestamp, tuple[pd.Timestamp, str]] = {}
    if verified is not None and not verified.empty:
        for _, r in verified.iterrows():
            vmap[pd.Timestamp(r["storage_week_ending_date"])] = (
                pd.Timestamp(r["report_release_date"]),
                str(r["report_release_time_et"]),
            )
    rows = []
    for w in we:
        nominal = w + pd.Timedelta(days=nominal_offset_days)
        if w in vmap:
            rel, t = vmap[w]
            rows.append((w, nominal, rel, f"{rel.date()} {t}", True, "verified_schedule", False, ""))
        else:
            rel, src, unc, note = rule_based_release(w, holidays, nominal_offset_days, year_end_window)
            known = src == "rule_nominal" and not unc
            ts = f"{rel.date()} {nominal_time_et}" + ("" if known else " (time assumed)")
            rows.append((w, nominal, rel, ts, known, src, unc, note))
    return pd.DataFrame(
        rows,
        columns=[
            "storage_week_ending_date", "nominal_release_date", "report_release_date",
            "report_release_timestamp_et", "release_time_standard_or_verified", "release_date_source",
            "release_date_uncertain", "release_note",
        ],
    )


def align_releases_to_prices(
    calendar: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    entry_lag_trading_days: int = 1,
    max_entry_gap_calendar_days: int = 7,
) -> pd.DataFrame:
    """Attach the pre-release and first-eligible price dates to each release.

    Parameters
    ----------
    calendar:
        Output of :func:`build_release_calendar`.
    price_dates:
        Sorted dates **that have a valid price** (missing prices removed).
    entry_lag_trading_days:
        1 = first valid price strictly after the release date; 2 = second, etc.

    Returns
    -------
    DataFrame with added columns ``before_pos, price_before_release_date,
    entry_pos, first_eligible_price_date, entry_gap_days, alignment_ok``.
    ``*_pos`` are integer positions into ``price_dates`` used by
    :mod:`natgas_quant.targets` and :mod:`natgas_quant.features`.
    """
    if entry_lag_trading_days < 1:
        raise ValueError("entry_lag_trading_days must be >= 1 (never trade on the release day)")
    pdates = pd.DatetimeIndex(price_dates)
    if not pdates.is_monotonic_increasing or pdates.has_duplicates:
        raise ValueError("price_dates must be sorted and unique")
    out = calendar.copy()
    rel = pd.DatetimeIndex(out["report_release_date"]).values
    # position of first price date strictly greater than release date
    first_after = np.searchsorted(pdates.values, rel, side="right")
    entry_pos = first_after + (entry_lag_trading_days - 1)
    before_pos = np.searchsorted(pdates.values, rel, side="left") - 1
    n = len(pdates)
    entry_valid = entry_pos < n
    before_valid = before_pos >= 0
    entry_dates = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    entry_dates[entry_valid] = pdates.values[entry_pos[entry_valid]]
    before_dates = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    before_dates[before_valid] = pdates.values[before_pos[before_valid]]
    gap = (entry_dates - pd.Series(rel, index=out.index)).dt.days
    first_gap = pd.Series(np.nan, index=out.index)
    fa_valid = first_after < n
    first_gap[fa_valid] = (pdates.values[first_after[fa_valid]] - rel[fa_valid]) / np.timedelta64(1, "D")
    before_gap = (pd.Series(rel, index=out.index) - before_dates).dt.days
    ok = (
        entry_valid
        & before_valid
        & (first_gap <= max_entry_gap_calendar_days)
        & (before_gap <= max_entry_gap_calendar_days)
    )
    out["before_pos"] = np.where(before_valid, before_pos, -1)
    out["price_before_release_date"] = before_dates
    out["entry_pos"] = np.where(entry_valid, entry_pos, -1)
    out["first_eligible_price_date"] = entry_dates
    out["entry_gap_days"] = gap
    out["alignment_ok"] = ok.astype(bool)
    return out


def assert_no_lookahead(aligned: pd.DataFrame) -> None:
    """Raise if any entry is on/before its release date or before-price is not before it.

    Called by the dataset builder as a hard guard, and exercised by the tests.
    """
    ok = aligned[aligned["alignment_ok"]]
    if (ok["first_eligible_price_date"] <= ok["report_release_date"]).any():
        raise AssertionError("Entry price dated on or before the release date")
    if (ok["price_before_release_date"] >= ok["report_release_date"]).any():
        raise AssertionError("'Before' price is not strictly before the release date")
    if (ok["report_release_date"] <= ok["storage_week_ending_date"]).any():
        raise AssertionError("Release date not after storage week-ending date")
