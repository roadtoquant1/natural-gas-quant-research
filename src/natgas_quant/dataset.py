"""Assemble the one-row-per-release event table from validated inputs.

Pipeline:  storage levels -> weekly change -> release calendar -> price
alignment -> forward targets -> prior-price features -> anomalies -> lags.

The function is pure (no file I/O) so it can be tested on tiny synthetic
inputs. ``scripts/build_dataset.py`` handles reading and writing files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import features as F
from . import release_calendar as RC
from . import targets as T

ALIGNMENT_TABLE_COLUMNS = {
    "storage_week_ending_date": "storage_week_ending_date",
    "report_release_date": "report_release_date",
    "report_release_timestamp_et": "report_release_timestamp_et",
    "storage_level_bcf": "storage_level_bcf",
    "storage_change_bcf": "storage_change_bcf",
    "first_eligible_price_date": "first_eligible_price_date",
    "price_before_release": "price_before_release",
    "price_after_release": "price_after_release",
    "fwd_ret_1d": "one_day_forward_return",
    "fwd_ret_3d": "three_day_forward_return",
    "fwd_ret_5d": "five_day_forward_return",
}


def build_event_table(
    storage: pd.DataFrame,
    prices: pd.DataFrame,
    verified_releases: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Return ``(events, valid_prices, valid_price_dates)``.

    Parameters
    ----------
    storage:
        Columns ``week_ending_date, storage_level_bcf`` (validated).
    prices:
        Columns ``date, price`` (validated; may contain NaN prices).
    verified_releases:
        Output of :func:`natgas_quant.release_calendar.load_verified_releases`.
    cfg:
        Full research configuration.
    """
    rcfg, fcfg, tcfg = cfg["release"], cfg["features"], cfg["targets"]
    st = storage.sort_values("week_ending_date").reset_index(drop=True)
    st = st.rename(columns={"week_ending_date": "storage_week_ending_date"})
    st["storage_change_bcf"] = F.weekly_change(st["storage_week_ending_date"], st["storage_level_bcf"]).to_numpy()

    cal = RC.build_release_calendar(
        st["storage_week_ending_date"],
        verified_releases,
        nominal_offset_days=rcfg["nominal_offset_days"],
        nominal_time_et=rcfg["nominal_time_et"],
        year_end_window=tuple(rcfg["flag_year_end_window"]) if rcfg.get("flag_year_end_window") else None,
    )
    ev = cal.merge(st[["storage_week_ending_date", "storage_level_bcf", "storage_change_bcf"]],
                   on="storage_week_ending_date", how="left")

    valid = prices.dropna(subset=["price"]).sort_values("date")
    p = valid["price"].to_numpy(dtype=float)
    pdates = pd.DatetimeIndex(valid["date"])

    ev = RC.align_releases_to_prices(
        ev, pdates,
        entry_lag_trading_days=rcfg["entry_lag_trading_days"],
        max_entry_gap_calendar_days=rcfg["max_entry_gap_calendar_days"],
    )
    RC.assert_no_lookahead(ev)

    tg = T.forward_targets(ev, p, pdates, tcfg["horizons"])
    ev = pd.concat([ev, tg], axis=1)
    for h in tcfg["horizons"]:
        ev[f"overlaps_next_release_{h}d"] = T.flag_overlap_with_next_release(
            ev["report_release_date"], ev[f"exit_date_{h}d"]
        ).to_numpy()

    pf = F.prior_price_features(p, ev["before_pos"].to_numpy(), fcfg["trailing_vol_days"])
    pf.index = ev.index
    ev = pd.concat([ev, pf], axis=1)

    cf = F.calendar_features(ev["storage_week_ending_date"], ev["report_release_date"])
    cf.index = ev.index
    ev = pd.concat([ev, cf], axis=1)

    ev = F.add_anomaly_features(ev, fcfg)
    ev["z_lag1"] = F.lag_by_one_week(ev["storage_week_ending_date"], ev["z"]).to_numpy()
    ev["entry_date_duplicated"] = ev["first_eligible_price_date"].duplicated(keep=False) & ev["alignment_ok"]
    return ev, p, pdates


def alignment_table(events: pd.DataFrame) -> pd.DataFrame:
    """The Stage-3 table with the exact column names requested in the protocol."""
    cols = [c for c in ALIGNMENT_TABLE_COLUMNS if c in events.columns]
    extra = ["release_date_source", "release_date_uncertain", "price_before_release_date", "alignment_ok"]
    out = events[cols + [c for c in extra if c in events.columns]].rename(columns=ALIGNMENT_TABLE_COLUMNS)
    return out


def modeling_sample(events: pd.DataFrame, feature: str = "z", target: str = "fwd_ret_1d") -> pd.DataFrame:
    """Rows usable for the primary analysis: aligned, with feature and target present."""
    m = events["alignment_ok"] & events[feature].notna() & events[target].notna()
    return events.loc[m].copy()
