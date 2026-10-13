"""Parse raw files into tidy tables and check them.

Nothing here changes the raw files. Each parser keeps the original text of
the date and value columns (``*_raw``) next to the parsed versions, so any
parsing decision can be audited.

Validation failures that make the data unusable (duplicate dates, negative
prices, wrong units) raise :class:`DataValidationError`. Softer issues
(missing business days, irregular week spacing) are *reported*, never fixed
silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MISSING_SYMBOLS = {"", ".", "NA", "N/A", "--", "-", "W", "nan", "NaN"}


class DataValidationError(ValueError):
    """Raised when data fail a check that makes them unusable."""


@dataclass
class ValidationReport:
    """Human-readable summary of checks run on one dataset."""

    name: str
    facts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "facts": self.facts, "warnings": self.warnings}

    def to_markdown(self) -> str:
        lines = [f"### {self.name}", "", "| Check | Result |", "|---|---|"]
        for k, v in self.facts.items():
            lines.append(f"| {k} | {v} |")
        if self.warnings:
            lines += ["", "**Warnings**", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)


def _to_number(raw: pd.Series) -> pd.Series:
    """Convert text to float, mapping known missing-value symbols to NaN.

    Anything that is neither a number nor a known missing symbol raises, so an
    unexpected code in the source cannot quietly become NaN.
    """
    s = raw.astype("string").str.strip()
    is_missing = s.isna() | s.isin(MISSING_SYMBOLS)
    num = pd.to_numeric(s.where(~is_missing), errors="coerce")
    bad = num.isna() & ~is_missing
    if bad.any():
        examples = s[bad].unique()[:5].tolist()
        raise DataValidationError(f"Unparseable numeric values: {examples}")
    return num.astype(float)


# ---------------------------------------------------------------------------
# Henry Hub prices (FRED)
# ---------------------------------------------------------------------------
def parse_fred_csv(path: str | Path, series_id: str = "DHHNGSP") -> pd.DataFrame:
    """Parse FRED's CSV into ``date_raw, value_raw, date, price``.

    FRED has used both ``DATE`` and ``observation_date`` as the date header and
    both ``.`` and an empty field for missing values; all are handled.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    date_col = next((c for c in df.columns if c.lower() in {"date", "observation_date"}), None)
    if date_col is None or series_id not in df.columns:
        raise DataValidationError(f"Unexpected FRED columns: {list(df.columns)}")
    out = pd.DataFrame({"date_raw": df[date_col], "value_raw": df[series_id]})
    out["date"] = pd.to_datetime(out["date_raw"], format="%Y-%m-%d", errors="raise")
    out["price"] = _to_number(out["value_raw"])
    return out


def parse_fred_api_json(path: str | Path) -> pd.DataFrame:
    """Parse a FRED API JSON payload into the same layout as :func:`parse_fred_csv`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    obs = pd.DataFrame(payload["observations"])
    out = pd.DataFrame({"date_raw": obs["date"], "value_raw": obs["value"]})
    out["date"] = pd.to_datetime(out["date_raw"], format="%Y-%m-%d")
    out["price"] = _to_number(out["value_raw"])
    return out


def validate_prices(df: pd.DataFrame, name: str = "Henry Hub spot (DHHNGSP)") -> tuple[pd.DataFrame, ValidationReport]:
    """Check a parsed price table and return (clean_table, report).

    Clean table = sorted by date, all rows kept (missing prices stay NaN).
    Raises on duplicate dates or non-positive prices.
    """
    rep = ValidationReport(name)
    if df.empty:
        raise DataValidationError("Price table is empty")
    dups = df["date"].duplicated(keep=False)
    if dups.any():
        raise DataValidationError(f"Duplicate dates: {df.loc[dups, 'date'].dt.date.unique()[:5].tolist()}")
    nonpos = df["price"] <= 0
    if nonpos.any():
        raise DataValidationError(
            f"{int(nonpos.sum())} zero/negative prices, e.g. "
            f"{df.loc[nonpos, ['date', 'price']].head().to_dict('records')}. "
            "No source explanation is on file, so these are rejected."
        )
    was_sorted = bool(df["date"].is_monotonic_increasing)
    out = df.sort_values("date").reset_index(drop=True)
    valid = out.dropna(subset=["price"])
    weekend = out["date"].dt.dayofweek >= 5
    bdays = pd.bdate_range(out["date"].min(), out["date"].max())
    absent_bdays = bdays.difference(out["date"])
    gaps = valid["date"].diff().dt.days
    rep.facts.update(
        {
            "rows": len(out),
            "first_date": str(out["date"].min().date()),
            "last_date": str(out["date"].max().date()),
            "valid_prices": len(valid),
            "rows_with_missing_price": int(out["price"].isna().sum()),
            "weekend_rows": int(weekend.sum()),
            "business_days_absent_from_file": len(absent_bdays),
            "longest_gap_between_valid_prices_days": int(gaps.max()) if len(gaps) > 1 else 0,
            "min_price": float(valid["price"].min()),
            "max_price": float(valid["price"].max()),
            "input_was_sorted": was_sorted,
        }
    )
    if weekend.any():
        rep.warnings.append(
            "Weekend-dated rows exist; the date may then be a delivery (flow) date rather "
            "than a trade date. Re-check the entry-lag convention."
        )
    else:
        rep.warnings.append(
            "No weekend-dated rows: consistent with trade-date labelling, but this is an "
            "inference, not a documented fact. The 2-day entry-lag robustness test covers the alternative."
        )
    if rep.facts["longest_gap_between_valid_prices_days"] > 5:
        rep.warnings.append("A gap of more than 5 calendar days between valid prices exists.")
    if not was_sorted:
        rep.warnings.append("Input was not sorted by date; sorted copy returned.")
    return out, rep


def missing_price_table(df: pd.DataFrame) -> pd.DataFrame:
    """List every weekday row whose price is missing, with the weekday name.

    These are typically market holidays; the table makes that checkable.
    """
    miss = df[df["price"].isna()].copy()
    miss["weekday"] = miss["date"].dt.day_name()
    return miss[["date", "date_raw", "value_raw", "weekday"]]


# ---------------------------------------------------------------------------
# EIA storage
# ---------------------------------------------------------------------------
def parse_eia_storage_json(path: str | Path) -> pd.DataFrame:
    """Flatten the saved EIA pages into one table, preserving raw strings."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for page in payload["pages"]:
        rows.extend(page["response"]["data"])
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise DataValidationError("EIA storage file contains no rows")
    out = raw.copy()
    out["period_raw"] = raw["period"].astype(str)
    out["value_raw"] = raw["value"].astype("string")
    out["week_ending_date"] = pd.to_datetime(out["period_raw"], format="%Y-%m-%d")
    out["storage_level_bcf"] = _to_number(out["value_raw"])
    return out


def validate_storage(
    df: pd.DataFrame,
    expected_series: str,
    expected_units_contains: str = "BCF",
    name: str = "EIA weekly working gas, Lower 48",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Check the storage table and return (clean_table, report).

    Returns columns ``week_ending_date, storage_level_bcf, period_raw,
    value_raw, units, series``. Weekly *change* is computed later in
    :mod:`natgas_quant.features` so its rule is tested in one place.
    """
    rep = ValidationReport(name)
    if "series" in df.columns:
        series = set(df["series"].astype(str))
        if series != {expected_series}:
            raise DataValidationError(f"Expected only series {expected_series}, found {series}")
    units = set(df["units"].astype(str)) if "units" in df.columns else set()
    if not units or not all(expected_units_contains.lower() in u.lower() for u in units):
        raise DataValidationError(f"Units {units} do not contain '{expected_units_contains}'")
    dups = df["week_ending_date"].duplicated(keep=False)
    if dups.any():
        raise DataValidationError(f"Duplicate week-ending dates: {df.loc[dups, 'week_ending_date'].unique()[:5]}")
    out = df.sort_values("week_ending_date").reset_index(drop=True)
    keep = ["week_ending_date", "storage_level_bcf", "period_raw", "value_raw"]
    keep += [c for c in ("units", "series", "series-description", "duoarea", "area-name", "process", "process-name") if c in out.columns]
    out = out[keep]
    weekday = out["week_ending_date"].dt.dayofweek
    spacing = out["week_ending_date"].diff().dt.days
    irregular = spacing.notna() & (spacing != 7)
    rep.facts.update(
        {
            "rows": len(out),
            "series": expected_series,
            "units": ", ".join(sorted(units)),
            "first_week_ending": str(out["week_ending_date"].min().date()),
            "last_week_ending": str(out["week_ending_date"].max().date()),
            "missing_levels": int(out["storage_level_bcf"].isna().sum()),
            "non_friday_week_endings": int((weekday != 4).sum()),
            "irregular_spacing_count": int(irregular.sum()),
            "min_level_bcf": float(out["storage_level_bcf"].min()),
            "max_level_bcf": float(out["storage_level_bcf"].max()),
        }
    )
    if (weekday != 4).any():
        rep.warnings.append("Some week-ending dates are not Fridays; inspect before use.")
    if irregular.any():
        rep.warnings.append(
            f"{int(irregular.sum())} gaps where consecutive weeks are not 7 days apart; "
            "weekly change is left missing across those gaps."
        )
    rep.warnings.append(
        "Values are current vintage. EIA revises storage estimates (including "
        "reclassifications between base and working gas), so the weekly change computed "
        "from today's levels can differ from the 'net change' printed on release day."
    )
    return out, rep


def compare_vintages(old: pd.DataFrame, new: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """Return rows present in both vintages whose values differ.

    Use with an archived raw file (``data/raw/archive``) to measure revisions.
    """
    m = old[[key, value]].merge(new[[key, value]], on=key, suffixes=("_old", "_new"))
    diff = ~np.isclose(m[f"{value}_old"], m[f"{value}_new"], equal_nan=True)
    out = m.loc[diff].copy()
    out["revision"] = out[f"{value}_new"] - out[f"{value}_old"]
    return out
