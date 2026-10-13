"""Chronological validation and the "freeze before test" discipline.

Why not a random split?
-----------------------
A random train/test split would put, say, March 2023 in the test set and
April 2023 in the training set. The model would learn from the future and be
graded on the past. Time-series validation must always train on earlier data
and test on later data.

Three periods
-------------
* **train**: fit models.
* **validation**: compare models, pick hyperparameters and the trading
  threshold k.
* **test**: evaluated **once**, after the specification is frozen. Looking at
  the test set, changing the model, and looking again silently turns the test
  set into a second validation set.

This module writes the split boundaries and the frozen specification to
``reports/tables`` with a hash, and logs every test evaluation so repeated
peeking is visible in the report.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def chronological_split(dates: pd.Series, fractions: dict[str, float]) -> tuple[pd.Series, dict[str, str]]:
    """Label rows train/validation/test by date order.

    Returns the label Series and a dict of boundary dates (ISO strings).
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    d = pd.to_datetime(pd.Series(dates))
    if d.isna().any():
        raise ValueError("dates contain NaT")
    order = np.argsort(d.to_numpy(), kind="stable")
    n = len(d)
    n_train = int(round(fractions["train"] * n))
    n_val = int(round(fractions["validation"] * n))
    labels = np.empty(n, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train : n_train + n_val]] = "validation"
    labels[order[n_train + n_val :]] = "test"
    lab = pd.Series(labels, index=d.index)
    bounds = {
        "train_start": str(d[lab == "train"].min().date()),
        "train_end": str(d[lab == "train"].max().date()),
        "validation_start": str(d[lab == "validation"].min().date()),
        "validation_end": str(d[lab == "validation"].max().date()),
        "test_start": str(d[lab == "test"].min().date()),
        "test_end": str(d[lab == "test"].max().date()),
    }
    return lab, bounds


def labels_from_bounds(dates: pd.Series, bounds: dict[str, str]) -> pd.Series:
    """Re-apply frozen split boundaries to (possibly extended) data."""
    d = pd.to_datetime(pd.Series(dates))
    lab = pd.Series("train", index=d.index, dtype=object)
    lab[d >= pd.Timestamp(bounds["validation_start"])] = "validation"
    lab[d >= pd.Timestamp(bounds["test_start"])] = "test"
    lab[d < pd.Timestamp(bounds["train_start"])] = "pre_sample"
    return lab


def assert_chronological(train_dates: pd.Series, test_dates: pd.Series) -> None:
    """Raise unless every training date is strictly earlier than every test date."""
    if len(train_dates) and len(test_dates):
        if pd.to_datetime(train_dates).max() >= pd.to_datetime(test_dates).min():
            raise AssertionError("Training data overlap or follow the evaluation data")


def walk_forward_by_year(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    model_factory: Callable[[], Any],
    min_train_years: int = 3,
    first_year: int | None = None,
    last_year: int | None = None,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Expanding-window forecasts: fit on all years <= Y-1, predict year Y, repeat.

    Every prediction is stored with its forecast (release) date, the last
    training date and the model name.
    """
    d = pd.to_datetime(df[date_col])
    years = sorted(d.dt.year.unique())
    if not years:
        return pd.DataFrame()
    start_year = years[0] + min_train_years if first_year is None else first_year
    rows = []
    req = [target] + (required_cols or [])
    for y in years:
        if y < start_year or (last_year is not None and y > last_year):
            continue
        train = df[(d.dt.year < y)].dropna(subset=req)
        test = df[(d.dt.year == y)].dropna(subset=[c for c in req if c != target])
        if len(train) < 20 or test.empty:
            continue
        assert_chronological(train[date_col], test[date_col])
        model = model_factory().fit(train, target)
        pred = model.predict(test)
        rows.append(
            pd.DataFrame(
                {
                    "forecast_date": pd.to_datetime(test[date_col]).to_numpy(),
                    "year": y,
                    "train_end": pd.to_datetime(train[date_col]).max(),
                    "n_train": len(train),
                    "y": test[target].to_numpy(dtype=float),
                    "yhat": np.asarray(pred, dtype=float),
                    "model": getattr(model, "name", "model"),
                },
                index=test.index,
            )
        )
    return pd.concat(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Freezing specifications
# ---------------------------------------------------------------------------
def spec_hash(spec: dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def save_frozen(spec: dict[str, Any], path: Path) -> str:
    """Write a frozen specification with its hash and timestamp. Returns the hash."""
    h = spec_hash(spec)
    payload = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "spec_hash": h, "spec": spec}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return h


def load_frozen(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run both development phases first: "
            "python scripts/run_analysis.py --phase development, then "
            "python scripts/run_backtest.py --phase development"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if spec_hash(payload["spec"]) != payload["spec_hash"]:
        raise ValueError(f"{path} was edited after freezing (hash mismatch)")
    return payload


def log_test_evaluation(log_path: Path, spec_hash_value: str, note: str, kind: str = "models") -> pd.DataFrame:
    """Append one line per test-set evaluation; return the full log.

    The report shows this log. More than one distinct hash for the same
    ``kind`` ("models" or "backtest") means the test set was evaluated under
    more than one specification of that kind.
    """
    row = pd.DataFrame([{"evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "kind": kind, "spec_hash": spec_hash_value, "note": note}])
    if log_path.exists():
        log = pd.concat([pd.read_csv(log_path), row], ignore_index=True)
    else:
        log = row
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    return log
