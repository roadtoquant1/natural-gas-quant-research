"""Event study: what happened after large and small storage anomalies?

An *event study* groups events (here, weekly storage releases) by a
characteristic (the size of the anomaly) and compares what followed. It is
descriptive: it shows averages and their uncertainty, not causes.

Groups use **fixed thresholds on the point-in-time z-score** (default:
-1.5, -0.5, +0.5, +1.5). Because z is built only from past data, a week's group
is known at the time. Quantiles computed over the whole sample would use
future information to decide what counts as "large".
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import block_bootstrap, default_block_length, percentile_ci


def assign_groups(z: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    """Label each event by its z-score bucket (left-open, right-closed intervals)."""
    return pd.cut(pd.Series(z, dtype=float), bins=list(bins), labels=list(labels), right=True)


def group_summary(
    df: pd.DataFrame,
    group_col: str,
    ret_col: str,
    reps: int = 2000,
    block: int | None = None,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """Per-group statistics of forward returns with t and block-bootstrap intervals.

    The bootstrap resamples blocks of consecutive *weeks* from the whole
    time-ordered sample (all groups together) and recomputes each group's mean,
    so any week-to-week dependence is carried into the interval.
    """
    data = df[[group_col, ret_col]].dropna().reset_index(drop=True)
    groups = list(data[group_col].cat.categories) if hasattr(data[group_col], "cat") else sorted(data[group_col].unique())
    y = data[ret_col].to_numpy(dtype=float)
    g = data[group_col].astype(str).to_numpy()
    n = len(y)
    rows = []
    b = default_block_length(n) if block is None else block
    for k, grp in enumerate(groups):
        mask = g == str(grp)
        v = y[mask]
        m = len(v)
        if m == 0:
            rows.append({"group": grp, "n": 0})
            continue
        sd = float(np.std(v, ddof=1)) if m > 1 else float("nan")
        se = sd / math.sqrt(m) if m > 1 else float("nan")
        q = stats.t.ppf(1 - (1 - level) / 2, df=m - 1) if m > 1 else float("nan")

        def stat(idx: np.ndarray, grp=str(grp)) -> float:
            sel = g[idx] == grp
            return float(y[idx][sel].mean()) if sel.any() else float("nan")

        draws = block_bootstrap(stat, n, reps=reps, block=b, seed=seed + k)
        lo, hi = percentile_ci(draws, level)
        rows.append(
            {
                "group": grp,
                "n": m,
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "std": sd,
                "se": se,
                "t_ci_low": float(v.mean() - q * se) if m > 1 else float("nan"),
                "t_ci_high": float(v.mean() + q * se) if m > 1 else float("nan"),
                "boot_ci_low": lo,
                "boot_ci_high": hi,
                "positive_freq": float(np.mean(v > 0)),
                "mean_abs": float(np.mean(np.abs(v))),
                "small_sample": m < 30,
            }
        )
    return pd.DataFrame(rows)


def group_summary_by(
    df: pd.DataFrame, by: str, group_col: str, ret_col: str, **kwargs
) -> pd.DataFrame:
    """Run :func:`group_summary` separately for each value of ``by`` (e.g. season)."""
    parts = []
    for key, sub in df.groupby(by, observed=True):
        s = group_summary(sub, group_col, ret_col, **kwargs)
        s.insert(0, by, key)
        parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def extreme_minus_extreme(
    df: pd.DataFrame, group_col: str, ret_col: str, low: str, high: str,
    reps: int = 2000, seed: int = 0, level: float = 0.95,
) -> dict[str, float]:
    """Mean return after large-negative minus after large-positive anomalies.

    H1 predicts a positive difference (low storage surprise -> higher returns).
    """
    data = df[[group_col, ret_col]].dropna().reset_index(drop=True)
    y = data[ret_col].to_numpy(dtype=float)
    g = data[group_col].astype(str).to_numpy()
    n = len(y)

    def stat(idx: np.ndarray) -> float:
        gl, gh = g[idx] == low, g[idx] == high
        if not gl.any() or not gh.any():
            return float("nan")
        return float(y[idx][gl].mean() - y[idx][gh].mean())

    point = stat(np.arange(n))
    draws = block_bootstrap(stat, n, reps=reps, seed=seed)
    lo, hi = percentile_ci(draws, level)
    valid = draws[~np.isnan(draws)]
    p_one = float(np.mean(valid <= 0)) if len(valid) else float("nan")
    return {"difference": point, "boot_ci_low": lo, "boot_ci_high": hi,
            "boot_p_one_sided_le_0": p_one, "n_low": int((g == low).sum()), "n_high": int((g == high).sum())}


def cumulative_return_paths(
    events: pd.DataFrame, prices: np.ndarray, offsets: Sequence[int] = range(-3, 7)
) -> pd.DataFrame:
    """Cumulative log return from the last pre-release price, by trading-day offset.

    Offset 0 is the last valid price dated before the release. Offset 1 is the
    next valid price, which is usually the release day itself (its daily spot
    price mixes pre- and post-release trades); the first tradable price
    (``entry_pos``) is usually offset 2. These are daily closes, so nothing
    here says anything about intraday timing.
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    rows = []
    for idx, ev in events.iterrows():
        b = int(ev["before_pos"])
        if b < 0 or not bool(ev.get("alignment_ok", True)):
            continue
        for k in offsets:
            j = b + k
            if 0 <= j < n:
                rows.append({"event": idx, "offset": k, "cum_ret": math.log(p[j] / p[b]),
                             "entry_offset": int(ev["entry_pos"]) - b})
    return pd.DataFrame(rows)
