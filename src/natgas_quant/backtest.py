"""A deliberately simple, educational backtest of the storage-anomaly signal.

READ THIS FIRST
---------------
* The Henry Hub **spot** price is an average of physical next-day deals
  reported after the fact. You cannot buy or sell "the spot index" at that
  price with one click. Results here are an **economic-signal simulation**:
  they ask whether the signal lines up with later price moves after costs,
  not whether a real account would have earned the same.
* A real implementation would trade futures or an exchange-traded fund, each
  with its own frictions (roll yield, contract expiry, fees, tracking error).
* Good historical performance is not evidence of future profitability.

Rule
----
    position = +1 (long)  if Z_t <= -k   (much less gas than seasonal norm)
             = -1 (short) if Z_t >= +k   (much more gas than seasonal norm)
             =  0         otherwise

Each trade enters at the first eligible post-release price, exits
``holding_days`` valid prices later, and pays a one-way cost plus slippage on
both entry and exit:

    gross  = position * (P_exit / P_entry - 1)
    cost   = 2 * (one_way_cost_bps + slippage_bps) / 10,000
    net    = gross - cost            (only when position != 0)

Example: long at 3.00, exit at 3.06, 10 bps cost + 5 bps slippage per side:
gross = +2.00%, cost = 0.30%, net = +1.70%.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import metrics as M


def generate_positions(z: pd.Series, k: float) -> pd.Series:
    """Threshold rule. Missing z means no position."""
    if k < 0:
        raise ValueError("k must be non-negative")
    zz = pd.Series(z, dtype=float)
    pos = pd.Series(0, index=zz.index, dtype=int)
    pos[zz <= -k] = 1
    pos[zz >= k] = -1
    pos[zz.isna()] = 0
    return pos


def run_backtest(
    events: pd.DataFrame,
    prices: np.ndarray,
    price_dates: pd.DatetimeIndex,
    positions: pd.Series,
    holding_days: int = 1,
    one_way_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate non-overlapping trades.

    Returns
    -------
    ledger:
        One row per executed trade: position, entry/exit dates and prices,
        gross return, cost, net return.
    weekly:
        One row per event (week): the position taken (0 if none) and the gross
        and net return attributed to that week (0 if no trade). This is the
        series used for Sharpe ratio and drawdown.
    """
    p = np.asarray(prices, dtype=float)
    pdates = pd.DatetimeIndex(price_dates)
    n = len(p)
    cost = 2.0 * (one_way_cost_bps + slippage_bps) / 1e4
    ev = events.sort_values("report_release_date")
    last_exit: pd.Timestamp | None = None
    ledger = []
    weekly = []
    for idx, row in ev.iterrows():
        pos = int(positions.get(idx, 0))
        e = int(row["entry_pos"])
        status = "no_signal"
        gross = net = 0.0
        taken = 0
        if pos != 0:
            if not bool(row["alignment_ok"]) or e < 0 or e + holding_days >= n:
                status = "no_price"
            elif last_exit is not None and pdates[e] < last_exit:
                status = "skipped_overlap"
            else:
                x = e + holding_days
                gross = pos * (p[x] / p[e] - 1.0)
                net = gross - cost
                taken = pos
                status = "traded"
                last_exit = pdates[x]
                ledger.append(
                    {
                        "event": idx,
                        "report_release_date": row["report_release_date"],
                        "signal_z": row.get("z", np.nan),
                        "position": pos,
                        "entry_date": pdates[e],
                        "exit_date": pdates[x],
                        "entry_price": p[e],
                        "exit_price": p[x],
                        "gross_return": gross,
                        "cost": cost,
                        "net_return": net,
                    }
                )
        weekly.append({"event": idx, "report_release_date": row["report_release_date"],
                       "position": taken, "status": status, "gross_return": gross, "net_return": net})
    return pd.DataFrame(ledger), pd.DataFrame(weekly).set_index("event")


def performance_summary(
    weekly: pd.DataFrame, ledger: pd.DataFrame, periods_per_year: float = 52
) -> dict[str, Any]:
    """Headline statistics. Sharpe uses weekly event returns (zeros when flat), x sqrt(52)."""
    if weekly.empty:
        return {"weeks": 0, "n_trades": 0}
    net = weekly["net_return"].to_numpy(dtype=float)
    gross = weekly["gross_return"].to_numpy(dtype=float)
    dates = pd.to_datetime(weekly["report_release_date"])
    years = max((dates.max() - dates.min()).days / 365.25, 1 / 52)
    n_tr = len(ledger)
    out: dict[str, Any] = {
        "weeks": len(weekly),
        "n_trades": n_tr,
        "n_long": int((ledger["position"] > 0).sum()) if n_tr else 0,
        "n_short": int((ledger["position"] < 0).sum()) if n_tr else 0,
        "skipped_overlap": int((weekly["status"] == "skipped_overlap").sum()),
        "cumulative_return_gross": float(np.prod(1 + gross) - 1),
        "cumulative_return_net": float(np.prod(1 + net) - 1),
        "annualized_return_net": M.annualized_return(net, periods_per_year),
        "annualized_vol_net": M.annualized_volatility(net, periods_per_year),
        "sharpe_net": M.sharpe_ratio(net, periods_per_year),
        "sharpe_gross": M.sharpe_ratio(gross, periods_per_year),
        "annualization_factor": f"sqrt({periods_per_year:g}) on weekly event returns",
        "max_drawdown_net": M.max_drawdown(net),
        "mean_net_return_per_week": float(np.mean(net)),
        "hit_rate": float((ledger["net_return"] > 0).mean()) if n_tr else float("nan"),
        "avg_gain": float(ledger.loc[ledger["net_return"] > 0, "net_return"].mean()) if n_tr else float("nan"),
        "avg_loss": float(ledger.loc[ledger["net_return"] <= 0, "net_return"].mean()) if n_tr else float("nan"),
        "total_costs": float(ledger["cost"].sum()) if n_tr else 0.0,
        "one_way_trades_per_year": 2 * n_tr / years,
    }
    return out


def select_k(
    events: pd.DataFrame,
    prices: np.ndarray,
    price_dates: pd.DatetimeIndex,
    k_grid: list[float],
    holding_days: int,
    one_way_cost_bps: float,
    slippage_bps: float,
    min_trades: int,
    periods_per_year: float = 52,
) -> tuple[float | None, pd.DataFrame]:
    """Choose k on the events you pass (the VALIDATION period only).

    Criterion (pre-registered): highest mean net return per week among k values
    with at least ``min_trades`` trades; ties go to the smaller k. Returns
    ``(None, table)`` if no k qualifies, in which case no strategy is tested.
    """
    rows = []
    for k in k_grid:
        pos = generate_positions(events["z"], k)
        led, wk = run_backtest(events, prices, price_dates, pos, holding_days, one_way_cost_bps, slippage_bps)
        s = performance_summary(wk, led, periods_per_year)
        rows.append({"k": k, **{c: s.get(c) for c in ("n_trades", "mean_net_return_per_week", "sharpe_net",
                                                       "cumulative_return_net", "hit_rate")}})
    tab = pd.DataFrame(rows)
    ok = tab[tab["n_trades"] >= min_trades]
    if ok.empty:
        return None, tab
    best = ok.sort_values(["mean_net_return_per_week", "k"], ascending=[False, True]).iloc[0]
    return float(best["k"]), tab


def random_signal_positions(index: pd.Index, probs: tuple[float, float, float], seed: int) -> pd.Series:
    """Random +1/0/-1 positions with given probabilities (long, flat, short)."""
    rng = np.random.default_rng(seed)
    draws = rng.choice([1, 0, -1], size=len(index), p=list(probs))
    return pd.Series(draws, index=index, dtype=int)


def random_benchmark_distribution(
    events: pd.DataFrame,
    prices: np.ndarray,
    price_dates: pd.DatetimeIndex,
    probs: tuple[float, float, float],
    draws: int,
    seed: int,
    holding_days: int,
    one_way_cost_bps: float,
    slippage_bps: float,
    periods_per_year: float = 52,
) -> pd.DataFrame:
    """Performance of many random strategies with the same trade frequency (a placebo)."""
    out = []
    for i in range(draws):
        pos = random_signal_positions(events.index, probs, seed + i)
        led, wk = run_backtest(events, prices, price_dates, pos, holding_days, one_way_cost_bps, slippage_bps)
        s = performance_summary(wk, led, periods_per_year)
        out.append({"draw": i, "mean_net_return_per_week": s["mean_net_return_per_week"],
                    "sharpe_net": s["sharpe_net"], "cumulative_return_net": s["cumulative_return_net"]})
    return pd.DataFrame(out)
