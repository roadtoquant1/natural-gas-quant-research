"""Publication-style static figures (PNG) for the research report.

Conventions: one y-axis per chart (never two scales), units in every axis
label, recessive grid, readable dates, a legend whenever two or more series are
shown, and direct labels where they help. Colors come from a validated
colorblind-aware palette; group identity is never carried by color alone
(axis labels or direct labels repeat it).

Every function saves a file and returns ``(path, caption)`` so the report
generator can place the caption under the figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated light-mode palette (categorical order fixed, never cycled)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
# Diverging blue <-> red with neutral gray midpoint (ordered anomaly groups)
DIVERGING_5 = ["#1c5cab", "#5598e7", "#a3a29b", "#e98a89", "#c0302f"]
DIVERGING_3 = ["#1c5cab", "#7a7974", "#c0302f"]
SEQ_BLUE = "#2a78d6"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "legend.frameon": False,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _pct(ax: plt.Axes, axis: str = "y") -> None:
    fmt = matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=1)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Data audit / exploratory figures
# ---------------------------------------------------------------------------
def storage_level(ev: pd.DataFrame, out: Path) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(ev["storage_week_ending_date"], ev["storage_level_bcf"], color=SEQ_BLUE)
    ax.set_title("U.S. Lower 48 working gas in underground storage")
    ax.set_ylabel("Working gas (Bcf)")
    ax.set_xlabel("Storage week-ending date")
    _date_axis(ax)
    return _save(fig, out), (
        "Weekly working gas in storage (EIA series NW2_EPG0_SWO_R48_BCF, current vintage). "
        "The saw-tooth pattern is the annual injection (spring-autumn) and withdrawal (winter) cycle."
    )


def weekly_change(ev: pd.DataFrame, out: Path) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(ev["storage_week_ending_date"], ev["storage_change_bcf"], width=5, color=SEQ_BLUE)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title("Weekly change in working gas")
    ax.set_ylabel("Change from prior week (Bcf)")
    ax.set_xlabel("Storage week-ending date")
    _date_axis(ax)
    return _save(fig, out), "Weekly storage change, S_t - S_(t-1). Positive = net injection; negative = net withdrawal."


def seasonal_change(ev: pd.DataFrame, out: Path) -> tuple[Path, str]:
    apply_style()
    d = ev.dropna(subset=["storage_change_bcf"])
    months = range(1, 13)
    data = [d.loc[d["storage_week_ending_date"].dt.month == m, "storage_change_bcf"] for m in months]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bp = ax.boxplot(data, tick_labels=[pd.Timestamp(2000, m, 1).strftime("%b") for m in months],
                    patch_artist=True, widths=0.6, medianprops={"color": INK},
                    flierprops={"marker": "o", "markersize": 3, "markerfacecolor": AXIS, "markeredgecolor": AXIS})
    for b in bp["boxes"]:
        b.set_facecolor("#cde2fb")
        b.set_edgecolor(SEQ_BLUE)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_title("Weekly storage change by month: why raw changes are seasonal")
    ax.set_ylabel("Weekly change (Bcf)")
    ax.set_xlabel("Month of storage week-ending date")
    return _save(fig, out), (
        "Distribution of weekly storage changes by calendar month. Boxes span the middle 50%; the line is the median. "
        "A change that is normal in one month is extreme in another, so anomalies are measured against a seasonal baseline."
    )


def price_history(prices: pd.DataFrame, out: Path) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(prices["date"], prices["price"], color=SEQ_BLUE, lw=1.2)
    ax.set_title("Henry Hub natural-gas spot price (daily)")
    ax.set_ylabel("Price (US$ per MMBtu)")
    ax.set_xlabel("Date")
    ax.set_ylim(bottom=0)
    _date_axis(ax)
    return _save(fig, out), (
        "FRED series DHHNGSP (source: EIA/Refinitiv). Gaps are market holidays or missing reports. "
        "The y-axis starts at zero to avoid exaggerating moves."
    )


def histogram(values: pd.Series, out: Path, title: str, xlabel: str, caption: str, pct: bool = False) -> tuple[Path, str]:
    apply_style()
    v = pd.Series(values).dropna()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(v, bins=50, color=SEQ_BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(0, color=INK_2, lw=1, ls="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of weeks")
    if pct:
        _pct(ax, "x")
    return _save(fig, out), caption + f" n = {len(v)}."


def scatter_anomaly_return(ev: pd.DataFrame, x: str, y: str, out: Path, caption: str) -> tuple[Path, str]:
    apply_style()
    d = ev[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(d[x], d[y], s=14, color=SEQ_BLUE, alpha=0.45, edgecolors="none", label="One release week")
    if len(d) >= 10:
        bins = pd.qcut(d[x], 10, duplicates="drop")
        g = d.groupby(bins, observed=True).agg(xm=(x, "mean"), ym=(y, "mean"))
        ax.plot(g["xm"], g["ym"], color=SERIES[1], marker="o", markersize=6, lw=2, label="Decile average")
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_title("Storage anomaly vs subsequent return")
    ax.set_xlabel("Standardized storage anomaly Z (unitless)")
    ax.set_ylabel("Forward log return")
    _pct(ax)
    ax.legend(loc="upper right")
    return _save(fig, out), caption + f" n = {len(d)}. Association is not causation."


def bucket_bars(tab: pd.DataFrame, out: Path, title: str, ylabel: str, caption: str,
                label_col: str = "group", lo: str = "boot_ci_low", hi: str = "boot_ci_high",
                colors: Sequence[str] | None = None, pct: bool = True) -> tuple[Path, str]:
    """Means with confidence intervals for ordered groups."""
    apply_style()
    t = tab.dropna(subset=["mean"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(t))
    cols = list(colors) if colors is not None else [SEQ_BLUE] * len(t)
    ax.bar(x, t["mean"], color=cols[: len(t)], width=0.6, edgecolor=SURFACE, linewidth=2)
    err = np.vstack([t["mean"] - t[lo], t[hi] - t["mean"]])
    ax.errorbar(x, t["mean"], yerr=err, fmt="none", ecolor=INK_2, capsize=4, lw=1.2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, [f"{str(g).replace('_', ' ')}\n(n={int(n)})" for g, n in zip(t[label_col], t["n"])])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if pct:
        _pct(ax)
    return _save(fig, out), caption


def correlation_heatmap(corr: pd.DataFrame, out: Path, caption: str) -> tuple[Path, str]:
    apply_style()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#c0302f"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(corr), 0.8 + 0.55 * len(corr)))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Pearson correlation")
    ax.set_title("Correlation matrix (development sample)")
    return _save(fig, out), caption


def sample_counts(counts: pd.DataFrame, out: Path) -> tuple[Path, str]:
    """Grouped bars: weeks per year by storage season."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    years = counts.index.astype(int).to_numpy()
    w = 0.38
    for i, col in enumerate(counts.columns):
        ax.bar(years + (i - 0.5) * w, counts[col], width=w, color=SERIES[i], label=col, edgecolor=SURFACE, linewidth=1)
    ax.set_title("Usable release weeks per year, by storage season")
    ax.set_ylabel("Number of weeks")
    ax.set_xlabel("Release year")
    ax.legend(loc="upper left", ncols=2)
    return _save(fig, out), "Weeks with a valid anomaly and 1-day target. Early years are lost to the minimum-history requirement."


def car_paths(car: pd.DataFrame, out: Path) -> tuple[Path, str]:
    """Average cumulative log return around the release for three anomaly groups."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = ["large_negative", "normal", "large_positive"]
    for g, c in zip(groups, DIVERGING_3):
        sub = car[car["group"] == g]
        if sub.empty:
            continue
        m = sub.groupby("offset")["cum_ret"].mean()
        n = sub["event"].nunique()
        ax.plot(m.index, m.values, color=c, marker="o", markersize=5, label=f"{g.replace('_', ' ')} (n={n})")
        ax.annotate(g.replace("_", " "), (m.index[-1], m.values[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK_2, fontsize=9, va="center")
    ax.axvspan(0, 2, color=GRID, alpha=0.5, lw=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xlabel("Valid trading days after the last pre-release price (0)")
    ax.set_ylabel("Average cumulative log return")
    ax.set_title("Price path around the storage release, by anomaly group")
    _pct(ax)
    ax.legend(loc="upper left")
    return _save(fig, out), (
        "Average cumulative log return measured from the last daily price before the release (day 0). "
        "The shaded span covers the release: day 1 is usually the release day itself, whose daily price "
        "mixes pre- and post-release trades; day 2 is usually the first tradable price. Daily data "
        "say nothing about intraday timing. Only the extreme groups and the middle group are drawn for legibility."
    )


def residual_diagnostics(fitted: np.ndarray, resid: np.ndarray, out: Path, max_lag: int = 12) -> tuple[Path, str]:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(fitted, resid, s=12, color=SEQ_BLUE, alpha=0.5, edgecolors="none")
    axes[0].axhline(0, color=AXIS, lw=1)
    axes[0].set_title("Residuals vs fitted values")
    axes[0].set_xlabel("Fitted forward return")
    axes[0].set_ylabel("Residual")
    e = resid - resid.mean()
    acf = [float(e[l:] @ e[:-l] / (e @ e)) for l in range(1, max_lag + 1)]
    axes[1].bar(range(1, max_lag + 1), acf, color=SEQ_BLUE, width=0.6)
    band = 1.96 / np.sqrt(len(e))
    axes[1].axhline(band, color=INK_2, ls="--", lw=1)
    axes[1].axhline(-band, color=INK_2, ls="--", lw=1)
    axes[1].axhline(0, color=AXIS, lw=1)
    axes[1].set_title("Residual autocorrelation")
    axes[1].set_xlabel("Lag (release weeks)")
    axes[1].set_ylabel("Autocorrelation")
    return _save(fig, out), (
        "Left: a funnel shape would indicate heteroskedasticity. Right: bars outside the dashed "
        "approximate 95% band indicate serial correlation. Both motivate Newey-West standard errors."
    )


def coefficient_stability(tab: pd.DataFrame, out: Path, label: str) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(tab["end_year"], tab["ci_low"], tab["ci_high"], color="#cde2fb", lw=0, label="95% HAC interval")
    ax.plot(tab["end_year"], tab["coef"], color=SEQ_BLUE, marker="o", markersize=5, label="Estimate")
    ax.axhline(0, color=INK_2, lw=1, ls="--")
    ax.set_title(f"Expanding-window estimate of the {label} coefficient")
    ax.set_xlabel("Last year included in the estimation sample")
    ax.set_ylabel("Coefficient (log return per unit Z)")
    ax.legend(loc="upper right")
    return _save(fig, out), (
        "Each point re-estimates the univariate regression using all data up to the year shown. "
        "A coefficient whose sign or size wanders, or whose band always straddles zero, is not stable evidence."
    )


def equity_curves(curves: dict[str, pd.Series], out: Path, title: str, caption: str) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, s) in enumerate(curves.items()):
        ax.plot(s.index, s.values, color=SERIES[i % len(SERIES)], label=name, lw=2 if i == 0 else 1.6)
        ax.annotate(name, (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center")
    ax.axhline(1.0, color=AXIS, lw=1)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (compounded)")
    ax.set_xlabel("Release date")
    ax.legend(loc="upper left")
    return _save(fig, out), caption


def calibration_plot(tab: pd.DataFrame, out: Path, title: str, xlabel: str, ylabel: str, caption: str,
                     pct: bool = False) -> tuple[Path, str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lo = float(min(tab["mean_prediction"].min(), tab["mean_outcome"].min()))
    hi = float(max(tab["mean_prediction"].max(), tab["mean_outcome"].max()))
    ax.plot([lo, hi], [lo, hi], color=INK_2, ls="--", lw=1, label="Perfect calibration")
    ax.plot(tab["mean_prediction"], tab["mean_outcome"], color=SEQ_BLUE, marker="o", markersize=7, label="Model")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if pct:
        _pct(ax)
        _pct(ax, "x")
    ax.legend(loc="upper left")
    return _save(fig, out), caption
