"""
Chapter 1 (sector diversification) helpers for `chapter1_sector_diversification.ipynb`.

The chapter compares the 11 sectors over three time windows with two metrics:
- returns: correlation of daily sector returns;
- volatility: correlation of 60-day sector volatility.
Each sector is summarised by its average correlation with the other sectors:
the lower it is, the more independent (diversifying) the sector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

WINDOW_COLORS = {"full": "#3987e5", "20y": "#d95926", "5y": "#199e70"}


# ---------- Time windows ----------

def make_windows(full_start: pd.Timestamp, end: str, starts: dict[str, str]) -> pd.DataFrame:
    """
    Table of windows: one row per key ("full" + the keys of `starts`), with
    start, end and the label shown on the website.
    """
    end = pd.Timestamp(end)
    rows = [{"key": "full", "start": full_start, "end": end, "name": "Full sample"}]
    names = {"20y": "Last 20 years", "5y": "Last 5 years"}
    for key, start in starts.items():
        rows.append({"key": key, "start": pd.Timestamp(start), "end": end, "name": names.get(key, key)})
    out = pd.DataFrame(rows).set_index("key")
    out["label"] = [f"{r.name} ({r.start.year}–{r.end.year})" for r in out.itertuples()]
    return out


# ---------- Correlations ----------

def corr_matrix(series: pd.DataFrame, start, end, min_periods: int) -> pd.DataFrame:
    """Pearson correlation of the columns of `series` between `start` and `end`."""
    return series.loc[start:end].corr(min_periods=min_periods)


def avg_corr(m: pd.DataFrame) -> pd.Series:
    """Average correlation of each sector with the other sectors (diagonal excluded)."""
    off = m.where(~np.eye(len(m), dtype=bool))
    return off.mean(axis=1)


def mean_pair_corr(m: pd.DataFrame) -> float:
    """Average correlation over all distinct pairs of sectors."""
    return float(np.nanmean(m.values[np.triu_indices(len(m), 1)]))


def cluster_order(m: pd.DataFrame) -> list[int]:
    """
    Sector order from average-linkage hierarchical clustering on the correlation
    distance sqrt(2 (1 - rho)), so related sectors sit next to each other.
    """
    d = np.sqrt(np.clip(2.0 * (1.0 - m.values), 0, None))
    np.fill_diagonal(d, 0.0)
    return leaves_list(linkage(squareform(d, checks=False), "average", optimal_ordering=True)).tolist()


# ---------- Website data ----------

def site_json(sectors, windows: pd.DataFrame, mats: dict) -> dict:
    """
    Build website/content/chapter1.json.

    `mats[metric][window_key]` is a sector × sector correlation DataFrame.
    """
    out = {
        "sectors": list(sectors),
        "windows": windows["label"].to_dict(),
        "metrics": {},
    }
    for metric, by_window in mats.items():
        avgs = {w: avg_corr(m) for w, m in by_window.items()}
        out["metrics"][metric] = {
            "corr": {w: m.round(2).values.tolist() for w, m in by_window.items()},
            "avg_corr": {w: a.round(3).tolist() for w, a in avgs.items()},
            "order": cluster_order(by_window["full"]),
            "mean_pair_corr": {w: round(mean_pair_corr(m), 3) for w, m in by_window.items()},
        }
    return out


# ---------- Figures (static versions of the website charts) ----------

def plot_avg_corr_dots(avgs: dict[str, pd.Series], windows: pd.DataFrame, title: str) -> plt.Figure:
    """
    Website chart 1: each sector's average correlation with the other sectors,
    one dot per window.
    """
    order = avgs["full"].sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.5))
    for i, sector in enumerate(order):
        vals = [avgs[w][sector] for w in windows.index]
        ax.plot([min(vals), max(vals)], [i, i], color="#c8c8c8", lw=2, zorder=1)
    for w in windows.index:
        ax.scatter(avgs[w][order], y, s=55, color=WINDOW_COLORS.get(w), label=windows.loc[w, "label"], zorder=2)
    ax.set_yticks(y, order)
    ax.set_xlabel("Average correlation with the other sectors (lower = better diversifier)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_corr_heatmaps(by_window: dict[str, pd.DataFrame], windows: pd.DataFrame, order: list[int], metric: str) -> plt.Figure:
    """
    Website chart 2: clustered lower-triangle correlation heatmaps, one panel per
    window plus the change "last window minus full sample".
    """
    sectors = by_window["full"].index[order]
    last = windows.index[-1]
    panels = {w: by_window[w].loc[sectors, sectors] for w in windows.index}
    panels["change"] = panels[last] - panels["full"]
    mask = np.triu(np.ones((len(sectors), len(sectors)), dtype=bool))

    ncols = 2
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 6.2 * nrows))
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax, (key, m) in zip(axes, panels.items()):
        data = np.ma.array(m.values, mask=mask)
        if key == "change":
            lim = max(np.nanmax(np.abs(data)), 1e-6)
            im = ax.imshow(data, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
            title = f"Change: {last.upper()} vs Full"
        elif metric == "returns":
            im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
            title = windows.loc[key, "label"]
        else:
            im = ax.imshow(data, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
            title = windows.loc[key, "label"]
        for i in range(len(sectors)):
            for j in range(i):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(sectors)), sectors, rotation=90, fontsize=8)
        ax.set_yticks(range(len(sectors)), sectors, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Sector correlations ({metric}), clustered order")
    fig.tight_layout()
    return fig
