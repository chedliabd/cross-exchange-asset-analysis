"""
Chapter 2 (market regimes) helpers for `chapter2_market_regimes.ipynb`.

Each day is labelled Bear / Recovery / Neutral / Bull from SPY's drawdown, then sector
returns, volatility and correlations are compared across regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from src.utils.figures import themed

REGIMES = ["Bear", "Recovery", "Neutral", "Bull"]


# ---------- Regimes ----------

def spy_regimes(spy_prices: pd.Series, bear_threshold: float = -0.20, neutral_threshold: float = -0.05) -> pd.DataFrame:
    """
    Label each day from SPY's drawdown (distance below its running maximum):

    - Bull: drawdown above `neutral_threshold`;
    - Neutral: drawdown between `bear_threshold` and `neutral_threshold`;
    - Bear: drawdown at or below `bear_threshold`, from the peak down to the
      lowest point of that decline;
    - Recovery: drawdown still at or below `bear_threshold`, but after that
      lowest point (the market is rising again from the bottom).

    Finding the lowest point uses the whole decline, so this labelling looks at
    the future: fine to describe the past, not usable as a trading signal.
    """
    spy_prices = spy_prices.sort_index()
    peak = spy_prices.cummax()
    drawdown = spy_prices / peak - 1
    regime = pd.Series("Bull", index=spy_prices.index)
    regime[drawdown <= neutral_threshold] = "Neutral"
    regime[drawdown <= bear_threshold] = "Bear"

    # one decline = all the days below the same peak; its bottom splits Bear from Recovery
    decline_id = (peak != peak.shift()).cumsum()
    bear = regime == "Bear"
    for _, prices in spy_prices[bear].groupby(decline_id[bear]):
        regime[prices.index[prices.index > prices.idxmin()]] = "Recovery"
    return pd.DataFrame({"SPY": spy_prices, "Drawdown": drawdown, "Regime": pd.Categorical(regime, REGIMES, ordered=True)})


# ---------- Correlations ----------

def rolling_avg_corr(returns: pd.DataFrame, window: int) -> pd.Series:
    """Average pairwise correlation between the columns of `returns` over a rolling window."""
    n = returns.shape[1]
    corr = returns.rolling(window).corr()
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    values = corr.values.reshape(-1, n, n)[:, upper]
    out = pd.DataFrame(values, index=returns.index).mean(axis=1).rename(f"avg_corr_{window}d")
    return out[returns.notna().sum(axis=1).rolling(window).min() == n].dropna()


def clustered(corr: pd.DataFrame) -> pd.DataFrame:
    """Reorder a correlation matrix by average-linkage clustering on 1 - correlation."""
    d = (1 - corr).values.copy()
    np.fill_diagonal(d, 0.0)
    order = leaves_list(linkage(squareform(d, checks=False), "average"))
    return corr.iloc[order, order]


# ---------- Figures (website versions use theme="dark") ----------

def plot_corr_heatmap(corr: pd.DataFrame, title: str, theme: str = "light") -> plt.Figure:
    """Full correlation heatmap with values, on a -1..1 diverging scale."""
    with themed(theme) as t:
        cmap = LinearSegmentedColormap.from_list(f"div_{theme}", t["diverging"])
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=t["text"])
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)), corr.index)
        ax.grid(False)
        ax.spines[:].set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Correlation")
        cbar.outline.set_visible(False)
        ax.set_title(title)
        fig.tight_layout()
    return fig


def plot_line(series: pd.Series, title: str, ylabel: str, theme: str = "light") -> plt.Figure:
    """A single time series."""
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(series.index, series.values, color=t["accent2"])
        ax.set(title=title, ylabel=ylabel, xlabel="")
        fig.tight_layout()
    return fig


def plot_bars(values: pd.Series, title: str, ylabel: str, theme: str = "light") -> plt.Figure:
    """Vertical bars in the order of `values`."""
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(max(6, 0.75 * len(values) + 2), 4))
        ax.bar(values.index, values.values, color=t["accent"])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="x", visible=False)
        plt.setp(ax.get_xticklabels(), rotation=45 if len(values) > 4 else 0, ha="right" if len(values) > 4 else "center")
        fig.tight_layout()
    return fig


def plot_barh(values: pd.Series, title: str, xlabel: str, theme: str = "light") -> plt.Figure:
    """Horizontal bars, largest at the top, with a zero line."""
    values = values.sort_values()
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(values) + 1.2))
        ax.barh(values.index, values.values, color=t["accent"])
        ax.axvline(0, color=t["muted"], lw=1)
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig
