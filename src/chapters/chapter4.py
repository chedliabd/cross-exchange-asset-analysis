"""
Chapter 4 (sector ETF vs DIY basket) helpers for `chapter4_etf_vs_basket.ipynb`.

For each sector, a "DIY basket" of the sector's stocks is compared with the
sector's SPDR ETF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.strategy.data import SECTOR_ETFS  # noqa: F401  (re-exported for the notebook)
from src.utils.figures import themed


def capped_weights(raw: pd.DataFrame, cap: float | None, max_iter: int = 50) -> pd.DataFrame:
    """
    Normalise each row of `raw` (non-negative, NaN = not eligible) to sum to 1,
    with no weight above `cap`: weights above the cap are fixed at the cap and
    the excess is spread over the others, until no weight exceeds it.
    """
    w = raw.div(raw.sum(axis=1), axis=0)
    if cap is None:
        return w
    # With fewer than 1/cap eligible stocks the cap cannot hold: use equal weights that day.
    n = raw.notna().sum(axis=1)
    too_few = (n > 0) & (n * cap < 1)
    w.loc[too_few] = raw.loc[too_few].notna().div(n[too_few], axis=0).where(raw.loc[too_few].notna())
    for _ in range(max_iter):
        over = w > cap + 1e-12
        if not over.any().any():
            break
        fixed = w.where(over).notna()
        free = w.where(~fixed)
        excess = 1 - cap * fixed.sum(axis=1)
        w = free.div(free.sum(axis=1), axis=0).mul(excess, axis=0).where(~fixed, cap)
    return w


def basket_returns(
    stocks: pd.DataFrame, sector: str, dates: pd.DatetimeIndex, cap: float | None, lag: int = 1
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Daily return of a DIY basket of the sector's stocks.

    - Weight of stock i on day t: its dollar volume on day t - `lag` (known before
      day t's return), among the stocks that have a return on day t, capped at `cap`.
    - Basket return: sum of weight × simple daily return.

    Returns the basket returns and the weights (index = Date, one column per stock).
    """
    d = stocks[stocks["Sector"] == sector]
    ret = d.pivot(index="Date", columns="Symbol", values="Return").sort_index()
    dv = d.pivot(index="Date", columns="Symbol", values="DollarVolume").sort_index()
    raw = dv.shift(lag).where(ret.notna() & (dv.shift(lag) > 0))
    raw = raw.reindex(dates)
    weights = capped_weights(raw, cap)
    basket = (weights * ret.reindex(dates)).sum(axis=1, min_count=1)
    return basket.rename("Basket"), weights


def etf_returns(etfs: pd.DataFrame, ticker: str) -> pd.Series:
    """Simple daily returns of one ETF from `etfs_clean` (Adj Close, so dividends included)."""
    prices = etfs.loc[etfs["Symbol"] == ticker].set_index("Date")["Adj Close"].sort_index()
    return prices.pct_change().dropna().rename("ETF")


# ---------- Figures (website versions use theme="dark") ----------

def _year_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_performance(pair: pd.DataFrame, sector: str, etf: str, theme: str = "light") -> plt.Figure:
    """ETF vs basket growth, both rebased to 100 on the first common day."""
    index = 100 * (1 + pair[["ETF", "Basket"]]).cumprod()
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(13, 5.2))
        ax.plot(index.index, index["Basket"], lw=2.2, color=t["series"][0], label=f"{sector} DIY basket")
        ax.plot(index.index, index["ETF"], lw=2.2, color=t["accent"], label=f"{etf} (ETF)")
        ax.set(title=f"{sector}: ETF vs DIY sector basket (index = 100)", ylabel="Index (base = 100)", xlabel="")
        ax.legend(loc="upper left")
        _year_axis(ax)
        fig.tight_layout()
    return fig


def plot_volatility(pair: pd.DataFrame, sector: str, etf: str, window: int, theme: str = "light") -> plt.Figure:
    """Rolling annualised volatility of the ETF and the basket."""
    vol = pair[["ETF", "Basket"]].rolling(window).std() * np.sqrt(252)
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(13, 5.2))
        ax.plot(vol.index, vol["Basket"], lw=2, color=t["series"][0], label=f"{sector} DIY basket")
        ax.plot(vol.index, vol["ETF"], lw=2, color=t["accent"], label=f"{etf} (ETF)")
        ax.set(title=f"{sector}: rolling volatility ({window}-day, annualised)", ylabel="Annualised volatility", xlabel="")
        ax.legend(loc="upper left")
        _year_axis(ax)
        fig.tight_layout()
    return fig
