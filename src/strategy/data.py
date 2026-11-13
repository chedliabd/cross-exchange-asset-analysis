"""Price and risk-free data for the Chapter 6 strategy.

The strategy trades the SPDR sector ETFs, so it only needs ETF/SPY prices and a
cash rate. Prices are total-return adjusted (dividends reinvested), which is the
same convention as the `Adj Close` column used in Chapters 1-5.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Sector -> SPDR ETF (same mapping as Chapter 4 and the Strategy Builder)
SECTOR_ETFS: dict[str, str] = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}
ETF_TO_SECTOR = {v: k for k, v in SECTOR_ETFS.items()}
BENCHMARK = "SPY"
RISK_FREE = "^IRX"  # 13-week T-bill yield, in percent (annualised)

# Study periods. The project's stock/ETF dataset ends on 2020-04-01, so everything
# after that date was never seen during the analysis -> a genuine holdout.
PERIODS: dict[str, tuple[str, str | None]] = {
    "design": ("1999-12-31", "2012-12-31"),
    "validation": ("2013-01-01", "2020-04-01"),
    "holdout": ("2020-04-02", None),
}


def download_prices(
    tickers: list[str] | None = None,
    start: str = "1998-12-01",
    end: str | None = None,
    cache_path: str | Path | None = "data/strategy/prices.csv",
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily adjusted close prices (columns = tickers), cached to CSV.

    `^IRX` is returned as a yield in percent, not a price.
    """
    tickers = tickers or [*SECTOR_ETFS.values(), BENCHMARK, RISK_FREE]
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(tickers) <= set(prices.columns):
            return prices[tickers]

    import yfinance as yf  # imported lazily so the rest of the module works offline

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )["Close"]
    prices = raw[tickers].sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.dropna(how="all")

    missing = [t for t in tickers if prices[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(f"No data downloaded for {missing}; try again with refresh=True")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return prices


def split_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the downloaded frame into (sector ETF prices, SPY prices, daily risk-free rate)."""
    sector_cols = [t for t in SECTOR_ETFS.values() if t in prices.columns]
    trading_days = prices[BENCHMARK].dropna().index
    sectors = prices.loc[trading_days, sector_cols]
    spy = prices.loc[trading_days, BENCHMARK]
    # ^IRX is an annualised yield in %, convert to a daily simple rate
    rf = (prices[RISK_FREE].reindex(trading_days).ffill().fillna(0.0) / 100.0) / 252.0
    rf.name = "rf"
    return sectors, spy, rf
