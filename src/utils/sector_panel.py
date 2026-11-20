"""
Helpers for `preparing_sectors.ipynb`: the stock-level sector table
(`stocks_sect.parquet`) used by Chapters 2 and 5 and the volatility clustering.

Sector-level series are weighted averages of stock-level values, with weights
proportional to each stock's dollar volume (`Close * Volume`) that day.
"""

import pandas as pd


def attach_sectors(stocks: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Keep the stocks that have a sector in the metadata and add their `Sector` column."""
    sectors = (
        meta.loc[meta["Sector"].notna(), ["symbol", "Sector"]]
        .rename(columns={"symbol": "Symbol"})
        .drop_duplicates("Symbol")
    )
    return stocks.merge(sectors, on="Symbol", how="inner")


def add_daily_returns(df: pd.DataFrame, price_col: str = "Adj Close", cap: float = 0.5) -> tuple[pd.DataFrame, int]:
    """
    Add `Return`, the simple daily return of `price_col` for each symbol,
    capped at +/- `cap`.

    Simple (not log) returns are used because a sector's return on a day is a
    weighted average of its stocks' simple returns.

    Returns the DataFrame and the number of capped returns.
    """
    df = df.sort_values(["Symbol", "Date"]).copy()
    raw = df.groupby("Symbol")[price_col].pct_change()
    n_capped = int((raw.abs() > cap).sum())
    df["Return"] = raw.clip(lower=-cap, upper=cap)
    return df, n_capped


def add_rolling_volatility(df: pd.DataFrame, windows=(20, 60, 120)) -> pd.DataFrame:
    """
    Add `RollingVol{N}`: standard deviation of each symbol's daily returns over
    its last N rows (NaN until N returns are available).

    Requires df sorted by (Symbol, Date).
    """
    df = df.copy()
    grouped = df.groupby("Symbol")["Return"]
    for n in windows:
        df[f"RollingVol{n}"] = grouped.rolling(n).std().reset_index(level=0, drop=True)
    return df


def filter_window(df: pd.DataFrame, start: str | None, end: str | None, min_rows: int) -> pd.DataFrame:
    """
    Keep rows between `start` and `end` (None = no limit), then symbols with
    at least `min_rows` rows in that window.
    """
    if start is not None:
        df = df[df["Date"] >= start]
    if end is not None:
        df = df[df["Date"] <= end]
    counts = df["Symbol"].value_counts()
    return df[df["Symbol"].isin(counts[counts >= min_rows].index)].copy()


def _share_of_group(weight: pd.Series, valid: pd.Series, keys: list[pd.Series]) -> pd.Series:
    """weight / sum(weight) within each group, counting only rows where `valid` is True."""
    w = weight.where(valid)
    return w / w.groupby(keys).transform("sum")


def add_sector_weights(df: pd.DataFrame, windows=(20, 60, 120)) -> pd.DataFrame:
    """
    Add dollar-volume weights within each (Date, Sector):

    - `DollarVolume` = Close * Volume (both adjusted for splits only).
    - `Weight`: share of the sector's dollar volume that day, among the stocks
      that have a `Return` that day. Summing `Weight * Return` over a sector
      gives the sector return.
    - `WeightedVol{N}`: the stock's share of dollar volume among the stocks
      that have a `RollingVol{N}` that day, times `RollingVol{N}`. Summing it
      over a sector gives the sector volatility.

    Stocks without a value that day get no weight, so the weights of the
    stocks that are used always add up to 1.
    """
    df = df.copy()
    df["DollarVolume"] = df["Close"] * df["Volume"]
    keys = [df["Date"], df["Sector"]]

    df["Weight"] = _share_of_group(df["DollarVolume"], df["Return"].notna(), keys)
    for n in windows:
        vol = df[f"RollingVol{n}"]
        df[f"WeightedVol{n}"] = _share_of_group(df["DollarVolume"], vol.notna(), keys) * vol
    return df


def sector_series(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Sum a weighted column (e.g. `WeightedVol60`) per (Date, Sector) and return
    a wide table: index = Date, one column per sector.
    """
    out = df.groupby(["Date", "Sector"])[column].sum(min_count=1)
    return out.unstack("Sector").sort_index()


def sector_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Daily sector returns (sum of Weight * Return): index = Date, one column per sector."""
    weighted = df.assign(_wr=df["Weight"] * df["Return"])
    return sector_series(weighted, "_wr")


def stocks_per_sector(df: pd.DataFrame, column: str = "Return") -> pd.DataFrame:
    """Number of stocks with a value in `column` per day and sector: index = Date, one column per sector."""
    return df[df[column].notna()].groupby(["Date", "Sector"])["Symbol"].nunique().unstack("Sector").fillna(0).astype(int)


def first_date_all_sectors(stock_counts: pd.DataFrame, min_stocks: int) -> pd.Timestamp:
    """First date on which every sector has at least `min_stocks` stocks (see `stocks_per_sector`)."""
    ok = (stock_counts >= min_stocks).all(axis=1)
    if not ok.any():
        raise ValueError(f"No date where every sector has {min_stocks} stocks")
    return ok.idxmax()
