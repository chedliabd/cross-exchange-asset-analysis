"""
Cleaning helpers shared by `preparing_stocks.ipynb` and `preparing_etfs.ipynb`.

Every function works on a dict {symbol: DataFrame} with the raw Kaggle columns
`Date, Open, High, Low, Close, Adj Close, Volume`.
"""

import numpy as np
import pandas as pd


def remove_symbols(all_data: dict[str, pd.DataFrame], symbols_to_remove) -> dict[str, pd.DataFrame]:
    """
    Return a new dict without the given symbols.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.
    symbols_to_remove : iterable of str
        Symbols to remove completely.
    """
    to_remove = set(symbols_to_remove)
    return {symbol: df for symbol, df in all_data.items() if symbol not in to_remove}


def get_symbols_with_price_jumps(
    all_data: dict[str, pd.DataFrame], column: str = "Close", max_ratio: float = 5.0
) -> list[str]:
    """
    Symbols whose price moves by more than `max_ratio` (up) or 1/`max_ratio` (down)
    from one day to the next.

    `Close` is already adjusted for splits, so such a jump is not a real price
    move: it is usually a ticker that belonged to another security before
    (e.g. data from years before the company listed).
    """
    jumps = []
    for symbol, df in all_data.items():
        ratio = df[column] / df[column].shift(1)
        if ((ratio > max_ratio) | (ratio < 1 / max_ratio)).any():
            jumps.append(symbol)
    return sorted(jumps)


def add_weekday_column(all_data: dict[str, pd.DataFrame], date_col: str = "Date") -> None:
    """
    Add a 'Weekday' column (0 = Mon ... 4 = Fri) right after the date column.

    Modifies the DataFrames in place.
    """
    for df in all_data.values():
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if "Weekday" in df.columns:
            df["Weekday"] = df[date_col].dt.weekday
        else:
            df.insert(df.columns.get_loc(date_col) + 1, "Weekday", df[date_col].dt.weekday)


def trim_before_last_nonpositive(all_data: dict[str, pd.DataFrame], column: str = "Adj Close") -> list[str]:
    """
    Cut each series so it starts the day after its last non-positive `column` value.

    Yahoo's `Adj Close` is adjusted backwards for all later dividends; for very old
    dates of high-dividend stocks this can push it to zero or below, while the rest
    of the history is valid. Cutting the start keeps one continuous series.

    Modifies the dict in place and returns the symbols that were cut.
    """
    trimmed = []
    for symbol, df in all_data.items():
        bad = df.index[df[column] <= 0]
        if len(bad) == 0:
            continue
        all_data[symbol] = df.loc[df.index > bad.max()].copy()
        trimmed.append(symbol)
    return sorted(trimmed)


def fill_zero_open_with_prev_close(all_data: dict[str, pd.DataFrame], date_col: str = "Date") -> None:
    """
    Repair `Open == 0` with the previous trading day's `Close`.

    `Open` and `Close` are on the same price basis (adjusted for splits only),
    so `Open` is filled with the previous `Close`, not with `Adj Close`
    (which is also adjusted for dividends).

    - Rows with `Open == 0` get the previous day's `Close`.
    - If the first row has `Open == 0`, there is no previous day to use,
      so that row is dropped.

    Assumes there are no NaNs or zeros left in `Close`. Modifies the dict in place.
    """
    for symbol, df in all_data.items():
        if df.empty or not {"Open", "Close"}.issubset(df.columns):
            continue

        df = df.sort_values(date_col)
        prev_close = df["Close"].shift(1)

        zero_open = df["Open"] == 0
        if not zero_open.any():
            all_data[symbol] = df
            continue

        df.loc[zero_open, "Open"] = prev_close[zero_open]

        # The first row has no previous Close (NaN after the fill): drop it.
        df = df[df["Open"].notna()]
        all_data[symbol] = df


def trim_leading_zero_volumes(all_data: dict[str, pd.DataFrame]) -> None:
    """
    Drop the rows at the start of each series until the first non-zero `Volume`.

    These rows are usually days before the symbol really started trading.
    Symbols with no non-zero volume at all are left unchanged.
    Modifies the dict in place.
    """
    for symbol, df in all_data.items():
        if "Volume" not in df.columns:
            raise ValueError(f"Symbol {symbol} has no 'Volume' column")

        nonzero_idx = df.index[df["Volume"] > 0]
        if len(nonzero_idx) > 0:
            all_data[symbol] = df.loc[nonzero_idx[0]:].copy()


def add_volatility(all_data: dict[str, pd.DataFrame]) -> None:
    """
    Add `Volatility = (High - Low) / Open`, the normalised intraday price range.

    Modifies the DataFrames in place.
    """
    for df in all_data.values():
        df["Volatility"] = (df["High"] - df["Low"]) / df["Open"]


def fill_zero_volumes(
    all_data: dict[str, pd.DataFrame],
    n_same_weekday: int = 10,
    vol_weight: float = 0.7,
) -> None:
    """
    Impute `Volume == 0` rows.

    For each zero-volume row:
        baseline = median of the last `n_same_weekday` non-zero volumes
                   on the same weekday (earlier rows only);
                   if there are none, the median of the last
                   `n_same_weekday` non-zero volumes on any day
        volume   = baseline * max(0.01, 1 + vol_weight * z)

    where `z` is that day's `Volatility` z-score, using the mean and standard
    deviation over the symbol's history. Rows with no earlier non-zero volume
    are left at 0.

    Requires the columns 'Weekday', 'Volume' and 'Volatility'.
    `Volume` is converted to float. Modifies the DataFrames in place.
    """
    for symbol, df in all_data.items():
        for col in ["Weekday", "Volume", "Volatility"]:
            if col not in df.columns:
                raise ValueError(f"{symbol}: missing column '{col}'")

        df["Volume"] = df["Volume"].astype(float)

        vol_mean = df["Volatility"].mean()
        vol_std = df["Volatility"].std()
        if vol_std == 0 or np.isnan(vol_std):
            vol_z = pd.Series(0.0, index=df.index)
        else:
            vol_z = (df["Volatility"] - vol_mean) / vol_std

        for idx in df.index[df["Volume"] == 0]:
            past = df.iloc[: df.index.get_loc(idx)]  # strictly earlier rows
            past_nonzero = past[past["Volume"] > 0]

            same_weekday = past_nonzero.loc[past_nonzero["Weekday"] == df.at[idx, "Weekday"], "Volume"]
            base_source = same_weekday if len(same_weekday) > 0 else past_nonzero["Volume"]
            if len(base_source) == 0:
                continue  # no earlier information: leave at 0

            baseline = base_source.tail(n_same_weekday).median()
            adj = max(0.01, 1 + vol_z.at[idx] * vol_weight)
            df.at[idx, "Volume"] = baseline * adj
