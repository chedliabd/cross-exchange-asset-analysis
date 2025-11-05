import pandas as pd
    

def remove_symbols(all_data: dict[str, pd.DataFrame], symbols_to_remove: list[str]):
    """
    Remove specified symbols from the data dictionary.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.
    symbols_to_remove : list[str]
        List of symbols to remove completely.

    Returns
    -------
    dict[str, pd.DataFrame]
        New dictionary with the specified symbols removed.
    """
    return {symbol: df for symbol, df in all_data.items() if symbol not in symbols_to_remove}


def fill_adj_close_nans(all_data: dict[str, pd.DataFrame]):
    for symbol, df in all_data.items():
        if "Adj Close" in df.columns:
            df["Adj Close"] = df["Adj Close"].ffill().bfill()
            
            
            
def fix_open_with_prev_adjclose(
    all_data: dict[str, pd.DataFrame],
    open_col: str = "Open",
    adj_col: str = "Adj Close",
    date_col: str = "Date"
):
    """
    For each symbol:
      - If Open is NaN or 0, replace it with the previous day's Adj Close.
      - If it's the first row or no valid previous Adj Close exists, drop the row.

    Modifies the input dict in place.
    """
    for symbol, df in all_data.items():
        # Require columns
        required = {open_col, adj_col, date_col}
        if not required.issubset(df.columns):
            continue

        # Work on a copy to avoid pandas warnings, then assign back
        tmp = df.copy()

        # Ensure datetime and sort by date
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.sort_values(date_col).reset_index(drop=True)

        # Ensure numeric
        tmp[open_col] = pd.to_numeric(tmp[open_col], errors="coerce")
        tmp[adj_col]  = pd.to_numeric(tmp[adj_col],  errors="coerce")

        # Previous day's Adj Close
        prev_adj = tmp[adj_col].shift(1)

        # Bad Open = NaN or 0
        bad_open = tmp[open_col].isna() | (tmp[open_col] == 0)

        # Fill bad Open with previous day's Adj Close when available
        fillable = bad_open & prev_adj.notna() & (prev_adj != 0)
        tmp.loc[fillable, open_col] = prev_adj[fillable]

        # Drop rows where Open is still NaN or 0 (e.g., first day or no valid prev adj)
        still_bad = tmp[open_col].isna() | (tmp[open_col] == 0)
        if still_bad.any():
            tmp = tmp.loc[~still_bad].reset_index(drop=True)

        # Write back
        all_data[symbol] = tmp
        
        
def fix_close_with_adjclose(
    all_data: dict[str, pd.DataFrame],
    close_col: str = "Close",
    adj_col: str = "Adj Close"
):
    """
    For each symbol:
      - If Close is NaN or 0, replace it with the same day's Adj Close.
      - Assumes Adj Close is already valid and filled.

    Modifies the input dict in place.
    """
    for symbol, df in all_data.items():
        if close_col not in df.columns or adj_col not in df.columns:
            continue

        # Work on a copy to avoid SettingWithCopy warnings
        tmp = df.copy()

        # Ensure numeric columns
        tmp[close_col] = pd.to_numeric(tmp[close_col], errors="coerce")
        tmp[adj_col] = pd.to_numeric(tmp[adj_col], errors="coerce")

        # Identify bad Close values
        bad_close = tmp[close_col].isna() | (tmp[close_col] == 0)

        # Fill them with same-day Adj Close
        tmp.loc[bad_close, close_col] = tmp.loc[bad_close, adj_col]

        # Write back
        all_data[symbol] = tmp


