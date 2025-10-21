# data_cleaning.py
import pandas as pd
from typing import Dict, Any, Optional

# Reuse validations from your inspection module (no redundancy)
from src.utils.data_inspection import _require_columns, REQUIRED_COLS

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "Symbol" in df.columns and "Date" in df.columns:
        return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    return df

def _rate_open_le_zero_by_symbol(df: pd.DataFrame) -> pd.Series:
    """Internal fallback if you don't pass precomputed rates."""
    total = df.groupby("Symbol").size()
    bad = df[df["Open"] <= 0].groupby("Symbol").size()
    return (bad / total * 100).fillna(0)

def _fill_adj_close_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Adj Close <= 0, try to replace with mean(prev, next) in the SAME symbol.
    If either neighbor is missing or <= 0, we mark it for drop later.
    """
    g = g.copy()
    bad_mask = g["Adj Close"] <= 0

    # neighbors within the symbol (Date sorted already)
    prev_adj = g["Adj Close"].shift(1)
    next_adj = g["Adj Close"].shift(-1)

    can_fill = bad_mask & (prev_adj > 0) & (next_adj > 0)
    g.loc[can_fill, "Adj Close"] = (prev_adj[can_fill] + next_adj[can_fill]) / 2.0

    # mark rows that are still bad (cannot be filled)
    g["_drop_bad_adj_close"] = bad_mask & ~can_fill
    return g

def _fix_open_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Open <= 0, try to replace with previous Close (>0) within the SAME symbol.
    If not possible, mark for drop.
    """
    g = g.copy()
    bad_mask = g["Open"] <= 0
    prev_close = g["Close"].shift(1)
    can_fill = bad_mask & (prev_close > 0)
    g.loc[can_fill, "Open"] = prev_close[can_fill]
    g["_drop_bad_open"] = bad_mask & ~can_fill
    return g

def clean_market_dataframe(
    df: pd.DataFrame,
    open_rate_threshold: float,
    *,
    precomputed_open_rates: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Apply the cleaning rules:
      1) Drop rows with NaN in any of: Open, Close, Adj Close, High, Low, Volume.
      2) Drop rows where High <= 0 OR Low <= 0.
      3) For Adj Close <= 0: replace by mean(prev, next) within symbol; if impossible -> drop those rows.
      4) For Open <= 0:
         - If a symbol's rate(Open<=0) > threshold -> drop the whole symbol.
         - Else try to replace each bad Open by previous Close (>0) within symbol; if impossible -> drop that row.

    Parameters
    ----------
    df : DataFrame with at least columns: Symbol, Date, Open, High, Low, Close, Adj Close, Volume
    open_rate_threshold : float (e.g., 18.4 for your 95th percentile)
    precomputed_open_rates : optional Series indexed by Symbol (from analyze_open_zero_distribution)

    Returns
    -------
    dict with:
      'df'               : cleaned DataFrame
      'report'           : dict of counts
      'dropped_symbols'  : list of symbols removed by threshold
    """
    _require_columns(df)
    if "Symbol" not in df.columns or "Date" not in df.columns:
        raise ValueError("DataFrame must contain 'Symbol' and 'Date' columns.")
    df0 = df.copy()
    df0 = _ensure_sorted(df0)

    report: Dict[str, int] = {}

    # 1) Drop rows with NaNs in core columns
    before = len(df0)
    df0 = df0.dropna(subset=REQUIRED_COLS)
    report["rows_dropped_nan_core_cols"] = before - len(df0)

    # 2) Drop rows where High <= 0 OR Low <= 0
    before = len(df0)
    df0 = df0[(df0["High"] > 0) & (df0["Low"] > 0)]
    report["rows_dropped_highlow_le_zero"] = before - len(df0)

    # 3) Fix Adj Close <= 0 via neighbor mean; drop remaining unfixables
    before = len(df0)
    df0 = (
        df0.groupby("Symbol", group_keys=False)
           .apply(_fill_adj_close_per_symbol)
    )
    to_drop_adj = df0["_drop_bad_adj_close"].fillna(False)
    report["rows_imputed_adj_close"] = int(((df0["Adj Close"] <= 0) & ~to_drop_adj).sum())  # before drop mark
    df0 = df0[~to_drop_adj].drop(columns=["_drop_bad_adj_close"])
    report["rows_dropped_unfixable_adj_close"] = before - len(df0) - report["rows_dropped_highlow_le_zero"] - report["rows_dropped_nan_core_cols"]

    # 4) Open rule
    # 4a) Decide symbols to drop by rate
    rates = precomputed_open_rates if precomputed_open_rates is not None else _rate_open_le_zero_by_symbol(df0)
    bad_symbols = rates[rates > open_rate_threshold].index.tolist()
    report["symbols_dropped_open_rate"] = len(bad_symbols)

    df_kept = df0[~df0["Symbol"].isin(bad_symbols)].copy()

    # 4b) For remaining symbols, try to fix Open<=0 using previous Close; drop if still bad
    before = len(df_kept)
    df_kept = (
        df_kept.groupby("Symbol", group_keys=False)
               .apply(_fix_open_per_symbol)
    )
    # count how many we successfully imputed
    imputed_open = int((df_kept["_drop_bad_open"] == False).sum() - (df_kept["Open"] <= 0).sum())
    report["rows_imputed_open_from_prev_close"] = max(imputed_open, 0)

    to_drop_open = df_kept["_drop_bad_open"].fillna(False)
    df_kept = df_kept[~to_drop_open].drop(columns=["_drop_bad_open"])

    report["rows_dropped_unfixable_open"] = before - len(df_kept)

    # Final tidy
    df_clean = _ensure_sorted(df_kept).reset_index(drop=True)

    return {
        "df": df_clean,
        "report": report,
        "dropped_symbols": bad_symbols,
    }
