# src/utils/data_inspection.py

import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _require_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def invalids_by_symbol(df: pd.DataFrame, symbol_col: str = "Symbol") -> pd.DataFrame:
    """
    Returns a table counting invalidities per symbol.

    Columns:
      - volume_lt_0
      - open_le_0
      - close_le_0
      - adj_close_le_0
      - high_le_0
      - low_le_0
      - duplicates
    """
    _require_columns(df)

    # Boolean masks for each invalidity
    m_volume_lt0   = df["Volume"] < 0
    m_open_le0     = df["Open"] <= 0
    m_close_le0    = df["Close"] <= 0
    m_adj_le0      = df["Adj Close"] <= 0
    m_high_le0     = df["High"] <= 0
    m_low_le0      = df["Low"] <= 0

    # Detect duplicates (True for duplicates)
    m_dups = df.duplicated()

    invalid_df = pd.DataFrame({
        "volume_lt_0":    m_volume_lt0.astype(int),
        "open_le_0":      m_open_le0.astype(int),
        "close_le_0":     m_close_le0.astype(int),
        "adj_close_le_0": m_adj_le0.astype(int),
        "high_le_0":      m_high_le0.astype(int),
        "low_le_0":       m_low_le0.astype(int),
        "duplicates":     m_dups.astype(int),
    })

    if symbol_col in df.columns:
        # Group by symbol and sum invalid counts
        out = invalid_df.groupby(df[symbol_col], dropna=False).sum()
        out.index.name = symbol_col
        return out

    # If no symbol column, return total for all rows
    totals = invalid_df.sum().to_frame().T
    totals.index = ["ALL"]
    return totals
