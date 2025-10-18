import pandas as pd
import numpy as np
from typing import Iterable

def nan_zero_report(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Return a per-column table with:
      - missing_count, missing_% 
      - zero_count, zero_%   (only computed for numeric columns; others shown as NaN)
    """
    if cols is None:
        cols = df.columns

    n = len(df)
    if n == 0:
        # empty df -> just return zeros/NaNs with proper columns
        return pd.DataFrame({
            "column": list(cols),
            "missing_count": 0,
            "missing_%": 0.0,
            "zero_count": np.nan,
            "zero_%": np.nan,
        })

    # NaNs
    missing_count = df[list(cols)].isna().sum()
    missing_pct   = (missing_count / n * 100).round(2)

    # Zeros (numeric only)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    zero_count_series = pd.Series(np.nan, index=cols, dtype="float")
    zero_pct_series   = pd.Series(np.nan, index=cols, dtype="float")
    if num_cols:
        zero_counts = (df[num_cols] == 0).sum()
        zero_pcts   = (zero_counts / n * 100).round(2)
        zero_count_series.loc[num_cols] = zero_counts.astype(float)
        zero_pct_series.loc[num_cols]   = zero_pcts.astype(float)

    out = pd.DataFrame({
        "column": list(cols),
        "missing_count": missing_count.values,
        "missing_%": missing_pct.values,
        "zero_count": zero_count_series.values,
        "zero_%": zero_pct_series.values,
    })
    return out.sort_values(["missing_%", "zero_%"], ascending=False, na_position="last").reset_index(drop=True)


def check_duplicates(df: pd.DataFrame):
    """Return number of duplicated rows."""
    return df.duplicated().sum()

def check_negative_prices(df: pd.DataFrame):
    """Detect rows with strictly negative price or volume values."""
    # Check for any strictly negative price in the key price columns
    bad_price = (df[["Open", "High", "Low", "Close", "Adj Close"]] < 0).any(axis=1)   
    # Combine both conditions
    invalid = df[bad_price ]

    return invalid


