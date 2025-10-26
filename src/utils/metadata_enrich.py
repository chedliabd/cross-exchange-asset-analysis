# src/utils/metadata_enrich.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# ---------- Loaders ----------

def load_nasdaq_symbol_directory(path: str | Path) -> pd.DataFrame:
    """
    Load NASDAQ Trader 'nasdaqlisted.txt' (pipe-delimited).
    Returns a normalized DataFrame with snake_case columns.
    """
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        engine="python"
    )

    # Remove footer rows (keep rows with non-empty 'Symbol')
    if "Symbol" in df.columns:
        df = df[df["Symbol"].str.len() > 0].copy()

    # Trim
    for c in df.columns:
        df[c] = df[c].str.strip()

    # Select + rename to snake_case
    keep = {
        "Symbol": "symbol",
        "Security Name": "security_name",
        "Market Category": "market_category",
        "Test Issue": "test_issue",
        "Financial Status": "financial_status",
        "Round Lot Size": "round_lot_size",
        "NASDAQ Symbol": "nasdaq_symbol",
        "NextShares": "nextshares",
    }
    present = [c for c in keep.keys() if c in df.columns]
    df = df[present].rename(columns=keep)

    # Types
    df["round_lot_size"] = pd.to_numeric(df.get("round_lot_size"), errors="coerce")

    def _yn(x: str):
        if x is None: return pd.NA
        s = str(x).strip().upper()
        if s == "Y": return True
        if s == "N": return False
        return pd.NA

    for c in ("test_issue", "nextshares"):
        if c in df.columns:
            df[c] = df[c].map(_yn).astype("boolean")

    # Empty strings -> NA
    for c in df.columns:
        if c not in ("round_lot_size", "test_issue", "nextshares"):
            df[c] = df[c].replace({"": pd.NA})

    return df

# ---------- Enrichment ----------

def enrich_market_category_and_fin_status(meta: pd.DataFrame,
                                          nasdaq_dir_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'market_category' and 'financial_status' in `meta`
    using `nasdaq_dir_df` (join on symbol, case-insensitive).
    Existing non-missing values in `meta` are preserved.
    """
    left = meta.copy()
    left["symbol_key"] = left["symbol"].str.upper()

    right = nasdaq_dir_df.copy()
    right["symbol_key"] = right["symbol"].str.upper()

    right_small = right[["symbol_key", "market_category", "financial_status"]]

    merged = left.merge(right_small, on="symbol_key", how="left", suffixes=("", "_from_dir"))

    for col in ("market_category", "financial_status"):
        merged[col] = merged[col].where(~merged[col].isna(), merged[f"{col}_from_dir"])
        merged = merged.drop(columns=[f"{col}_from_dir"])

    return merged.drop(columns=["symbol_key"])

# ---------- Readable labels (optional but handy) ----------

MARKET_CATEGORY_MAP = {
    "Q": "NASDAQ Global Select Market",
    "G": "NASDAQ Global Market",
    "S": "NASDAQ Capital Market",
}

FINANCIAL_STATUS_MAP = {
    "N": "Normal",
    "D": "Deficient",
    "E": "Delinquent",
    "Q": "Bankrupt",
    "G": "Deficient & Delinquent",
    "H": "Deficient & Bankrupt",
    "J": "Delinquent & Bankrupt",
    "K": "Deficient, Delinquent & Bankrupt",
}

def add_readable_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable columns:
      - market_category_label
      - financial_status_label
    """
    out = df.copy()
    out["market_category_label"] = out["market_category"].map(MARKET_CATEGORY_MAP)
    out["financial_status_label"] = out["financial_status"].map(FINANCIAL_STATUS_MAP)
    return out

# ---------- Diagnostics ----------

def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple missingness per column (count and percentage).
    """
    total = len(df)
    counts = df.isna().sum()
    return pd.DataFrame({
        "column": counts.index,
        "missing_count": counts.values,
        "missing_percentage": (counts.values / total * 100).round(2)
    })

def exchange_missing_breakdown(meta: pd.DataFrame) -> pd.DataFrame:
    """
    For each listing_exchange, show missing counts of market_category and financial_status.
    """
    cols = ["listing_exchange", "market_category", "financial_status"]
    existing = [c for c in cols if c in meta.columns]
    df = meta[existing].copy()
    return (df.assign(
                mc_missing=df["market_category"].isna(),
                fs_missing=df["financial_status"].isna())
              .groupby("listing_exchange", dropna=False)
              .agg(mc_missing=("mc_missing", "sum"),
                   fs_missing=("fs_missing", "sum"),
                   rows=("mc_missing", "size"))
              .reset_index()
              .sort_values("rows", ascending=False))
