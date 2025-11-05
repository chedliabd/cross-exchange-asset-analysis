# src/utils/metadata_enrich.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import yfinance as yf
import os
from tqdm import tqdm
import numpy as np
import time




def drop_cqs_symbol(df: pd.DataFrame) :
    """
    Drop the 'cqs_symbol' column from the metadata DataFrame if it exists.

    Returns a new DataFrame (does not modify the input in place).
    """
    if "cqs_symbol" in df.columns:
        df = df.drop(columns=["cqs_symbol"])
    return df

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

def missing_values_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing (NaN) values per column in the metadata DataFrame.

    Columns:
      - column: name of the column
      - missing_count: number of missing (NaN) entries
      - missing_percentage: % of missing values relative to total rows
    """
    total_rows = len(df)
    missing_counts = df.isna().sum()
    missing_perc = (missing_counts / total_rows) * 100

    summary = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_percentage": missing_perc.round(2)
    })

    return summary

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
    
    
def check_valid_securities(data_dir, output_csv):
    
    # load exsiting results if exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_tickers = set(existing_df["Ticker"])
        results = existing_df.to_dict("records")
        print(f"Resuming... {len(processed_tickers)} tickers already processed.")
    else:
        processed_tickers = set()
        results = []
    
    # process new files
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    new_files = [f for f in all_files if os.path.splitext(f)[0] not in processed_tickers]
    
    valid_count = sum(1 for r in results if r["Status"] == "Valid")
    total_count = len(all_files)
    
    for file in tqdm(new_files, desc="Checking tickers", ncols=100):
        ticker = os.path.splitext(file)[0]
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                status = "Valid"
                valid_count += 1
            else:
                status = "Invalid or no data"
        except Exception as e:
            status = f"Error: {str(e)[:60]}"
        results.append({"Ticker": ticker, "Status": status})
    
    # save updated results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # summary
    percent = (valid_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Total tickers: {total_count}")
    print(f"Valid tickers: {valid_count}")
    print(f"Yahoo Finance coverage: {percent:.2f}%")
    print(f"Results saved to: {output_csv}")

def create_update_sector_csv(valid_securities, output_csv):
    
    # load valid tickers
    df = pd.read_csv(valid_securities)
    valid_tickers = df[df["Status"] == "Valid"]["Ticker"].tolist()
    print(f" Found {len(valid_tickers)} valid tickers to enrich with sector data.")
    
    # load existing
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        processed = set(existing["Ticker"])
        results = existing.to_dict("records")
        print(f"🔁 Resuming... {len(processed)} already processed.")
    else:
        processed = set()
        results = []
    
    # fetch sectors
    for ticker in tqdm(valid_tickers, desc="Fetching sector info", ncols=100):
        if ticker in processed:
            continue
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", None)
            industry = info.get("industry", None)
            results.append({"Ticker": ticker, "Sector": sector, "Industry": industry})
            time.sleep(0.2)
        except Exception as e:
            results.append({"Ticker": ticker, "Sector": None, "Industry": None})
    
    # save updated results
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    
    print(f"\nEnriched data saved to {output_csv}")
    print(out_df["Sector"].value_counts(dropna=False).head(10))

def summarize_sector_market_distribution(csv_path):
    """
    Summarize counts and percentages of stocks per (Sector, Market Category)
    from a metadata CSV file.

    Args:
        csv_path (str): Path to the metadata CSV file.
    
    Returns:
        counts (DataFrame): Raw counts per (Sector, Market Category)
        percents (DataFrame): Normalized percentages (row-wise)
    """
    # --- LOAD CSV ---
    meta_df = pd.read_csv(csv_path)

    # --- STANDARDIZE COLUMN NAMES (in case of case differences) ---
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]

    if not {"Sector", "market_category"}.issubset(meta_df.columns):
        raise ValueError("CSV must contain 'sector' and 'market_category' columns.")

    # --- GROUP AND SUMMARIZE ---
    counts = (
        meta_df.groupby(["sector", "market_category"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    return counts, percents


def filter_metadata_by_valid_tickers(
    metadata_path,
    valid_tickers_path,
    output_path="nasdaq_metadata_filtered.csv"
):
    """
    Filter the NASDAQ metadata file to include only securities whose tickers
    are marked as 'Valid' in the ticker validation results.

    Args:
        metadata_path (str): Path to the original NASDAQ metadata CSV.
        valid_tickers_path (str): Path to the CSV with validation results.
        output_path (str, optional): Path to save the filtered metadata CSV.

    Returns:
        pd.DataFrame: Filtered metadata DataFrame.
    """
    # load files
    meta = pd.read_csv(metadata_path)
    valid = pd.read_csv(valid_tickers_path)

    # clean and filter
    valid_tickers = set(
        valid[valid["Status"] == "Valid"]["Ticker"].astype(str).str.upper()
    )

    meta["Symbol"] = meta["Symbol"].astype(str).str.upper()
    filtered = meta[meta["Symbol"].isin(valid_tickers)].copy()

    # save result
    filtered.to_csv(output_path, index=False)

    # summary
    print(f"Filtered metadata saved to: {output_path}")
    print(f"Original rows: {len(meta)}")
    print(f"Kept: {len(filtered)} ({len(filtered) / len(meta) * 100:.2f}%)")
    print(f"Removed: {len(meta) - len(filtered)}")

    return filtered

import pandas as pd

def enrich_metadata_with_sectors(metadata_path, sectors_path, output_path, ticker_col="Ticker"):
    """
    Enrich a metadata CSV with sector information based on ticker matches.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file.
    sectors_path : str
        Path to the sectors CSV file.
    output_path : str
        Path to save the enriched metadata CSV.
    ticker_col : str, optional
        Column name for ticker symbols (default: "ticker").
    """

    # Load both CSVs
    metadata = pd.read_csv(metadata_path)
    sectors = pd.read_csv(sectors_path)

    # Merge on ticker column
    enriched = metadata.merge(sectors, on=ticker_col, how="left")

    # Save to file
    enriched.to_csv(output_path, index=False)
    print(f"Enriched metadata saved to {output_path}")
    print(f"{enriched['Sector'].isna().sum()} tickers have no matching sector")

    return enriched

import pandas as pd

def add_sector_column(file_main, file_with_sector, output_file=None, fill_missing=None):
    """
    Adds a 'Sector' column to a CSV that has tickers but no sector info.
    
    Parameters:
    ----------
    file_main : str
        Path to the CSV file that contains 'Ticker' but no 'Sector'.
    file_with_sector : str
        Path to the CSV file that contains 'Ticker' and 'Sector'.
    output_file : str, optional
        Path to save the merged file. If None, the merged DataFrame is returned.
    fill_missing : str or None, default "Unknown"
        Value to fill missing sectors. Use None to keep NaN.
    
    Returns:
    -------
    pandas.DataFrame
        The merged DataFrame (also saved to CSV if output_file is provided).
    """
    
    # Read CSVs
    df_main = pd.read_csv(file_main, dtype={"symbol": str})
    df_sec = pd.read_csv(file_with_sector, dtype={"symbol": str, "Sector": str})
    
    # Clean ticker formatting
    df_main["symbol"] = df_main["symbol"].str.strip().str.upper()
    df_sec["symbol"] = df_sec["symbol"].str.strip().str.upper()
    
    # Remove duplicates in the sector file
    df_sec = df_sec.drop_duplicates(subset=["symbol"], keep="first")
    
    # Merge on Ticker
    merged = df_main.merge(df_sec[["symbol", "Sector"]], on="symbol", how="left")
    
    # Fill missing values if requested
    if fill_missing is not None:
        merged["Sector"] = merged["Sector"].fillna(fill_missing)
    
    # Save or return
    if output_file:
        merged.to_csv(output_file, index=False)
        print(f"Saved merged file to: {output_file}")
    else:
        return merged


def summarize_sector_market_distribution(csv_path):
    """
    Summarize counts and percentages of stocks per (Sector, Market Category)
    from a metadata CSV file.

    Args:
        csv_path (str): Path to the metadata CSV file.
    
    Returns:
        counts (DataFrame): Raw counts per (Sector, Market Category)
        percents (DataFrame): Normalized percentages (row-wise)
    """
    # --- LOAD CSV ---
    meta_df = pd.read_csv(csv_path)

    if not {"Sector", "market_category_label"}.issubset(meta_df.columns):
        raise ValueError("CSV must contain 'sector' and 'market_category' columns.")

    # --- GROUP AND SUMMARIZE ---
    counts = (
        meta_df.groupby(["Sector", "market_category_label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    return counts, percents

# Calculating log returns
def compute_log_return(df: pd.DataFrame, symbol) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")

    # Replace zeros or negatives with previous valid value (forward fill)
    df.loc[df["Adj Close"] <= 0, "Adj Close"] = np.nan
    df["Adj Close"] = df["Adj Close"].ffill()

    # Compute log returns
    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)

    # Print a small sample for inspection
    print(f"\n✅ {symbol}: computed log returns for {len(df)} rows")
    print(df[["Date", "Adj Close", "lag_adj_close", "log_return"]].head(5))
    
    return df

# Calculating volatility
def compute_volatility(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of log returns).
    Default window: 20 trading days (~1 month)
    """
    df = df.sort_values("Date").copy()

    if "log_return" not in df.columns:
        raise ValueError("log_return column not found. Compute it first.")

    df["volatility"] = df["log_return"].rolling(window=window, min_periods=30).std()

    print(f"✅ Computed rolling volatility with window={window}")
    print(df[["Date", "log_return", "volatility"]].head(10))

    return df

