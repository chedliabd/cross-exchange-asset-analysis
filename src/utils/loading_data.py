# loading_data.py
# Simple utilities to load stock and ETF CSVs + metadata (Yahoo Finance format).

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def read_one_csv(path, symbol=None):
    """Read one CSV file and return a clean DataFrame with a Symbol column."""
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["Symbol"] = symbol or path.stem
    return df[["Symbol"] + REQUIRED_COLUMNS]

def load_folder(folder):
    """Load all CSVs in a folder (stocks or ETFs) and concatenate them."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["Symbol"] + REQUIRED_COLUMNS)
    frames = [read_one_csv(p, p.stem) for p in files]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

def load_stocks(folder):
    """Shortcut for loading all stock CSVs."""
    return load_folder(folder)

def load_etfs(folder):
    """Shortcut for loading all ETF CSVs."""
    return load_folder(folder)

def read_metadata(path):
    """Load the metadata file (symbols_valid_meta.csv)."""
    meta = pd.read_csv(path)
    if "Symbol" in meta.columns:
        meta["Symbol"] = meta["Symbol"].astype(str).str.strip()
    return meta

def basic_info(name, df):
    """Print simple information about a loaded dataset."""
    print(f"\n{name.upper()} — {len(df):,} rows")
    if "Symbol" in df.columns and not df.empty:
        print(f"Symbols: {df['Symbol'].nunique()}")
        print(f"Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"Columns: {list(df.columns)}")
