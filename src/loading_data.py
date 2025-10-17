# loading_data.py
# Minimal utilities to load local CSVs and print quick info.
# No merging. Works for both stocks and ETFs.

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _read_one_price_csv(path: Path, symbol: str | None = None):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Symbol"] = symbol or path.stem
    # keep only expected cols in a clean order
    return df[["Symbol"] + REQUIRED_COLUMNS]

def load_price_folder(folder: str | Path):
    """Read all CSVs in a folder into ONE dataframe (adds 'Symbol' from filename)."""
    folder = Path(folder)
    frames = []
    for p in sorted(folder.glob("*.csv")):
        frames.append(_read_one_price_csv(p, p.stem))
    if not frames:
        # empty frame with the right columns
        return pd.DataFrame(columns=["Symbol"] + REQUIRED_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

def load_stocks(stocks_folder: str | Path) -> pd.DataFrame:
    return load_price_folder(stocks_folder)

def load_etfs(etfs_folder: str | Path) -> pd.DataFrame:
    return load_price_folder(etfs_folder)

def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read the symbols metadata file (no merge)."""
    meta = pd.read_csv(path)
    # light cleanup
    if "Symbol" in meta.columns:
        meta["Symbol"] = meta["Symbol"].astype(str).str.strip()
    return meta

# ---------- Basic info / describe (no plots, no merge) ----------

def print_basic_info(name: str, df: pd.DataFrame):
    """Print shape, columns, date range, and counts."""
    print(f"\n=== {name.upper()} ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    if not df.empty and "Date" in df.columns:
        print(f"date range: {df['Date'].min()}  ->  {df['Date'].max()}")
    if "Symbol" in df.columns:
        print(f"symbols: {df['Symbol'].nunique()}")
        # quick per-symbol row count (top 10)
        counts = (df.groupby('Symbol').size()
                    .sort_values(ascending=False)
                    .head(10))
        print("\nTop 10 symbols by rows:")
        print(counts.to_string())

def describe_symbols(df: pd.DataFrame, top_n: int = 10):
    """Per-symbol coverage (rows + min/max date)."""
    if df.empty:
        return pd.DataFrame(columns=["Symbol", "rows", "date_min", "date_max"])
    g = df.groupby("Symbol", as_index=False).agg(
        rows=("Date", "size"),
        date_min=("Date", "min"),
        date_max=("Date", "max"),
    ).sort_values("rows", ascending=False)
    return g.head(top_n) if top_n else g

def describe_missing(df: pd.DataFrame):
    """Overall missing ratio and per-symbol missing for Adj Close & Volume."""
    if df.empty:
        return pd.DataFrame()
    overall = df.isna().mean().rename("missing_ratio").to_frame().reset_index(names="column")
    by_symbol = (
        df.groupby("Symbol")[["Adj Close", "Volume"]]
          .apply(lambda x: x.isna().mean() * 100)
          .rename(columns={"Adj Close": "AdjClose_missing_%", "Volume": "Volume_missing_%"})
          .reset_index()
          .sort_values("AdjClose_missing_%", ascending=False)
    )
    return overall, by_symbol  # tuple of dataframes (no printing here)
