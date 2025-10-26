# src/utils/loading_data.py

from pathlib import Path
import pandas as pd

def load_symbol_csv(csv_path: str | Path, *, symbol: str | None = None, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load exactly ONE symbol from ONE CSV file. No cleaning, no coercion, no drops.

    - If `symbol` is provided, a 'Symbol' column is added/overwritten with that value.
    - Otherwise, 'Symbol' is taken from the filename (e.g., AAPL.csv -> AAPL).
    - If `parse_dates=True`, tries to parse a 'Date' column to datetime (errors='coerce').
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    df = pd.read_csv(p)

    if symbol is None:
        symbol = p.stem
    df["Symbol"] = symbol

    if parse_dates and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def load_symbol_from_dir(root_dir: str | Path, symbol: str, *, ext: str = ".csv", parse_dates: bool = True) -> pd.DataFrame:
    """
    Convenience: load `{root_dir}/{symbol}{ext}`.
    """
    return load_symbol_csv(Path(root_dir) / f"{symbol}{ext}", symbol=symbol, parse_dates=parse_dates)


def iter_symbols(root_dir: str | Path, *, ext: str = ".csv", parse_dates: bool = True):
    """
    Iterate all CSVs in a folder. Yields (symbol, df) with NO cleaning.
    """
    root = Path(root_dir)
    for f in sorted(root.glob(f"*{ext}")):
        sym = f.stem
        yield sym, load_symbol_csv(f, symbol=sym, parse_dates=parse_dates)
        
        
# src/utils/metadata.py
import pandas as pd

_METADATA_COLS = [
    "Nasdaq Traded","Symbol","Security Name","Listing Exchange","Market Category",
    "ETF","Round Lot Size","Test Issue","Financial Status","CQS Symbol",
    "NASDAQ Symbol","NextShares"
]

_YN_COLS = ["Nasdaq Traded", "ETF", "Test Issue", "NextShares"]

def _yn_to_bool(s):
    if pd.isna(s):
        return pd.NA
    s = str(s).strip().upper()
    if s == "Y": return True
    if s == "N": return False
    return pd.NA

def load_nasdaq_metadata(csv_path, *, normalize_columns=True):
    """
    Load the NASDAQ metadata file.

    Expected columns:
      Nasdaq Traded, Symbol, Security Name, Listing Exchange, Market Category,
      ETF, Round Lot Size, Test Issue, Financial Status, CQS Symbol,
      NASDAQ Symbol, NextShares

    Example row (as provided):
      Y, AACG, "ATA Creativity Global - American Depositary Shares, each representing two common shares",
      Q, G, N, 100.0, N, N, , AACG, N
    """
    # Load everything as string first to be safe; we’ll coerce types after.
    df = pd.read_csv(
        csv_path,
        usecols=_METADATA_COLS,
        dtype=str,
        keep_default_na=False,   # keep empty fields as "", then convert to NA below
        na_values=[]
    )

    # Strip whitespace
    for c in df.columns:
        df[c] = df[c].str.strip()

    # Convert Y/N columns → boolean
    for c in _YN_COLS:
        df[c] = df[c].map(_yn_to_bool).astype("boolean")

    # Round Lot Size → float (nullable)
    df["Round Lot Size"] = pd.to_numeric(df["Round Lot Size"], errors="coerce")

    # Empty strings → NA for the remaining text columns
    text_cols = [c for c in df.columns if c not in _YN_COLS + ["Round Lot Size"]]
    for c in text_cols:
        df[c] = df[c].replace({"": pd.NA})

    if normalize_columns:
        df = df.rename(columns={
            "Nasdaq Traded": "nasdaq_traded",
            "Symbol": "symbol",
            "Security Name": "security_name",
            "Listing Exchange": "listing_exchange",
            "Market Category": "market_category",
            "ETF": "etf",
            "Round Lot Size": "round_lot_size",
            "Test Issue": "test_issue",
            "Financial Status": "financial_status",
            "CQS Symbol": "cqs_symbol",
            "NASDAQ Symbol": "nasdaq_symbol",
            "NextShares": "nextshares",
        })

    return df

