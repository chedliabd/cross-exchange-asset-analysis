"""
Metadata helpers for `preparing_metadata.ipynb`.

- NASDAQ symbol directory: load `nasdaqlisted.txt` and fill missing
  `market_category` / `financial_status`.
- Readable labels for the exchange, market category and financial status codes.
- Sector and industry from Yahoo Finance, fetched once into a resumable cache,
  with a check that each ticker still belongs to the same company.
"""

from __future__ import annotations

import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

# ---------- Code definitions ----------
# Source: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs

LISTING_EXCHANGE_MAP = {
    "Q": "NASDAQ",
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Z": "Cboe BZX (BATS)",
    "V": "IEX",
}

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
    "G": "Deficient & Bankrupt",
    "H": "Deficient & Delinquent",
    "J": "Delinquent & Bankrupt",
    "K": "Deficient, Delinquent & Bankrupt",
}


# ---------- NASDAQ symbol directory ----------

def drop_cqs_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the metadata without the 'cqs_symbol' column."""
    return df.drop(columns=["cqs_symbol"], errors="ignore")


def load_nasdaq_symbol_directory(path: str | Path) -> pd.DataFrame:
    """
    Load NASDAQ Trader 'nasdaqlisted.txt' (pipe-delimited).
    Returns a normalized DataFrame with snake_case columns.
    """
    df = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False, na_values=[], engine="python")

    for c in df.columns:
        df[c] = df[c].str.strip()

    # Drop empty rows and the "File Creation Time: ..." footer line
    df = df[(df["Symbol"].str.len() > 0) & ~df["Symbol"].str.startswith("File Creation Time")].copy()

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
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)

    df["round_lot_size"] = pd.to_numeric(df.get("round_lot_size"), errors="coerce")

    def _yn(x):
        s = str(x).strip().upper()
        return True if s == "Y" else False if s == "N" else pd.NA

    for c in ("test_issue", "nextshares"):
        if c in df.columns:
            df[c] = df[c].map(_yn).astype("boolean")

    for c in df.columns:
        if c not in ("round_lot_size", "test_issue", "nextshares"):
            df[c] = df[c].replace({"": pd.NA})

    return df


def enrich_market_category_and_fin_status(meta: pd.DataFrame, nasdaq_dir_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing 'market_category' and 'financial_status' in `meta`
    using `nasdaq_dir_df` (join on symbol, case-insensitive).
    Existing non-missing values in `meta` are preserved.
    """
    left = meta.copy()
    left["symbol_key"] = left["symbol"].str.upper()

    right = nasdaq_dir_df.copy()
    right["symbol_key"] = right["symbol"].str.upper()
    right = right[["symbol_key", "market_category", "financial_status"]].drop_duplicates("symbol_key")

    merged = left.merge(right, on="symbol_key", how="left", suffixes=("", "_from_dir"))
    for col in ("market_category", "financial_status"):
        merged[col] = merged[col].astype("string").fillna(merged[f"{col}_from_dir"].astype("string"))
        merged = merged.drop(columns=[f"{col}_from_dir"])

    return merged.drop(columns=["symbol_key"])


def add_readable_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable columns:
      - listing_exchange_label  (e.g. "NASDAQ", "NYSE")
      - market_category_label   (NASDAQ tier, only for NASDAQ-listed symbols)
      - financial_status_label
    """
    out = df.copy()
    out["listing_exchange_label"] = out["listing_exchange"].map(LISTING_EXCHANGE_MAP)
    out["market_category_label"] = out["market_category"].map(MARKET_CATEGORY_MAP)
    out["financial_status_label"] = out["financial_status"].map(FINANCIAL_STATUS_MAP)
    return out


# ---------- Diagnostics ----------

def missing_values_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count and percentage per column."""
    missing = df.isna().sum()
    return pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": (missing / len(df) * 100).round(2),
    })


def exchange_missing_breakdown(meta: pd.DataFrame, by: str = "listing_exchange") -> pd.DataFrame:
    """For each exchange, the number of symbols and of missing market_category / financial_status."""
    return (
        meta.assign(
            mc_missing=meta["market_category"].isna(),
            fs_missing=meta["financial_status"].isna(),
        )
        .groupby(by, dropna=False)
        .agg(rows=("mc_missing", "size"), mc_missing=("mc_missing", "sum"), fs_missing=("fs_missing", "sum"))
        .sort_values("rows", ascending=False)
    )


# ---------- Sector & industry (Yahoo Finance) ----------

PROFILE_COLUMNS = ["symbol", "yf_name", "Sector", "Industry", "status", "fetched_on"]


def to_yahoo_symbol(symbol: str) -> str:
    """Kaggle/NASDAQ ticker -> Yahoo ticker (share classes use '-' on Yahoo: BRK.B -> BRK-B)."""
    return symbol.replace(".", "-")


def fetch_yfinance_profiles(
    symbols,
    cache_csv: str | Path,
    pause: float = 0.2,
    save_every: int = 100,
) -> pd.DataFrame:
    """
    Fetch the company name, sector and industry of each symbol from Yahoo Finance.

    Results are cached in `cache_csv` (columns: PROFILE_COLUMNS). The function is
    resumable: symbols already in the cache are skipped (failed lookups are
    retried), and progress is saved every `save_every` symbols. Delete the cache
    to fetch everything again.

    `status` is "ok", "no data" (Yahoo returned nothing) or "error: ...".
    `fetched_on` records the date of the lookup: Yahoo returns the CURRENT
    profile of a ticker, not the one at the time of the price data.
    """
    import yfinance as yf
    from tqdm import tqdm

    cache_csv = Path(cache_csv)
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv, dtype=str)
        # failed lookups (network errors, rate limits) are retried on the next run
        cached = cached[~cached["status"].fillna("").str.startswith("error")]
        rows = cached.to_dict("records")
        done = set(cached["symbol"])
    else:
        rows, done = [], set()

    todo = [s for s in dict.fromkeys(symbols) if s not in done]
    print(f"{len(done)} symbols already cached, {len(todo)} to fetch")

    def _save():
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=PROFILE_COLUMNS).to_csv(cache_csv, index=False)

    today = date.today().isoformat()
    for i, symbol in enumerate(tqdm(todo, desc="Fetching Yahoo profiles", ncols=100), start=1):
        try:
            info = yf.Ticker(to_yahoo_symbol(symbol)).info or {}
            name = info.get("longName") or info.get("shortName")
            status = "ok" if (name or info.get("sector")) else "no data"
            rows.append({
                "symbol": symbol, "yf_name": name, "Sector": info.get("sector"),
                "Industry": info.get("industry"), "status": status, "fetched_on": today,
            })
        except Exception as e:  # network errors, unknown tickers, rate limits
            rows.append({
                "symbol": symbol, "yf_name": None, "Sector": None, "Industry": None,
                "status": f"error: {str(e)[:60]}", "fetched_on": today,
            })
        if i % save_every == 0:
            _save()
        time.sleep(pause)

    _save()
    return pd.DataFrame(rows, columns=PROFILE_COLUMNS)


_NAME_STOPWORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited",
    "plc", "llc", "lp", "holdings", "holding", "group", "the", "class", "common", "stock", "shares",
    "share", "ordinary", "depositary", "depository", "american", "ads", "adr", "sa", "ag", "nv",
    "se", "de", "of", "and", "new", "units", "unit", "warrants", "warrant", "rights", "series",
    "preferred", "par", "value",
}


def _name_tokens(name) -> set[str]:
    """Significant words of a company name ("Apple Inc. - Common Stock" -> {"apple"})."""
    if not isinstance(name, str):
        return set()
    name = name.split(" - ")[0].lower().replace("&", "")
    words = re.sub(r"[^a-z0-9 ]", " ", name).split()
    return {w for w in words if len(w) > 1 and w not in _NAME_STOPWORDS}


def names_match(name_a, name_b) -> bool:
    """True if the two company names share at least one significant word."""
    return bool(_name_tokens(name_a) & _name_tokens(name_b))


def add_sector_industry(meta: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Yahoo's Sector / Industry into the metadata.

    A Yahoo profile is only trusted if its company name matches the
    `security_name` of the 2020 metadata (`yf_name_match`). Otherwise the ticker
    may have been reused by another company since 2020, so Sector and Industry
    are left missing.
    """
    prof = profiles.drop_duplicates("symbol")[["symbol", "yf_name", "Sector", "Industry", "fetched_on"]]
    out = meta.drop(columns=["Sector", "Industry"], errors="ignore").merge(prof, on="symbol", how="left")

    # Yahoo sometimes returns an empty string instead of no sector
    out[["Sector", "Industry"]] = out[["Sector", "Industry"]].replace(r"^\s*$", pd.NA, regex=True)

    out["yf_name_match"] = [names_match(a, b) for a, b in zip(out["security_name"], out["yf_name"])]
    out.loc[~out["yf_name_match"], ["Sector", "Industry"]] = pd.NA
    return out


def sector_coverage(meta: pd.DataFrame, by: str = "listing_exchange_label") -> pd.DataFrame:
    """Number of stocks (non-ETF) per group, and how many have a sector."""
    stocks = meta[~meta["etf"].fillna(False).astype(bool)]
    out = stocks.groupby(by, dropna=False).agg(
        stocks=("symbol", "size"),
        with_sector=("Sector", lambda s: s.notna().sum()),
    )
    out.loc["Total"] = out.sum()
    out["coverage_%"] = (out["with_sector"] / out["stocks"] * 100).round(1)
    return out
