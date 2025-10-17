# src/main.py
from pathlib import Path
from src.loading_data import (
    load_stocks, load_etfs, read_metadata,
    print_basic_info, describe_symbols, describe_missing
)

print(">>> main.py STARTED")  # prove we’re running
print(f"__file__ = {__file__}")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STOCKS_DIR = DATA / "stocks"
ETFS_DIR   = DATA / "etfs"
META_FILE  = DATA / "symbols_valid_meta.csv"

print(f"ROOT      = {ROOT}")
print(f"STOCKS_DIR= {STOCKS_DIR} (exists={STOCKS_DIR.exists()})")
print(f"ETFS_DIR  = {ETFS_DIR} (exists={ETFS_DIR.exists()})")
print(f"META_FILE = {META_FILE} (exists={META_FILE.exists()})")

# Count CSVs before loading
stocks_csv = list(STOCKS_DIR.glob("*.csv"))
etfs_csv   = list(ETFS_DIR.glob("*.csv"))
print(f"Found {len(stocks_csv)} stock CSVs, {len(etfs_csv)} ETF CSVs")

try:
    stocks = load_stocks(STOCKS_DIR)
    etfs   = load_etfs(ETFS_DIR)
except Exception as e:
    import traceback; print("LOAD ERROR:"); traceback.print_exc(); raise

print_basic_info("stocks", stocks)
print_basic_info("etfs", etfs)

try:
    meta = read_metadata(META_FILE)
    print("\n=== METADATA ===")
    print(f"shape: {meta.shape}")
    print(f"columns: {list(meta.columns)}")
except Exception as e:
    print(f"\nMetadata read failed: {e}")

print("\nPer-symbol coverage (stocks):")
print(describe_symbols(stocks, top_n=10).to_string(index=False))

overall_miss, by_symbol_miss = describe_missing(stocks)
print("\nOverall missing ratios (stocks):")
print(overall_miss.to_string(index=False))
print("\nPer-symbol missing % (stocks) — top 10:")
print(by_symbol_miss.head(10).to_string(index=False))

print("\n>>> main.py FINISHED")
