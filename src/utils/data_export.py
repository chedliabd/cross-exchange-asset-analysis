"""
Export helper shared by the preparation notebooks.

Every prepared dataset is saved exactly once, as a parquet file in `data_processed/`.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data_processed")


def export_clean_parquet(
    data_dict: dict[str, pd.DataFrame],
    parquet_name: str,
    date_col: str = "Date",
    processed_dir: str | Path = PROCESSED_DIR,
) -> pd.DataFrame | None:
    """
    Concatenate {symbol: DataFrame} into one long table sorted by (Symbol, Date)
    and save it to `{processed_dir}/{parquet_name}`.

    Each DataFrame must already have a `Symbol` column (added at load time).
    """
    if not data_dict:
        print(f"No data for {parquet_name}, nothing saved.")
        return None

    df_all = pd.concat(data_dict.values(), ignore_index=True)
    df_all[date_col] = pd.to_datetime(df_all[date_col])
    df_all = df_all.sort_values(["Symbol", date_col]).reset_index(drop=True)

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / parquet_name
    df_all.to_parquet(out_path, index=False)

    print(f"Saved {out_path}: {df_all['Symbol'].nunique()} symbols, {len(df_all):,} rows")
    return df_all
