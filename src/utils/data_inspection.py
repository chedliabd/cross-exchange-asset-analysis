import pandas as pd

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _require_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def invalids_summary(df: pd.DataFrame):
    _require_columns(df)
    summary = {
        "Negative volumes": int((df["Volume"] < 0).sum()),
        "Open price <= 0":  int((df["Open"]   <= 0).sum()),
        "Close price <= 0": int((df["Close"]  <= 0).sum()),
        "Adj Close <= 0":   int((df["Adj Close"] <= 0).sum()),
        "High <= 0":        int((df["High"]   <= 0).sum()),
        "Low <= 0":         int((df["Low"]    <= 0).sum()),
        "Duplicates":       int(df.duplicated().sum()),
        "Missing Open":     int(df["Open"].isna().sum()),
        "Missing Close":    int(df["Close"].isna().sum()),
        "Missing Adj Close":int(df["Adj Close"].isna().sum()),
        "Missing High":     int(df["High"].isna().sum()),
        "Missing Low":      int(df["Low"].isna().sum()),
        "Missing Volume":   int(df["Volume"].isna().sum()),
    }
    out = pd.DataFrame([summary])
    out.index = ["TOTAL"]
    return out
