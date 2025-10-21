import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _require_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def invalids_summary(df: pd.DataFrame):
    _require_columns(df)
    summary = {
        "Negative volumes": int
        ((df["Volume"] < 0).sum()),
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

def rate_open_le_zero_by_symbol(df: pd.DataFrame):
    """
    Display the rate (percentage) of rows where Open <= 0 for each symbol.
    Example output:
        AAME: 12.45%
        APPAL: 3.10%
    """
    _require_columns(df)

    if "Symbol" not in df.columns:
        raise ValueError("Expected a 'Symbol' column in the DataFrame.")

    # Total rows per symbol
    total_counts = df.groupby("Symbol").size()

    # Invalid rows per symbol (Open <= 0)
    bad_counts = df[df["Open"] <= 0].groupby("Symbol").size()

    # Compute rate (%)
    rates = (bad_counts / total_counts * 100).sort_values(ascending=False)

    if rates.empty:
        print("✅ No symbols have Open <= 0 values.")
        return

    # Print nicely formatted rates
    for symbol, rate in rates.items():
        print(f"{symbol}: {rate:.2f}%")
        

def analyze_open_zero_distribution(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """
    Analyze and visualize the rate of invalid Open prices (Open <= 0) per symbol.
    - Computes per-symbol invalid rates (%)
    - Displays a histogram
    - Prints key quantiles and top offenders
    - Suggests a data-driven cutoff (95th percentile)

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least 'Symbol', 'Open' columns.
    dataset_name : str, optional
        Name used for display (e.g., "Stocks" or "ETFs")

    Returns
    -------
    dict
        {
            'rates': Series of per-symbol invalid rates,
            'suggested_cutoff': float (95th percentile value)
        }
    """
    # --- Sanity checks ---
    if "Symbol" not in df.columns or "Open" not in df.columns:
        raise ValueError("DataFrame must contain 'Symbol' and 'Open' columns.")

    # --- Compute rates ---
    total_counts = df.groupby("Symbol").size()
    bad_counts = df[df["Open"] <= 0].groupby("Symbol").size()
    rates = (bad_counts / total_counts * 100).fillna(0)

    # --- Plot histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(rates, bins=50)
    plt.xlabel("Rate of Open ≤ 0 (%)")
    plt.ylabel("Number of symbols")
    plt.title(f"Distribution of invalid Open rates per symbol – {dataset_name}")
    plt.grid(True)
    plt.show()

    # --- Quantiles & cutoff ---
    quantiles = rates.quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    suggested_cutoff = quantiles.loc[0.95]

    print(f"\n📊 Key quantiles of Open≤0 rate for {dataset_name}:")
    print(quantiles)
    print(f"\n📈 Suggested threshold (top 5% worst symbols): {suggested_cutoff:.2f}%")

    # --- Top offenders ---
    print(f"\n🚨 Top 10 symbols with highest invalid Open rates ({dataset_name}):")
    print(rates.sort_values(ascending=False).head(10))

    # --- Return results ---
    return {
        "rates": rates,
        "suggested_cutoff": suggested_cutoff
    }


