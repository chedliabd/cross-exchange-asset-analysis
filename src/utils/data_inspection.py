import pandas as pd
import matplotlib.pyplot as plt

def dates_in_increasing_order(df, date_column="Date"):
    """
    Check if the dates in a DataFrame are in increasing order.

    Parameters:
        df (pd.DataFrame): DataFrame containing the date column.
        date_column (str): Name of the column containing dates (default: 'Date').

    Returns:
        bool: True if dates are in increasing order, False otherwise.
    """
    dates = pd.to_datetime(df[date_column])
    return dates.is_monotonic_increasing


def get_date_range(df, date_column="Date"):
    """
    Returns the first and last available dates for a stock DataFrame.
    """
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min(), dates.max()


def show_date_ranges(all_stocks, date_column="Date"):
    """
    Prints the first and last date available for each stock.
    """
    print("\n📅 Date ranges for all stocks:\n")
    for symbol, df in all_stocks.items():
        first, last = get_date_range(df, date_column)
        if first and last:
            print(f"{symbol}: from {first.date()} → {last.date()}")
        else:
            print(f"{symbol}: ⚠️ No valid dates found.")
            
            


def ends_before(all_stocks, cutoff_date="2020-04-01", date_column="Date"):
    """
    Checks which stocks end before a given cutoff date.

    Parameters:
        all_stocks (dict): Dictionary of {symbol: DataFrame}.
        cutoff_date (str): The cutoff date (YYYY-MM-DD).
        date_column (str): Name of the date column.

    Returns:
        list: List of stock symbols that end before the cutoff date.
    """
    cutoff = pd.to_datetime(cutoff_date)
    ended_early = []

    for symbol, df in all_stocks.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty and dates.max() < cutoff:
            ended_early.append(symbol)

    # Print results
    if ended_early:
        print(f"⚠️ The following stocks end before {cutoff.date()}:")
        for s in ended_early:
            print(f"  - {s}")
    else:
        print(f"✅ All stocks extend at least until {cutoff.date()}.")

    return ended_early




def get_start_dates(all_stocks, date_column="Date"):
    """
    Returns a DataFrame with the first date for each stock.
    """
    rows = []
    for symbol, df in all_stocks.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            rows.append({"symbol": symbol, "start_date": dates.min()})
    return pd.DataFrame(rows)

def plot_start_date_distribution(all_stocks, date_column="Date"):
    """
    Plots the distribution (histogram) of stock start dates.
    """
    rows = []
    for symbol, df in all_stocks.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            rows.append(dates.min())

    if not rows:
        print("⚠️ No valid start dates found.")
        return

    plt.figure(figsize=(8, 4))
    plt.hist(rows, bins=20, edgecolor="black")
    plt.title("Distribution of Stock Start Dates")
    plt.xlabel("Start Date")
    plt.ylabel("Number of Stocks")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    
import pandas as pd
import matplotlib.pyplot as plt

def start_date_counts(all_stocks, date_column="Date"):
    """
    Return a Series indexed by exact start date (YYYY-MM-DD) with counts of stocks.
    Sorted by date ascending.
    """
    starts = []
    for symbol, df in all_stocks.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            starts.append(dates.min().normalize())  # exact date at midnight

    if not starts:
        return pd.Series(dtype="int64")

    s = pd.Series(starts)
    s = s.dt.date.value_counts().sort_index()  # index: datetime.date
    return s


def plot_start_date_distribution(all_stocks, date_column="Date", bins=60, print_per_date=True):
    """
    Plot histogram of start dates with more bins,
    and optionally print counts per exact start date.
    """
    # collect start dates
    rows = []
    for symbol, df in all_stocks.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            rows.append(dates.min().normalize())

    if not rows:
        print("⚠️ No valid start dates found.")
        return

    # histogram with more bins
    plt.figure(figsize=(8, 4))
    plt.hist(rows, bins=bins, edgecolor="black")
    plt.title("Distribution of Symbols Start Dates")
    plt.xlabel("Start Date")
    plt.ylabel("Number of Stocks")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # print exact-date counts
    if print_per_date:
        counts = pd.Series(rows).dt.date.value_counts().sort_index()
        print("\n📅 Stocks starting on each exact date:\n")
        for d, c in counts.items():
            print(f"{d}: {c} stock(s)")
        return counts  # handy if you want to reuse it
    
    
# src/utils/metadata.py  (add this below load_nasdaq_metadata)
import pandas as pd

def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
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



def missing_rates_from_dict(all_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute missing value rates per column for each symbol,
    and overall missing rates across all symbols.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.

    Returns
    -------
    per_symbol : pd.DataFrame
        Missing rate (%) per column for each symbol.
    overall : pd.DataFrame
        Total missing rate (%) per column across all symbols combined.
    """
    # compute per-symbol missing rates
    per_symbol = {}
    total_missing = None
    total_rows = 0

    for symbol, df in all_data.items():
        # per-symbol rates
        per_symbol[symbol] = df.isna().mean() * 100

        # accumulate totals for overall rate
        miss_count = df.isna().sum()
        n = len(df)
        total_rows += n
        if total_missing is None:
            total_missing = miss_count.copy()
        else:
            total_missing += miss_count

    per_symbol = pd.DataFrame(per_symbol).T  # transpose to have symbols as rows
    per_symbol.index.name = "Symbol"

    # overall missing rate (weighted by row count)
    overall = (total_missing / total_rows * 100).to_frame("Total Missing Rate (%)")

    return per_symbol, overall


import pandas as pd
import numpy as np

def zero_rates_from_dict(all_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 0.0 value rates per column for each symbol,
    and overall 0.0 rates across all symbols.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.

    Returns
    -------
    per_symbol : pd.DataFrame
        Zero rate (%) per column for each symbol.
    overall : pd.DataFrame
        Total zero rate (%) per column across all symbols combined.
    """
    per_symbol = {}
    total_zeros = None
    total_rows = 0

    for symbol, df in all_data.items():
        # mask numeric columns only (to avoid issues with strings/dates)
        numeric_df = df.select_dtypes(include=[np.number])

        # compute per-symbol zero percentage
        zero_pct = (numeric_df == 0).mean() * 100
        per_symbol[symbol] = zero_pct

        # accumulate total zero counts
        zero_count = (numeric_df == 0).sum()
        n = len(numeric_df)
        total_rows += n
        if total_zeros is None:
            total_zeros = zero_count.copy()
        else:
            total_zeros += zero_count

    per_symbol = pd.DataFrame(per_symbol).T
    per_symbol.index.name = "Symbol"

    # overall 0.0 rate (weighted by total number of rows)
    overall = (total_zeros / total_rows * 100).to_frame("Total Zero Rate (%)")

    return per_symbol, overall










