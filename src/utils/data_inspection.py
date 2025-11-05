import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def show_date_ranges(data_dict, label, date_column="Date"):
    """
    Prints the first and last date available for each stock/ETF.
    """
    print(f"\n Date ranges for all {label}:\n")
    for symbol, df in data_dict.items():
        first, last = get_date_range(df, date_column)
        if first and last:
            print(f"{symbol}: from {first.date()} → {last.date()}")
        else:
            print(f"{symbol}: ⚠️ No valid dates found.")

            
            


def ends_before(data_dict, label, cutoff_date="2020-04-01", date_column="Date"):
    """
    Checks which stocks/ETFs end before a given cutoff date.

    Parameters:
        data_dict (dict): Dictionary of {symbol: DataFrame}.
        label (str): Label describing the dataset (e.g., "stocks" or "ETFs").
        cutoff_date (str): The cutoff date (YYYY-MM-DD).
        date_column (str): Name of the date column.

    Returns:
        list: List of symbols that end before the cutoff date.
    """
    cutoff = pd.to_datetime(cutoff_date)
    ended_early = []

    for symbol, df in data_dict.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty and dates.max() < cutoff:
            ended_early.append(symbol)

    # Print results
    if ended_early:
        print(f" The following {label} end before {cutoff.date()}:")
        for s in ended_early:
            print(f"  - {s}")
    else:
        print(f" All {label} extend at least until {cutoff.date()}.")

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




    
    


def start_date_counts(data_dict, label, date_column="Date"):
    """
    Return a Series indexed by exact start date (YYYY-MM-DD)
    with counts of stocks/ETFs, sorted by date ascending.

    Parameters:
        data_dict (dict): Dictionary of {symbol: DataFrame}.
        label (str): Label describing the dataset (e.g., "stocks" or "ETFs").
        date_column (str): Name of the date column.

    Returns:
        pd.Series: Counts of {label} per start date.
    """
    starts = []
    for symbol, df in data_dict.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            starts.append(dates.min().normalize())  # exact date at midnight

    if not starts:
        print(f" No valid start dates found for {label}.")
        return pd.Series(dtype="int64")

    s = pd.Series(starts)
    s = s.dt.date.value_counts().sort_index()  # index: datetime.date
    print(f" Start date counts for {label}:")
    return s






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


def get_symbols_with_zeros(all_data: dict[str, pd.DataFrame], column: str):
    """
    Return a list of symbols that have at least one 0.0 or NaN value
    in the specified column.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.
    column : str
        Column name to check.

    Returns
    -------
    list[str]
        List of symbols with 0.0 or NaN values in the given column.
    """
    symbols_with_issues = []

    for symbol, df in all_data.items():
        if column not in df.columns:
            continue

        series = df[column]

        if not pd.api.types.is_numeric_dtype(series):
            continue

        if (series == 0).any():
            symbols_with_issues.append(symbol)

    return symbols_with_issues


def get_symbols_with_zeros_or_nans(all_data: dict[str, pd.DataFrame], column: str):
    """
    Return a list of symbols that have at least one 0.0 or NaN value
    in the specified column.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol.
    column : str
        Column name to check for 0.0 or NaN values.

    Returns
    -------
    list[str]
        List of symbols that contain 0.0 or NaN values in the given column.
    """
    symbols_with_issues = []

    for symbol, df in all_data.items():
        # Skip symbols missing the target column
        if column not in df.columns:
            continue

        # Select the column
        series = df[column]

        # Skip non-numeric columns (e.g., strings or dates)
        if not pd.api.types.is_numeric_dtype(series):
            continue

        # Check for any 0.0 or NaN values
        if (series == 0).any() or series.isna().any():
            symbols_with_issues.append(symbol)























