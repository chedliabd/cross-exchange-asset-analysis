import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_start_date_distribution(
    data_dict,
    label,
    date_column="Date",
    bins=60,
    return_counts=False,
):
    """
    For a dict {symbol: DataFrame}, get the start date of each symbol
    and plot the distribution of these start dates.

    Parameters
    ----------
    data_dict : dict
        Dictionary of {symbol: DataFrame}.
    label : str
        Label describing the dataset (e.g., "stocks" or "ETFs").
    date_column : str, default "Date"
        Name of the date column in each DataFrame.
    bins : int, default 60
        Number of bins for the histogram.
    return_counts : bool, default False
        If True, returns a Series with counts per exact start date.

    Returns
    -------
    pandas.Series or None
        Series of counts per exact start date (if return_counts=True),
        otherwise None.
    """
    # 1) collect start dates
    start_dates = []
    for symbol, df in data_dict.items():
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty:
            start_dates.append(dates.min().normalize())  # exact date at midnight

    if not start_dates:
        print(f"No valid start dates found for {label}.")
        return None

    # 2) plot histogram of start dates
    plt.figure(figsize=(8, 4))
    plt.hist(start_dates, bins=bins, edgecolor="black")
    plt.title(f"Distribution of {label} Start Dates")
    plt.xlabel("Start Date")
    plt.ylabel(f"Number of {label}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3) optional exact-date counts
    if return_counts:
        counts = pd.Series(start_dates).dt.date.value_counts().sort_index()
        return counts

    

def plot_missing_counts_over_time(all_data: dict[str, pd.DataFrame], label: str = "Stocks"):
    """
    Plot the total number of missing values over time across all symbols.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol (must have a 'Date' column).
    label : str, optional
        Label for the plot title (e.g., 'Stocks' or 'ETFs').
    """
    missing_by_date = []

    for symbol, df in all_data.items():
        if "Date" not in df.columns:
            continue

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Count missing values per row (sum over columns)
        df["missing_count"] = df.isna().sum(axis=1)

        # Aggregate per date (in case of duplicates)
        daily_missing = df.groupby("Date")["missing_count"].sum().reset_index()
        missing_by_date.append(daily_missing)

    # Combine all symbols' missing counts per date
    merged = pd.concat(missing_by_date, ignore_index=True)
    total_missing_per_date = merged.groupby("Date")["missing_count"].sum().reset_index()

    # Sort by date
    total_missing_per_date = total_missing_per_date.sort_values("Date")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(total_missing_per_date["Date"], total_missing_per_date["missing_count"], color="orange")
    plt.title(f"🕒 Total Missing Values Over Time ({label})")
    plt.xlabel("Date")
    plt.ylabel("Number of Missing Values")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return total_missing_per_date


def plot_zero_counts_over_time(
    all_data: dict[str, pd.DataFrame],
    label: str = "Stocks",
    column: str | None = None
):
    """
    Plot the total number of 0.0 values over time across all symbols.

    Parameters
    ----------
    all_data : dict[str, pd.DataFrame]
        Dictionary where key = symbol, value = DataFrame of that symbol (must have a 'Date' column).
    label : str, optional
        Label for the plot title (e.g., 'Stocks' or 'ETFs').
    column : str, optional
        Specific column to check for zeros. If None, all numeric columns are used.
    """
    zero_by_date = []

    for symbol, df in all_data.items():
        if "Date" not in df.columns:
            continue

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Select the column to analyze
        if column is not None:
            if column not in df.columns:
                continue
            numeric_df = df[[column]].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            continue

        # Count zero values per row
        df["zero_count"] = (numeric_df == 0).sum(axis=1)

        # Aggregate by date (sum across rows for that symbol)
        daily_zero = df.groupby("Date")["zero_count"].sum().reset_index()
        zero_by_date.append(daily_zero)

    if not zero_by_date:
        print(f"⚠️ No valid numeric data found for zero count analysis ({label}).")
        return None

    # Combine all symbols’ zero counts per date
    merged = pd.concat(zero_by_date, ignore_index=True)
    total_zero_per_date = merged.groupby("Date")["zero_count"].sum().reset_index()

    # Sort by date
    total_zero_per_date = total_zero_per_date.sort_values("Date")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(total_zero_per_date["Date"], total_zero_per_date["zero_count"], color="royalblue")
    col_text = f" ({column})" if column else ""
    plt.title(f"🕒 Total Zero Values Over Time ({label}){col_text}")
    plt.xlabel("Date")
    plt.ylabel("Number of Zero Values")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return total_zero_per_date
