from pathlib import Path
from typing import Dict, List, Tuple, Literal, Sequence

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Sequence
from sklearn.cluster import AgglomerativeClustering

try:
    import umap
except ImportError:
    umap = None

try:
    import hdbscan
except ImportError:
    hdbscan = None

# --------------------------------------------------------------------
# 1. Data loading & basic prep
# --------------------------------------------------------------------

def load_data(
    data_dir: Path | str = Path("data_processed"),
    stocks_file: str | Path = "stocks_sect_clean.parquet",
) -> Dict[str, pd.DataFrame]:
    """
    Load all necessary parquet files for Chapter 2.

    Parameters
    ----------
    data_dir : Path or str
        Base directory containing the processed parquet files.
    stocks_file : Path or str
        Name of the sector-level stocks parquet file to load
        (defaults to 'stocks_sect_clean.parquet').

    Returns a dict with:
        {
            "etfs": df_etfs_clean,
            "meta": df_meta,
            "stocks_sect": df_stock_sect_clean,
            "stocks": df_stocks_sect_clean (alias for backward compatibility)
        }
    """
    root = Path(data_dir)
    stock_path = root / stocks_file
    files = {
        "etfs": root / "etfs_clean.parquet",
        "meta": root / "metadata_clean.parquet",
        "stocks_sect": stock_path,
    }

    data = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")
        data[key] = pd.read_parquet(path)
    # Keep the previous "stocks" key pointing to the cleaned sector-level data.
    data["stocks"] = data["stocks_sect"]
    return data


def _infer_ticker_column(df: pd.DataFrame) -> str:
    if "Symbol" in df.columns:
        return "Symbol"
    if "ticker" in df.columns:
        return "ticker"
    raise KeyError("No ticker column found (expected 'Symbol' or 'ticker').")


def prepare_sector_panel(
    df_stock_sect: pd.DataFrame,
    sector: str,
    use_return_col: str = "Return",
    min_start_date: str | None = None,
    max_end_date: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a clean time-series panel for one sector.

    Parameters
    ----------
    df_stock_sect : DataFrame
        Full cleaned sector-level stocks dataframe.
    sector : str
        Sector name to filter on (e.g. 'Healthcare').
    use_return_col : str
        Column to use as stock return (e.g. 'Return').
    min_start_date, max_end_date : optional
        If provided, restrict the date range.

    Returns
    -------
    returns_wide : DataFrame
        Index = Date, columns = tickers, values = stock returns.
    sector_proxy : Series
        Sector proxy daily return, aligned on the same Date index,
        computed as sum_i(weight_i,t * return_i,t).
    volume_wide : DataFrame
        Index = Date, columns = tickers, values = Volume.
    """
    ticker_col = _infer_ticker_column(df_stock_sect)
    df = df_stock_sect.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Sector"] == sector]

    if min_start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(min_start_date)]
    if max_end_date is not None:
        df = df[df["Date"] <= pd.to_datetime(max_end_date)]

    df = df.sort_values("Date")

    returns_wide = df.pivot(index="Date", columns=ticker_col, values=use_return_col).sort_index()
    volume_wide = df.pivot(index="Date", columns=ticker_col, values="Volume").sort_index()

    weighted_returns = df.assign(
        weighted_ret=df["Weight"].fillna(0) * df[use_return_col]
    )
    sector_proxy = (
        weighted_returns.groupby("Date")["weighted_ret"].sum().reindex(returns_wide.index)
    )

    return returns_wide, sector_proxy, volume_wide


# --------------------------------------------------------------------
# 2. Liquidity filter & representativeness scores
# --------------------------------------------------------------------

def filter_liquid_stocks(
    returns_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    min_nonzero_fraction: float = 0.8,
    min_avg_volume_quantile: float = 0.2,
) -> List[str]:
    """
    Filter out illiquid stocks within a sector.

    Logic:
    - Keep only stocks with at least `min_nonzero_fraction` of days
      having Volume > 0.
    - Among those, keep stocks whose average volume is above the
      `min_avg_volume_quantile` quantile across the sector.

    Returns
    -------
    kept_tickers : list of str
        List of tickers passing the liquidity filter.
    """
    # Use only the dates where volume is actually observed to avoid penalizing
    # stocks that simply have shorter histories (e.g., recent IPOs).
    observed_counts = volume_wide.notna().sum(axis=0)
    positive_counts = (volume_wide > 0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_positive_frac = positive_counts / observed_counts
    volume_positive_frac = volume_positive_frac.fillna(0)

    liquid_candidates = volume_positive_frac[volume_positive_frac >= min_nonzero_fraction].index
    if len(liquid_candidates) == 0:
        return []

    avg_volume = volume_wide[liquid_candidates].mean(axis=0, skipna=True)
    if avg_volume.dropna().empty:
        return []
    threshold = avg_volume.quantile(min_avg_volume_quantile)
    kept = avg_volume[avg_volume >= threshold].index.tolist()
    return kept


def compute_representativeness_scores(
    returns_wide: pd.DataFrame,
    sector_proxy: pd.Series,
    window: int = 60,
    min_periods: int = 40,
) -> pd.DataFrame:
    """
    Compute representativeness scores for each stock in a sector.

    For each ticker:
        - compute rolling correlation between stock return and sector proxy
          using a trailing window (e.g. 60 days),
        - average the rolling correlation over time to get a score.

    Returns
    -------
    scores_df : DataFrame
        Index = ticker, columns include:
            - 'rep_score' (average rolling correlation)
            - 'avg_corr' (alias for rep_score)
            - 'n_obs' (number of valid rolling windows)
    """
    if returns_wide.empty or returns_wide.shape[1] == 0:
        return pd.DataFrame(columns=["rep_score", "avg_corr", "n_obs"])
    if returns_wide.dropna(how="all").empty:
        return pd.DataFrame(index=returns_wide.columns, columns=["rep_score", "avg_corr", "n_obs"])

    # Align on the time index (rows) only; sector_proxy is a Series.
    aligned_returns, aligned_proxy = returns_wide.align(sector_proxy, axis=0, join="inner")

    rolling_corr = aligned_returns.rolling(window=window, min_periods=min_periods).corr(aligned_proxy)
    rep_score = rolling_corr.mean(skipna=True)
    n_obs = rolling_corr.count()

    scores_df = pd.DataFrame(
        {
            "rep_score": rep_score,
            "avg_corr": rep_score,
            "n_obs": n_obs,
        }
    )
    return scores_df


def compute_lifetime_fraction(returns_wide: pd.DataFrame) -> pd.Series:
    """
    Compute the fraction of the analysis period where each ticker has valid returns.

    Lifetime fraction = (# non-NaN returns for ticker) / (total # dates in panel).

    Using the panel length avoids over-weighting recent tickers; short histories
    are penalized when selecting representative stocks.
    """
    if returns_wide.empty:
        return pd.Series(dtype=float)

    total = len(returns_wide.index)
    if total == 0:
        return pd.Series(dtype=float)

    counts = returns_wide.notna().sum(axis=0)
    return counts / total


def compute_risk_metrics(
    returns_wide: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute risk metrics for each stock in the sector:

    - full-period volatility (std of daily returns),
    - max drawdown (from normalized index),
    - expected shortfall (ES) at level alpha.

    Returns
    -------
    risk_df : DataFrame
        Index = ticker, columns:
            - 'vol'
            - 'max_drawdown'
            - 'es_alpha'
    """
    risk = {}
    for ticker, series in returns_wide.items():
        s = series.dropna()
        if s.empty:
            risk[ticker] = {"vol": np.nan, "max_drawdown": np.nan, "es_alpha": np.nan}
            continue

        vol = s.std()

        # Detect whether returns are log-like (if any return <= -1, simple compounding fails)
        use_log = (1 + s).min() <= 0
        if use_log:
            equity = np.exp(s.cumsum())
        else:
            equity = (1 + s).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = -drawdown.min() if len(drawdown) else np.nan

        var_alpha = s.quantile(alpha)
        es = s[s <= var_alpha].mean() if not s[s <= var_alpha].empty else np.nan

        risk[ticker] = {"vol": vol, "max_drawdown": max_dd, "es_alpha": es}

    risk_df = pd.DataFrame.from_dict(risk, orient="index")
    return risk_df


# --------------------------------------------------------------------
# 3. Intra-sector correlation, UMAP & clustering
# --------------------------------------------------------------------

def compute_intra_sector_correlation(
    returns_wide: pd.DataFrame,
    min_overlap: int = 2,
) -> pd.DataFrame:
    """
    Compute the stock–stock correlation matrix using all available history
    per pair (pairwise overlap), not only the common intersection across
    all tickers.

    For each pair (i, j):
        - find dates where both returns are non-NaN,
        - compute Pearson correlation on that full overlap (subject to a tiny
          min_overlap guard to avoid degenerate windows),
        - otherwise return NaN for that pair.

    Returns
    -------
    corr : DataFrame
        Ticker x ticker correlation matrix.
    """
    if returns_wide.empty or returns_wide.shape[1] == 0:
        return pd.DataFrame()

    tickers = list(returns_wide.columns)
    n = len(tickers)
    corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers, dtype=float)

    for i in range(n):
        s_i = returns_wide.iloc[:, i]
        for j in range(i + 1, n):
            s_j = returns_wide.iloc[:, j]
            mask = s_i.notna() & s_j.notna()
            overlap = mask.sum()
            if overlap >= max(2, min_overlap):
                rho = s_i[mask].corr(s_j[mask])
            else:
                rho = np.nan
            corr.iat[i, j] = rho
            corr.iat[j, i] = rho

    return corr


def _corr_to_distance(
    corr: pd.DataFrame,
    scale: float = 2.0,
    distance_metric: Literal["linear", "angular"] = "linear",
    nan_distance_fill: Literal["max", "p90", "median"] = "max",
) -> pd.DataFrame:
    """
    Convert correlation matrix to a distance matrix with explicit NaN handling.

    Parameters
    ----------
    corr : DataFrame
        Correlation matrix (tickers x tickers).
    scale : float
        Scaling factor for linear distance (default 2.0).
    distance_metric : {"linear", "angular"}
        - linear: d = scale * (1 - rho)
        - angular: d = scale * sqrt(2 * (1 - rho))
    nan_distance_fill : {"max", "p90", "median"}
        How to fill NaN distances (from missing correlations) to discourage merging:
        - "max": max(existing_distance, 2*scale)
        - "p90": 90th percentile of finite distances (fallback 2*scale)
        - "median": median of finite distances (fallback 2*scale)
    """
    if corr.empty:
        return pd.DataFrame()

    if distance_metric == "linear":
        distance = scale * (1 - corr)
    elif distance_metric == "angular":
        distance = scale * np.sqrt(2 * (1 - corr.clip(-1, 1)))
    else:
        raise ValueError(f"Unsupported distance_metric: {distance_metric}")

    dist_values = distance.to_numpy().astype(float)
    np.fill_diagonal(dist_values, 0.0)

    nan_mask = np.isnan(dist_values)
    finite = dist_values[~nan_mask]
    fallback = 2 * scale

    if nan_distance_fill == "max":
        fill_val = fallback if finite.size == 0 else max(float(np.nanmax(finite)), fallback)
    elif nan_distance_fill == "p90":
        fill_val = fallback if finite.size == 0 else float(np.nanpercentile(finite, 90))
    elif nan_distance_fill == "median":
        fill_val = fallback if finite.size == 0 else float(np.nanmedian(finite))
    else:
        raise ValueError(f"Unsupported nan_distance_fill: {nan_distance_fill}")

    if nan_mask.any():
        dist_values[nan_mask] = fill_val

    np.fill_diagonal(dist_values, 0.0)
    distance = pd.DataFrame(dist_values, index=corr.index, columns=corr.columns)
    return distance.clip(lower=0)


def run_umap_embedding(
    corr: pd.DataFrame,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    random_state: int = 42,
    distance_scale: float = 2.0,
    distance_metric: Literal["linear", "angular"] = "linear",
    nan_distance_fill: Literal["max", "p90", "median"] = "max",
    missing_tickers: Sequence[str] | None = None,
    missingdata_spread: float = 5.0,
    missingdata_jitter: float = 0.2,
) -> pd.DataFrame:
    """
    Run UMAP on the distance matrix derived from correlations.

    Distance = scale * (1 - corr_ij) with default scale=2 (or angular metric).

    MissingData handling:
    - NaN correlations are converted to large distances via nan_distance_fill.
    - Optional missing_tickers are appended after embedding and pushed far away
      (spread + jitter) so they appear as a distinct cluster in plots.

    Returns
    -------
    embedding_df : DataFrame
        Index = ticker, columns = ['UMAP 1', 'UMAP 2'] with 2D UMAP coordinates.
    """
    if umap is None:
        raise ImportError("UMAP is not installed. Install `umap-learn` to use this function.")

    if corr.empty:
        base_df = pd.DataFrame(columns=["UMAP 1", "UMAP 2"])
        if missing_tickers:
            rng = np.random.default_rng(random_state)
            extra = []
            for t in missing_tickers:
                jitter_x = rng.normal(0, missingdata_jitter)
                jitter_y = rng.normal(0, missingdata_jitter)
                extra.append({"ticker": t, "UMAP 1": missingdata_spread + jitter_x, "UMAP 2": missingdata_spread + jitter_y})
            return pd.DataFrame(extra).set_index("ticker")
        return base_df

    if corr.shape[0] == 1:
        embedding_df = pd.DataFrame([[0.0, 0.0]], index=corr.index, columns=["UMAP 1", "UMAP 2"])
    else:
        distance = _corr_to_distance(
            corr,
            scale=distance_scale,
            distance_metric=distance_metric,
            nan_distance_fill=nan_distance_fill,
        )

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="precomputed",
            random_state=random_state,
        )
        embedding = reducer.fit_transform(distance)
        embedding_df = pd.DataFrame(embedding, index=corr.index, columns=["UMAP 1", "UMAP 2"])

    # Push MissingData tickers far away to make them visually separate.
    if missing_tickers:
        rng = np.random.default_rng(random_state)
        base_x = embedding_df["x"].max() if not embedding_df.empty else 0.0
        base_y = embedding_df["y"].max() if not embedding_df.empty else 0.0
        offset_x = base_x + missingdata_spread
        offset_y = base_y + missingdata_spread
        extra = []
        for t in missing_tickers:
            jitter_x = rng.normal(0, missingdata_jitter)
            jitter_y = rng.normal(0, missingdata_jitter)
            extra.append({"ticker": t, "UMAP 1": offset_x + jitter_x, "UMAP 2": offset_y + jitter_y})
        if extra:
            extra_df = pd.DataFrame(extra).set_index("ticker")
            embedding_df = pd.concat([embedding_df, extra_df], axis=0)

    return embedding_df


def cluster_stocks(
    corr: pd.DataFrame,
    method: Literal["agglomerative", "hdbscan", "hdbscan_umap"] = "agglomerative",
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    distance_scale: float = 2.0,
    linkage: Literal["average", "complete", "single"] = "complete",
    threshold_percentile: float = 30.0,
    distance_metric: Literal["linear", "angular"] = "linear",
    nan_distance_fill: Literal["max", "p90", "median"] = "max",
    max_nan_frac: float = 0.25,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
    missing_tickers: Sequence[str] | None = None,
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.1,
) -> pd.Series:
    """
    Cluster stocks within a sector based on their correlation structure.

    Distance options:
    - linear: scale * (1 - correlation)
    - angular: scale * sqrt(2 * (1 - correlation))

    Supported methods:
    - 'agglomerative' (default)
    - 'hdbscan' (density-based, requires the `hdbscan` package, metric=precomputed)
    - 'hdbscan_umap' (UMAP embedding then HDBSCAN on Euclidean coords)

    Notes:
    - NaN correlations are turned into large distances via nan_distance_fill.
    - Tickers with too many missing pairwise correlations (nan_frac > max_nan_frac)
      are assigned to cluster -99 ("MissingData") and excluded from fitting.
    - If neither n_clusters nor distance_threshold is provided (agglomerative),
      a distance threshold is chosen from the specified percentile of the
      empirical distance distribution.

    Examples
    --------
    # Agglomerative with data-driven threshold (p30)
    cluster_stocks(corr, method="agglomerative", threshold_percentile=30)
    # Agglomerative forcing 4 clusters
    cluster_stocks(corr, method="agglomerative", n_clusters=4)
    # HDBSCAN on UMAP embedding
    cluster_stocks(corr, method="hdbscan_umap", hdbscan_min_cluster_size=6)

    Returns
    -------
    cluster_labels : Series
        Index = ticker, values = integer cluster labels.
    """
    if corr.empty:
        return pd.Series(dtype=int, name="cluster")

    n = corr.shape[0]
    if missing_tickers is None:
        denom = max(1, n - 1)
        nan_frac = corr.isna().sum(axis=1) / denom
        missing_mask = nan_frac > max_nan_frac
        missing_set = set(corr.index[missing_mask])
    else:
        missing_set = set(missing_tickers)

    tickers = list(corr.index)
    good_tickers = [t for t in tickers if t not in missing_set]

    # If not enough good tickers, mark everything as MissingData.
    if len(good_tickers) < 2:
        labels = pd.Series(-99, index=corr.index, name="cluster", dtype=int)
        return labels

    corr_good = corr.loc[good_tickers, good_tickers]
    distance = _corr_to_distance(
        corr_good,
        scale=distance_scale,
        distance_metric=distance_metric,
        nan_distance_fill=nan_distance_fill,
    )

    if method == "hdbscan":
        if hdbscan is None:
            raise ImportError("HDBSCAN is not installed. Install `hdbscan` to use this clustering method.")
        model = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )
        labels_good = model.fit_predict(distance)
    elif method == "hdbscan_umap":
        if hdbscan is None:
            raise ImportError("HDBSCAN is not installed. Install `hdbscan` to use this clustering method.")
        embedding_df = run_umap_embedding(
            corr_good,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            distance_scale=distance_scale,
            distance_metric=distance_metric,
            nan_distance_fill=nan_distance_fill,
            missing_tickers=[],
        )
        labels_good = hdbscan.HDBSCAN(
            metric="euclidean",
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        ).fit_predict(embedding_df[["UMAP 1", "UMAP 2"]].to_numpy())
    elif method == "agglomerative":
        # Choose distance mode
        dist_threshold = distance_threshold
        if n_clusters is None and dist_threshold is None:
            triu_mask = np.triu(np.ones_like(distance, dtype=bool), k=1)
            dist_vals = distance.where(triu_mask).stack().dropna()
            if dist_vals.empty:
                dist_threshold = distance_scale
            else:
                dist_threshold = float(np.percentile(dist_vals, threshold_percentile))

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            distance_threshold=None if n_clusters is not None else dist_threshold,
            metric="precomputed",
            linkage=linkage,
        )
        labels_good = model.fit_predict(distance)
    else:
        raise NotImplementedError(f"Clustering method '{method}' is not supported.")

    labels_series = pd.Series(labels_good, index=good_tickers, name="cluster")
    if missing_set:
        missing_labels = pd.Series(-99, index=sorted(missing_set), name="cluster", dtype=int)
        labels_series = pd.concat([labels_series, missing_labels], axis=0)
    # Reindex to original order
    labels_series = labels_series.reindex(tickers)
    return labels_series


# --------------------------------------------------------------------
# 4. Basket construction & tracking error
# --------------------------------------------------------------------

def select_top_representative_stocks(
    scores_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    lifetime_fraction: pd.Series,
    top_n: int = 5,
    alpha: float = 2.0,
    min_lifetime_fraction: float | None = None,
) -> List[str]:
    """
    Select the Top-N representative stocks in a sector.

    Base rule (lifetime-weighted):
    - effective_score = rep_score * (lifetime_fraction ** alpha)
    - Rank by effective_score (desc), then lower vol (asc) as tie-break.
    - Optionally drop tickers below a minimum lifetime_fraction.

    Returns
    -------
    top_tickers : list of str
        Selected tickers.
    """
    merged = scores_df.join(risk_df[["vol"]], how="inner")
    merged = merged.join(lifetime_fraction.rename("lifetime_fraction"), how="left")

    if min_lifetime_fraction is not None:
        merged = merged[merged["lifetime_fraction"] >= min_lifetime_fraction]

    merged["effective_score"] = merged["rep_score"] * (merged["lifetime_fraction"] ** alpha)

    ranked = merged.sort_values(by=["effective_score", "vol"], ascending=[False, True])
    return ranked.head(top_n).index.tolist()


def build_basket_returns(
    returns_wide: pd.DataFrame,
    basket_tickers: List[str],
    weights: str = "equal",
) -> pd.Series:
    """
    Build the daily return series for a stock basket.

    If weights='equal', each stock has the same weight.

    Returns
    -------
    basket_return : Series
        Daily returns of the basket.
    """
    if len(basket_tickers) == 0:
        return pd.Series(dtype=float)

    basket_returns = returns_wide[basket_tickers]
    if weights != "equal":
        raise NotImplementedError("Only equal weighting is supported currently.")

    base_weights = np.ones(len(basket_tickers)) / len(basket_tickers)
    weight_matrix = pd.DataFrame([base_weights] * len(basket_returns), index=basket_returns.index, columns=basket_tickers)

    valid_weight_sum = weight_matrix.where(~basket_returns.isna()).sum(axis=1)
    weighted_sum = (basket_returns * weight_matrix).sum(axis=1, min_count=1)
    basket_series = weighted_sum / valid_weight_sum
    return basket_series


def compute_tracking_error_series(
    proxy_returns: pd.Series,
    sector_proxy: pd.Series,
    window: int = 60,
    min_periods: int = 40,
) -> pd.Series:
    """
    Compute rolling tracking error between a proxy (stock or basket)
    and the sector proxy.

    TE_t = rolling_std( r_proxy(t) - r_sector(t) )

    Returns
    -------
    te_series : Series
        Rolling tracking error time series.
    """
    if proxy_returns.empty:
        return pd.Series(dtype=float)

    aligned_proxy, aligned_sector = proxy_returns.align(sector_proxy, join="inner")
    active_return = aligned_proxy - aligned_sector
    return active_return.rolling(window=window, min_periods=min_periods).std()


# --------------------------------------------------------------------
# 5. Plotting utilities (Plotly, interactive)
# --------------------------------------------------------------------

def format_cluster_labels(raw_labels: pd.Series, include_size: bool = True) -> pd.Series:
    """
    Make user-friendly cluster labels for legends.

    - Standard clusters: sorted by size (largest first) then renamed
      "Cluster 1", "Cluster 2", ...
    - Special labels are preserved:
        - -99 -> "Missing data"
        - -1  -> "Outlier"
        - NaN -> "Unassigned"
    - Optionally appends cluster size for context.

    Clusters group stocks with similar historical return behavior; numbers are
    identifiers only and do not imply performance or quality.
    """
    if raw_labels.empty:
        return pd.Series(dtype=str)

    specials = {-99: "Missing data", -1: "Outlier"}
    label_sizes = raw_labels.value_counts(dropna=True)

    standard_labels = [
        lbl for lbl in label_sizes.index
        if not (pd.isna(lbl) or lbl in specials)
    ]

    standard_labels_sorted = sorted(
        standard_labels,
        key=lambda x: (-label_sizes[x], x)
    )

    remap = {}
    for i, lbl in enumerate(standard_labels_sorted, start=1):
        size = label_sizes[lbl]
        remap[lbl] = f"Cluster {i}" + (f" ({size} stocks)" if include_size else "")

    def _fmt(val):
        if pd.isna(val):
            return "Unassigned"
        if val in specials:
            return specials[val]
        return remap.get(val, str(val))

    return raw_labels.map(_fmt)


def plot_sector_umap(
    embedding_df: pd.DataFrame,
    cluster_labels: pd.Series,
    scores_df: pd.DataFrame,
    avg_volume: pd.Series,
    basket_tickers: List[str],
    sector_name: str,
):
    """
    Create an interactive UMAP scatterplot for one sector.

    - Points = stocks.
    - Position = UMAP 2D coordinates.
    - Color = cluster label.
    - Size = average volume.
    - Marker outline or symbol for selected basket_tickers.

    Hover tooltip should show:
        - ticker
        - representativeness score
        - average volume
        - cluster label

    Returns a Plotly Figure.
    """
    tickers = embedding_df.index
    df_plot = embedding_df.copy()
    df_plot["ticker"] = tickers
    clusters_raw = cluster_labels.reindex(tickers)
    cluster_display = format_cluster_labels(clusters_raw)
    df_plot["cluster"] = cluster_display
    df_plot["rep_score"] = scores_df.reindex(tickers)["rep_score"]
    avg_vol = avg_volume.reindex(tickers)
    fallback_vol = float(avg_vol.median(skipna=True)) if not avg_vol.dropna().empty else 1.0
    df_plot["avg_volume"] = avg_vol.fillna(fallback_vol)
    df_plot["in_basket"] = df_plot["ticker"].isin(basket_tickers)

    fig = px.scatter(
        df_plot,
        x="UMAP 1",
        y="UMAP 2",
        color="cluster",
        size="avg_volume",
        hover_data={
            "ticker": True,
            "rep_score": ":.2f",
            "avg_volume": ":.2s",
            "cluster": True,
            "in_basket": False,
        },
        title=f"{sector_name}: Intra-sector UMAP",
        category_orders={"cluster": sorted(df_plot["cluster"].unique())},
    )

    basket_df = df_plot[df_plot["in_basket"]]
    if not basket_df.empty:
        max_vol = basket_df["avg_volume"].max()
        size_scale = 30 if pd.notna(max_vol) and max_vol > 0 else 10
        fig.add_trace(
            go.Scatter(
                x=basket_df["UMAP 1"],
                y=basket_df["UMAP 2"],
                mode="markers+text",
                name="Top basket",
                text=basket_df["ticker"],
                textposition="top center",
                marker_symbol="star",
                marker=dict(
                    size=np.clip(basket_df["avg_volume"] / max_vol * size_scale, 10, size_scale) if max_vol and not np.isnan(max_vol) else 10,
                    color="black",
                    line=dict(width=2, color="white"),
                ),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        legend_title_text="Cluster",
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------
# Export helpers (HTML plots for website embedding)
# --------------------------------------------------------------------

def apply_blog_layout(
    fig: go.Figure,
    *,
    theme: str = "dark",
    width: int = 1100,
    height: int = 520,
    title: str | None = None,
) -> go.Figure:
    if fig is None:
        return fig
    is_dark = theme == "dark"
    paper_bg = "#0f172a" if is_dark else "#f8fafc"
    plot_bg = "#0b1220" if is_dark else "#ffffff"
    font_color = "#e5e7eb" if is_dark else "#0f172a"
    grid_color = "#1f2937" if is_dark else "#e5e7eb"
    axis_color = "#cbd5e1" if is_dark else "#475569"
    title_color = font_color

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(
            family='system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=14,
            color=font_color,
        ),
        title=dict(
            text=title or fig.layout.title.text if fig.layout.title.text else title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=20, color=title_color, family='system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'),
            pad=dict(b=10),
        ),
        margin=dict(l=60, r=20, t=70, b=60),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=font_color),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=axis_color,
        tickcolor=axis_color,
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=axis_color,
        tickcolor=axis_color,
        ticks="outside",
    )
    return fig


def export_plot_html(
    fig: go.Figure,
    out_path: str | Path,
    *,
    title: str | None = None,
    width: int = 1100,
    height: int = 520,
    theme: str = "dark",
) -> None:
    fig = apply_blog_layout(fig, theme=theme, width=width, height=height, title=title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": False},
    )


def export_sector_figures(
    results: dict,
    out_dir: str | Path,
    prefix: str,
    *,
    theme: str = "dark",
    per_sector_subdir: bool = False,
) -> List[str]:
    """
    Export Chapter 5 figures to standalone HTML files.

    If per_sector_subdir=True, files are placed under out_dir/prefix/ as:
      plot1.html (UMAP), plot2.html (index), plot3.html (tracking error).
    Otherwise, they are named {prefix}_umap/index/te.html in out_dir.
    """
    fig_sizes = {
        # Standardized sizes to match site Chapter 3 cards
        "fig_umap": (900, 260),
        "fig_index": (900, 260),
        "fig_te": (900, 260),
    }
    if per_sector_subdir:
        base = Path(out_dir) / prefix
        names = {
            "fig_umap": "plot1.html",
            "fig_index": "plot2.html",
            "fig_te": "plot3.html",
        }
    else:
        base = Path(out_dir)
        names = {
            "fig_umap": f"{prefix}_umap.html",
            "fig_index": f"{prefix}_index.html",
            "fig_te": f"{prefix}_te.html",
        }

    exported: List[str] = []
    for key in ("fig_umap", "fig_index", "fig_te"):
        fig = results.get(key)
        if fig is None:
            continue
        width, height = fig_sizes[key]
        filename = names[key]
        export_plot_html(
            fig,
            base / filename,
            width=width,
            height=height,
            theme=theme,
            title=fig.layout.title.text,
        )
        exported.append(str((base / filename).relative_to(Path(out_dir))))
    return exported


def write_iframe_snippets(out_dir: str | Path, exported_files: List[str], *, iframe_height: int = 560) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snippet_lines = []
    for fname in exported_files:
        snippet_lines.append(
            f'<iframe src="{fname}"\\n'
            f'        style="width:100%; height:{iframe_height}px; border:0;"\\n'
            f'        loading="lazy"></iframe>\\n'
        )
    snippets_path = out_dir / "iframe_snippets.txt"
    snippets_path.write_text("\\n".join(snippet_lines))
    return snippets_path
def _build_index_from_returns(returns: pd.Series) -> pd.Series:
    use_log = (1 + returns.dropna()).min() <= 0
    if use_log:
        idx = np.exp(returns.fillna(0).cumsum())
    else:
        idx = (1 + returns.fillna(0)).cumprod()
    return 100 * idx


def plot_index_comparison(
    sector_proxy: pd.Series,
    single_stock_returns: pd.Series,
    basket_returns: pd.Series | None,
    sector_name: str,
    single_ticker: str,
    basket_tickers: List[str] | None,
):
    """
    Plot normalized index comparison:

    - Sector proxy (baseline)
    - Best single stock
    - Optionally Top-N basket

    All starting at base = 100.

    Returns a Plotly Figure.
    """
    series = {
        f"{sector_name} proxy": sector_proxy,
        f"{single_ticker} (single)": single_stock_returns,
    }
    if basket_returns is not None and basket_tickers:
        basket_label = f"Basket ({', '.join(basket_tickers)})"
        series[basket_label] = basket_returns

    aligned = pd.concat(series, axis=1).dropna(how="all")
    idx_df = aligned.apply(_build_index_from_returns)

    fig = go.Figure()
    for col in idx_df.columns:
        fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df[col], mode="lines", name=col))

    fig.update_layout(
        title=f"{sector_name}: Representative vs Sector Proxy (Index=100)",
        yaxis_title="Index (base = 100)",
        hovermode="x unified",
    )
    return fig


def plot_tracking_error_comparison(
    te_single_60: pd.Series,
    te_basket_60: pd.Series | None,
    te_single_120: pd.Series | None,
    te_basket_120: pd.Series | None,
    sector_name: str,
    single_ticker: str,
    y_range: Tuple[float, float] | None = None,
):
    """
    Plot tracking error over time for:

    - Single stock vs sector (60d and optionally 120d),
    - Basket vs sector (60d and optionally 120d).

    Returns a Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=te_single_60.index,
            y=te_single_60,
            mode="lines",
            name=f"{single_ticker} TE (60d)",
        )
    )

    if te_basket_60 is not None:
        fig.add_trace(
            go.Scatter(
                x=te_basket_60.index,
                y=te_basket_60,
                mode="lines",
                name="Basket TE (60d)",
            )
        )

    if te_single_120 is not None:
        fig.add_trace(
            go.Scatter(
                x=te_single_120.index,
                y=te_single_120,
                mode="lines",
                name=f"{single_ticker} TE (120d)",
                line=dict(dash="dash"),
            )
        )

    if te_basket_120 is not None:
        fig.add_trace(
            go.Scatter(
                x=te_basket_120.index,
                y=te_basket_120,
                mode="lines",
                name="Basket TE (120d)",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{sector_name}: Tracking Error vs Sector Proxy",
        yaxis_title="Tracking Error",
        hovermode="x unified",
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


# --------------------------------------------------------------------
# 6. Blog/iframe export helpers (styling + HTML)
# --------------------------------------------------------------------

def apply_blog_layout(
    fig: go.Figure,
    *,
    theme: str = "dark",
    width: int = 1100,
    height: int = 520,
    title: str | None = None,
) -> go.Figure:
    """
    Apply a site-consistent, iframe-ready layout to any Plotly figure.

    - Dark focus visual with muted gridlines.
    - System font stack to match site typography.
    - Left-aligned titles, soft legends, no mode bar clutter.
    """
    if fig is None:
        return fig

    is_dark = theme == "dark"
    paper_bg = "#0f172a" if is_dark else "#f8fafc"
    plot_bg = "#0b1220" if is_dark else "#ffffff"
    font_color = "#e5e7eb" if is_dark else "#0f172a"
    grid_color = "#1f2937" if is_dark else "#e5e7eb"
    axis_color = "#cbd5e1" if is_dark else "#475569"
    title_color = font_color

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(
            family='system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            size=14,
            color=font_color,
        ),
        title=dict(
            text=title or fig.layout.title.text if fig.layout.title.text else title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=20, color=title_color, family='system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'),
            pad=dict(b=10),
        ),
        margin=dict(l=60, r=20, t=70, b=60),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=font_color),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=axis_color,
        tickcolor=axis_color,
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zeroline=False,
        linecolor=axis_color,
        tickcolor=axis_color,
        ticks="outside",
    )
    return fig


def export_plot_html(
    fig: go.Figure,
    out_path: str | Path,
    *,
    title: str | None = None,
    width: int = 500,
    height: int = 250,
    theme: str = "dark",
) -> None:
    """
    Apply blog layout and export a standalone, iframe-ready HTML for a figure.
    """
    fig = apply_blog_layout(fig, theme=theme, width=width, height=height, title=title)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": False},
    )


def export_sector_figures(results: dict, out_dir: str | Path, prefix: str, *, theme: str = "dark") -> List[str]:
    """
    Export available Chapter 2 figures as standalone HTML files (iframe-ready).

    Returns list of exported filenames.
    """
    fig_sizes = {
        "fig_umap": (500, 250),
        "fig_index": (500, 250),
        "fig_te": (500, 250),
    }
    names = {
        "fig_umap": f"{prefix}_umap.html",
        "fig_index": f"{prefix}_index.html",
        "fig_te": f"{prefix}_te.html",
    }
    out_dir = Path(out_dir)
    exported: List[str] = []
    for key in ("fig_umap", "fig_index", "fig_te"):
        fig = results.get(key)
        if fig is None:
            continue
        width, height = fig_sizes[key]
        filename = names[key]
        export_plot_html(
            fig,
            out_dir / filename,
            width=width,
            height=height,
            theme=theme,
            title=fig.layout.title.text,
        )
        exported.append(filename)
    return exported


def write_iframe_snippets(out_dir: str | Path, exported_files: List[str], *, iframe_height: int = 560) -> Path:
    """
    Write ready-to-paste iframe HTML snippets for the exported plot files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snippet_lines = []
    for fname in exported_files:
        snippet_lines.append(
            f'<iframe src="{fname}"\n'
            f'        style="width:100%; height:{iframe_height}px; border:0;"\n'
            f'        loading="lazy"></iframe>\n'
        )
    snippets_path = out_dir / "iframe_snippets.txt"
    snippets_path.write_text("\n".join(snippet_lines))
    return snippets_path


# --------------------------------------------------------------------
# 7. High-level orchestrator (per sector)
# --------------------------------------------------------------------

def run_sector_analysis(
    df_stock_sect: pd.DataFrame,
    sector: str,
    return_col: str = "Return",
    window_rep: int = 60,
    te_windows: Tuple[int, int] = (60, 120),
    min_overlap_corr: int = 2,
    distance_scale: float = 2.0,
    cluster_method: str = "agglomerative",
    cluster_n: int | None = None,
    cluster_distance_threshold: float | None = None,
    linkage: Literal["average", "complete", "single"] = "complete",
    threshold_percentile: float = 30.0,
    distance_metric: Literal["linear", "angular"] = "linear",
    nan_distance_fill: Literal["max", "p90", "median"] = "max",
    max_nan_frac: float = 0.25,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
    missingdata_spread: float = 5.0,
    missingdata_jitter: float = 0.2,
    umap_n_neighbors: int = 10,
    umap_min_dist: float = 0.1,
    lifetime_alpha: float = 2.0,
    min_lifetime_fraction: float | None = None,
    te_y_range: Tuple[float, float] | None = (0, 0.5),
) -> Dict[str, object]:
    """
    High-level pipeline to run the full Chapter 2 analysis for a single sector.

    Steps:
    - build sector panel (returns_wide, sector_proxy, volume_wide),
    - filter illiquid stocks,
    - compute representativeness scores,
    - compute risk metrics,
    - compute intra-sector correlation using pairwise-overlap windows,
    - isolate tickers with excessive missing overlaps (cluster -99 "MissingData"),
    - run UMAP embedding,
    - cluster stocks (agglomerative with distance threshold or HDBSCAN),
    - select best single stock and Top-5 basket,
    - compute basket returns and tracking error for both proxies,
    - create core plots (UMAP, index comparison, TE comparison).

    Returns a dict containing the intermediate results and Plotly figures.
    """
    returns_wide, sector_proxy, volume_wide = prepare_sector_panel(
        df_stock_sect, sector, use_return_col=return_col
    )

    kept = filter_liquid_stocks(returns_wide, volume_wide)
    if kept:
        returns_wide = returns_wide[kept]
        volume_wide = volume_wide[kept]
    else:
        # No liquid names for this sector; mark as empty to short-circuit downstream steps.
        returns_wide = returns_wide.iloc[:, 0:0]
        volume_wide = volume_wide.iloc[:, 0:0]

    scores_df = compute_representativeness_scores(returns_wide, sector_proxy, window=window_rep)
    risk_df = compute_risk_metrics(returns_wide)
    lifetime_fraction = compute_lifetime_fraction(returns_wide)

    top_tickers = select_top_representative_stocks(
        scores_df,
        risk_df,
        lifetime_fraction,
        top_n=5,
        alpha=lifetime_alpha,
        min_lifetime_fraction=min_lifetime_fraction,
    )
    single_ticker = top_tickers[0] if top_tickers else None
    n_tickers = returns_wide.shape[1]

    if n_tickers == 0:
        corr = pd.DataFrame()
        embedding_df = pd.DataFrame(columns=["UMAP 1", "UMAP 2"])
        cluster_labels = pd.Series(dtype=int)
        single_returns = pd.Series(dtype=float)
        basket_returns = pd.Series(dtype=float)
        missing_tickers: List[str] = []
    else:
        corr = compute_intra_sector_correlation(returns_wide, min_overlap=min_overlap_corr)
        denom = max(1, len(corr.columns) - 1)
        nan_frac = corr.isna().sum(axis=1) / denom
        missing_tickers = nan_frac[nan_frac > max_nan_frac].index.tolist()
        good_tickers = [t for t in corr.index if t not in set(missing_tickers)]

        corr_good = corr.loc[good_tickers, good_tickers] if good_tickers else pd.DataFrame()
        embedding_df = run_umap_embedding(
            corr_good,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            distance_scale=distance_scale,
            distance_metric=distance_metric,
            nan_distance_fill=nan_distance_fill,
            missing_tickers=missing_tickers,
            missingdata_spread=missingdata_spread,
            missingdata_jitter=missingdata_jitter,
        ) if (len(good_tickers) + len(missing_tickers)) > 0 else pd.DataFrame(columns=["UMAP 1", "UMAP 2"])

        cluster_labels = cluster_stocks(
            corr,
            method=cluster_method,
            n_clusters=cluster_n,
            distance_threshold=cluster_distance_threshold,
            distance_scale=distance_scale,
            linkage=linkage,
            threshold_percentile=threshold_percentile,
            distance_metric=distance_metric,
            nan_distance_fill=nan_distance_fill,
            max_nan_frac=max_nan_frac,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            missing_tickers=missing_tickers,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
        )
        single_returns = returns_wide[single_ticker] if single_ticker is not None else pd.Series(dtype=float)
        basket_returns = build_basket_returns(returns_wide, top_tickers) if top_tickers else pd.Series(dtype=float)

    te_single_60 = compute_tracking_error_series(single_returns, sector_proxy, window=te_windows[0])
    te_basket_60 = compute_tracking_error_series(basket_returns, sector_proxy, window=te_windows[0]) if not basket_returns.empty else None

    te_single_120 = compute_tracking_error_series(single_returns, sector_proxy, window=te_windows[1]) if te_windows and len(te_windows) > 1 else None
    te_basket_120 = compute_tracking_error_series(basket_returns, sector_proxy, window=te_windows[1]) if (te_windows and len(te_windows) > 1 and not basket_returns.empty) else None

    avg_volume = volume_wide.mean()
    umap_fig = plot_sector_umap(
        embedding_df,
        cluster_labels,
        scores_df,
        avg_volume,
        top_tickers,
        sector,
    ) if not embedding_df.empty else None

    idx_fig = plot_index_comparison(
        sector_proxy,
        single_returns,
        basket_returns if not basket_returns.empty else None,
        sector,
        single_ticker or "",
        top_tickers if top_tickers else None,
    )

    te_fig = plot_tracking_error_comparison(
        te_single_60,
        te_basket_60,
        te_single_120,
        te_basket_120,
        sector,
        single_ticker or "",
        y_range=te_y_range,
    )

    return {
        "returns_wide": returns_wide,
        "sector_proxy": sector_proxy,
        "volume_wide": volume_wide,
        "kept_tickers": kept,
        "scores": scores_df,
        "risk": risk_df,
        "corr": corr,
        "embedding": embedding_df,
        "clusters": cluster_labels,
        "top_tickers": top_tickers,
        "single_ticker": single_ticker,
        "single_returns": single_returns,
        "basket_returns": basket_returns,
        "te_single_60": te_single_60,
        "te_basket_60": te_basket_60,
        "te_single_120": te_single_120,
        "te_basket_120": te_basket_120,
        "fig_umap": umap_fig,
        "fig_index": idx_fig,
        "fig_te": te_fig,
        "missing_tickers": missing_tickers,
        "lifetime_fraction": lifetime_fraction,
        "lifetime_alpha": lifetime_alpha,
        "effective_score": scores_df["rep_score"] * (lifetime_fraction ** lifetime_alpha),
    }
