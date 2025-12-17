"""
Chapter 2 – Intra-Sector Structure & Representativeness

This module provides all functions needed to:
- load and prepare stock/sector data,
- compute sector proxies,
- filter illiquid stocks,
- compute stock representativeness scores,
- cluster stocks within a sector,
- build single-stock and Top-5 baskets,
- compute tracking error and risk metrics,
- generate the key plots (UMAP, index comparison, tracking error).

All functions are designed to be called from a notebook.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
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
) -> Dict[str, pd.DataFrame]:
    """
    Load all necessary parquet files for Chapter 2.

    Returns a dict with:
        {
            "stocks": df_stocks_clean,
            "etfs": df_etfs_clean,
            "meta": df_meta,
            "stocks_sect": df_stock_sect
        }
    """
    root = Path(data_dir)
    files = {
        "stocks": root / "stocks_clean.parquet",
        "etfs": root / "etfs_clean.parquet",
        "meta": root / "metadata_clean.parquet",
        "stocks_sect": root / "stocks_sect.parquet",
    }

    data = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")
        data[key] = pd.read_parquet(path)
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
        Full stocks_sect dataframe.
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
    volume_positive_frac = (volume_wide > 0).mean(axis=0)
    liquid_candidates = volume_positive_frac[volume_positive_frac >= min_nonzero_fraction].index
    if len(liquid_candidates) == 0:
        return []

    avg_volume = volume_wide[liquid_candidates].mean(axis=0)
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
    min_overlap: int = 30,
) -> pd.DataFrame:
    """
    Compute the stock–stock correlation matrix using all available history
    per pair (pairwise overlap), not only the common intersection across
    all tickers.

    For each pair (i, j):
        - find dates where both returns are non-NaN,
        - if overlap >= min_overlap, compute Pearson correlation on that window,
        - otherwise return NaN for that pair.

    Returns
    -------
    corr : DataFrame
        Ticker x ticker correlation matrix.
    """
    tickers = list(returns_wide.columns)
    n = len(tickers)
    corr = pd.DataFrame(np.eye(n), index=tickers, columns=tickers, dtype=float)

    for i in range(n):
        s_i = returns_wide.iloc[:, i]
        for j in range(i + 1, n):
            s_j = returns_wide.iloc[:, j]
            mask = s_i.notna() & s_j.notna()
            if mask.sum() >= min_overlap:
                rho = s_i[mask].corr(s_j[mask])
            else:
                rho = np.nan
            corr.iat[i, j] = rho
            corr.iat[j, i] = rho

    return corr


def _corr_to_distance(corr: pd.DataFrame, scale: float = 2.0) -> pd.DataFrame:
    """
    Convert correlation matrix to a distance matrix.

    Default: d_ij = scale * (1 - rho_ij) with rho in [-1,1].
    """
    distance = scale * (1 - corr)
    np.fill_diagonal(distance.values, 0.0)
    return distance.clip(lower=0)


def run_umap_embedding(
    corr: pd.DataFrame,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    random_state: int = 42,
    distance_scale: float = 2.0,
) -> pd.DataFrame:
    """
    Run UMAP on the distance matrix derived from correlations.

    Distance = scale * (1 - corr_ij) with default scale=2.

    Returns
    -------
    embedding_df : DataFrame
        Index = ticker, columns = ['x', 'y'] with 2D UMAP coordinates.
    """
    if umap is None:
        raise ImportError("UMAP is not installed. Install `umap-learn` to use this function.")

    distance = _corr_to_distance(corr.fillna(0), scale=distance_scale)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(distance)
    embedding_df = pd.DataFrame(embedding, index=corr.index, columns=["x", "y"])
    return embedding_df


def cluster_stocks(
    corr: pd.DataFrame,
    method: str = "agglomerative",
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    distance_scale: float = 2.0,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
) -> pd.Series:
    """
    Cluster stocks within a sector based on their correlation structure.

    Distance = scale * (1 - correlation).

    Supported methods:
    - 'agglomerative' (default)
    - 'hdbscan' (density-based, requires the `hdbscan` package)

    Returns
    -------
    cluster_labels : Series
        Index = ticker, values = integer cluster labels.
    """
    distance = _corr_to_distance(corr.fillna(0), scale=distance_scale)

    if method == "hdbscan":
        if hdbscan is None:
            raise ImportError("HDBSCAN is not installed. Install `hdbscan` to use this clustering method.")
        model = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )
        labels = model.fit_predict(distance)
        return pd.Series(labels, index=corr.index, name="cluster")

    if method != "agglomerative":
        raise NotImplementedError(f"Clustering method '{method}' is not supported.")

    # Default to distance-threshold mode to let the hierarchy decide cluster count.
    if distance_threshold is None and n_clusters is None:
        distance_threshold = distance_scale

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(distance)
    return pd.Series(labels, index=corr.index, name="cluster")


# --------------------------------------------------------------------
# 4. Basket construction & tracking error
# --------------------------------------------------------------------

def select_top_representative_stocks(
    scores_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    top_n: int = 5,
) -> List[str]:
    """
    Select the Top-N representative stocks in a sector.

    Base rule:
    - Rank by 'rep_score' (higher is better).
    - Break ties by lower volatility (from risk_df).

    Returns
    -------
    top_tickers : list of str
        Selected tickers.
    """
    merged = scores_df.join(risk_df[["vol"]], how="inner")
    ranked = merged.sort_values(by=["rep_score", "vol"], ascending=[False, True])
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
    aligned_proxy, aligned_sector = proxy_returns.align(sector_proxy, join="inner")
    active_return = aligned_proxy - aligned_sector
    return active_return.rolling(window=window, min_periods=min_periods).std()


# --------------------------------------------------------------------
# 5. Plotting utilities (Plotly, interactive)
# --------------------------------------------------------------------

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
    cluster_display = clusters_raw.astype("Int64").astype("string")
    cluster_display = cluster_display.fillna("Unassigned").replace({"<NA>": "Unassigned", "-1": "Noise"})
    df_plot["cluster"] = cluster_display
    df_plot["rep_score"] = scores_df.reindex(tickers)["rep_score"]
    df_plot["avg_volume"] = avg_volume.reindex(tickers)
    df_plot["in_basket"] = df_plot["ticker"].isin(basket_tickers)

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
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
        fig.add_trace(
            go.Scatter(
                x=basket_df["x"],
                y=basket_df["y"],
                mode="markers+text",
                name="Top basket",
                text=basket_df["ticker"],
                textposition="top center",
                marker_symbol="star",
                marker=dict(
                    size=np.clip(basket_df["avg_volume"] / basket_df["avg_volume"].max() * 30, 10, 30),
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
    return fig


# --------------------------------------------------------------------
# 6. High-level orchestrator (per sector)
# --------------------------------------------------------------------

def run_sector_analysis(
    df_stock_sect: pd.DataFrame,
    sector: str,
    return_col: str = "Return",
    window_rep: int = 60,
    te_windows: Tuple[int, int] = (60, 120),
    min_overlap_corr: int = 30,
    distance_scale: float = 2.0,
    cluster_method: str = "agglomerative",
    cluster_distance_threshold: float | None = None,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
) -> Dict[str, object]:
    """
    High-level pipeline to run the full Chapter 2 analysis for a single sector.

    Steps:
    - build sector panel (returns_wide, sector_proxy, volume_wide),
    - filter illiquid stocks,
    - compute representativeness scores,
    - compute risk metrics,
    - compute intra-sector correlation using pairwise-overlap windows,
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

    scores_df = compute_representativeness_scores(returns_wide, sector_proxy, window=window_rep)
    risk_df = compute_risk_metrics(returns_wide)

    corr = compute_intra_sector_correlation(returns_wide, min_overlap=min_overlap_corr)
    embedding_df = run_umap_embedding(corr, distance_scale=distance_scale) if len(corr) >= 2 else pd.DataFrame(columns=["x", "y"])
    cluster_labels = (
        cluster_stocks(
            corr,
            method=cluster_method,
            distance_scale=distance_scale,
            distance_threshold=cluster_distance_threshold,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
        )
        if len(corr) >= 2
        else pd.Series(dtype=int, index=corr.index)
    )

    top_tickers = select_top_representative_stocks(scores_df, risk_df, top_n=5)
    single_ticker = top_tickers[0] if top_tickers else None

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
    }
