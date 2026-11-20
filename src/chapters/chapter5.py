"""
Chapter 5 (sector proxy vs representative stocks) helpers for `chapter5_sector_proxy.ipynb`.

For each sector: can one stock, or a small basket, stand in for the whole sector?
- the sector proxy is the dollar-volume weighted sector return (from `stocks_sect`);
- stocks are ranked by how closely they follow the proxy over time;
- the best stock and the Top-5 basket are compared with the proxy (index and tracking error);
- a UMAP map with HDBSCAN clusters shows whether the sector is one block or several groups.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.figures import PLOTLY_THEMES


# ---------- Sector panel ----------

def sector_panel(stocks_sect: pd.DataFrame, sector: str, start: str, end: str):
    """
    Wide tables for one sector between `start` and `end`.

    Returns (returns_wide, dollar_volume_wide, proxy): stock returns and dollar
    volumes (index = Date, one column per stock) and the sector proxy return
    (sum of Weight × Return, i.e. the dollar-volume weighted sector return).
    """
    d = stocks_sect[(stocks_sect["Sector"] == sector) & (stocks_sect["Date"] >= start) & (stocks_sect["Date"] <= end)]
    returns = d.pivot(index="Date", columns="Symbol", values="Return").sort_index()
    dollar_volume = d.pivot(index="Date", columns="Symbol", values="DollarVolume").sort_index()
    proxy = (d["Weight"] * d["Return"]).groupby(d["Date"]).sum(min_count=1).sort_index().rename("proxy")
    return returns, dollar_volume, proxy


def liquid_stocks(dollar_volume: pd.DataFrame, min_trading_frac: float = 0.8, min_quantile: float = 0.2) -> list[str]:
    """
    Stocks that trade on at least `min_trading_frac` of their days and whose average
    dollar volume is above the sector's `min_quantile` quantile.
    """
    trading_frac = (dollar_volume > 0).sum() / dollar_volume.notna().sum().replace(0, np.nan)
    candidates = trading_frac[trading_frac >= min_trading_frac].index
    avg_dv = dollar_volume[candidates].mean()
    return avg_dv[avg_dv >= avg_dv.quantile(min_quantile)].index.tolist()


# ---------- Representativeness ----------

def representativeness(returns: pd.DataFrame, proxy: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Average rolling correlation between each stock and the sector proxy."""
    returns, proxy = returns.align(proxy, axis=0, join="inner")
    return returns.rolling(window, min_periods=min_periods).corr(proxy).mean().rename("rep_score")


def company_key(security_name) -> str:
    """
    Company part of a security name, so that share classes of one company match:
    "Alphabet Inc. - Class C Capital Stock" and "Alphabet Inc. - Class A Common
    Stock" both give "alphabet inc".
    """
    name = str(security_name).split(" - ")[0]
    name = re.split(r"\b(class|series) [a-z]\b", name, flags=re.IGNORECASE)[0]
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def select_representatives(
    returns: pd.DataFrame, rep_score: pd.Series, top_n: int, lifetime_alpha: float, company: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Rank stocks by rep_score × lifetime^alpha, where lifetime is the share of the
    period with a return: a stock that only exists for a short time cannot stand
    in for the sector. Ties are broken by lower volatility.

    `company` maps each symbol to its company (see `company_key`). When given,
    the top `top_n` are distinct companies: a second share class of a company
    already selected (e.g. GOOGL after GOOG) is skipped.

    Returns the ranking table (best first), with the top `top_n` marked.
    """
    table = pd.DataFrame({
        "rep_score": rep_score,
        "lifetime": returns.notna().mean(),
        "vol": returns.std(),
    }).dropna(subset=["rep_score"])
    table["score"] = table["rep_score"] * table["lifetime"] ** lifetime_alpha
    table = table.sort_values(["score", "vol"], ascending=[False, True])
    table["selected"] = False
    keys = company.reindex(table.index).fillna(pd.Series(table.index, index=table.index)) if company is not None \
        else pd.Series(table.index, index=table.index)
    chosen, seen = [], set()
    for symbol in table.index:
        if len(chosen) == top_n:
            break
        if keys[symbol] not in seen:
            chosen.append(symbol)
            seen.add(keys[symbol])
    table.loc[chosen, "selected"] = True
    return table


def equal_weight_returns(returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weighted return of `tickers`, among those that have a return each day."""
    return returns[tickers].mean(axis=1, skipna=True).rename("basket")


# ---------- Tracking error ----------

def rolling_tracking_error(r: pd.Series, proxy: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Annualised rolling standard deviation of (r - proxy)."""
    r, proxy = r.align(proxy, join="inner")
    return (r - proxy).rolling(window, min_periods=min_periods).std() * np.sqrt(252)


def comparison_stats(r: pd.Series, proxy: pd.Series, te: pd.Series) -> dict:
    """Correlation with the proxy, annualised tracking error, and p95/max of the rolling TE."""
    r, proxy = r.align(proxy, join="inner")
    ok = r.notna() & proxy.notna()
    return {
        "corr": float(r[ok].corr(proxy[ok])),
        "te_ann": float((r[ok] - proxy[ok]).std() * np.sqrt(252)),
        "te_p95": float(te.quantile(0.95)),
        "te_max": float(te.max()),
    }


# ---------- Structure: correlations, UMAP, clusters ----------

def stock_correlations(returns: pd.DataFrame, min_overlap: int) -> pd.DataFrame:
    """Pairwise correlations, each pair using all the days where both stocks trade (at least `min_overlap`)."""
    return returns.corr(min_periods=min_overlap)


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Angular distance sqrt(2 (1 - rho)). Missing correlations get the largest
    distance (2), so stocks with no common history are never pulled together.
    """
    values = np.sqrt(2 * (1 - corr.clip(-1, 1).to_numpy(dtype=float, copy=True)))
    values = np.nan_to_num(values, nan=2.0)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def umap_embedding(distance: pd.DataFrame, n_neighbors: int, min_dist: float, seed: int = 42) -> pd.DataFrame:
    """2-D UMAP map of the stocks from a precomputed distance matrix."""
    import warnings

    with warnings.catch_warnings():
        # umap's import and fit print notices that do not apply here (no ipywidgets
        # progress bar, no inverse_transform with a precomputed metric)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="IProgress not found")
        import umap  # imported here: umap is slow to import and only needed for this step

        coords = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, metric="precomputed",
            random_state=seed, n_jobs=1, compatibility_layout=True,
        ).fit_transform(distance.values)
    return pd.DataFrame(coords, index=distance.index, columns=["UMAP 1", "UMAP 2"])


def hdbscan_clusters(embedding: pd.DataFrame, min_cluster_size: int) -> pd.Series:
    """
    HDBSCAN clusters on the UMAP map, renamed by size: "Cluster 1" is the largest;
    points HDBSCAN cannot assign are "Outlier".
    """
    from sklearn.cluster import HDBSCAN

    raw = pd.Series(HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(embedding.values), index=embedding.index)
    sizes = raw[raw >= 0].value_counts()
    names = {c: f"Cluster {i + 1} ({n} stocks)" for i, (c, n) in enumerate(sizes.items())}
    return raw.map(names).fillna("Outlier").rename("cluster")


# ---------- Figures (website versions use theme="dark") ----------

def _layout(fig: go.Figure, title: str, theme: str, **axes) -> go.Figure:
    t = PLOTLY_THEMES[theme]
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font=dict(color=t["text"], size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=55, r=15, t=55, b=45), **axes,
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False)
    return fig


def plot_umap(embedding: pd.DataFrame, clusters: pd.Series, ranking: pd.DataFrame,
              avg_dollar_volume: pd.Series, sector: str, theme: str = "light") -> go.Figure:
    """Stocks on the UMAP map: colour = cluster, size = average dollar volume, stars = Top-5 basket."""
    df = embedding.copy()
    df["ticker"] = df.index
    df["cluster"] = clusters.reindex(df.index)
    df["rep_score"] = ranking["rep_score"].reindex(df.index).round(2)
    df["size"] = np.sqrt(avg_dollar_volume.reindex(df.index).fillna(0))
    order = sorted(df["cluster"].unique(), key=lambda c: (c == "Outlier", c))
    fig = px.scatter(df, x="UMAP 1", y="UMAP 2", color="cluster", size="size", size_max=18,
                     hover_data={"ticker": True, "rep_score": True, "cluster": True, "size": False},
                     category_orders={"cluster": order})
    top = df[df.index.isin(ranking.index[ranking["selected"]])]
    fig.add_trace(go.Scatter(x=top["UMAP 1"], y=top["UMAP 2"], mode="markers+text", text=top["ticker"],
                             textposition="top center", name="Top-5 basket", hoverinfo="skip",
                             marker=dict(symbol="star", size=14, color=PLOTLY_THEMES[theme]["text"])))
    return _layout(fig, f"{sector}: stocks on the UMAP map", theme, legend_title_text="Cluster")


def plot_index(proxy: pd.Series, single: pd.Series, basket: pd.Series, single_ticker: str,
               basket_tickers: list[str], sector: str, theme: str = "light") -> go.Figure:
    """Sector proxy, single stock and basket, each rebased to 100."""
    series = {f"{sector} proxy": proxy, f"{single_ticker} (single)": single,
              f"Basket ({', '.join(basket_tickers)})": basket}
    index = (100 * (1 + pd.concat(series, axis=1).fillna(0)).cumprod()).round(2)
    fig = go.Figure([go.Scatter(x=index.index, y=index[c], mode="lines", name=c) for c in index.columns])
    return _layout(fig, f"{sector}: representative vs sector proxy (index = 100)", theme,
                   yaxis_title="Index (base = 100)", hovermode="x unified")


def plot_tracking_error(te: dict[str, pd.Series], sector: str, theme: str = "light") -> go.Figure:
    """Rolling annualised tracking error of the single stock and the basket against the proxy."""
    fig = go.Figure()
    for name, s in te.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.round(4), mode="lines", name=name,
                                 line=dict(dash="dash" if "120" in name else "solid")))
    return _layout(fig, f"{sector}: tracking error vs sector proxy", theme,
                   yaxis_title="Tracking error (annualised)", hovermode="x unified")
