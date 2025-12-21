"""Helper functions for Chapter 3 / Question 3.

This module is designed to be imported from a notebook, e.g.:

    import Helper_3 as h
    meta, ret_tidy, stocks, etf_ret = h.load_chapter3_data("data_processed")
    results_df, series = h.build_sector_series(meta, ret_tidy, stocks, etf_ret)

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------------------------------------------------------
# Plot style (dark theme)
# -----------------------------------------------------------------------------
DARK_BG: str = "#0b0b0b"
GRID_COL: str = "#2a2a2a"
TEXT_COL: str = "#d9d9d9"
TICK_COL: str = "#bfbfbf"

C_60D: str = "#5b7cfa"   # light blue-ish
C_120D: str = "#8d5cf7"  # purple-ish


def apply_dark_style(fig, ax, show_zero_line: bool = False):
    """Apply the notebook's dark style to an axes."""
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.grid(True, which="major", color=GRID_COL, alpha=0.55, linewidth=1.0)
    ax.grid(False, which="minor")

    for s in ax.spines.values():
        s.set_visible(False)

    ax.tick_params(axis="both", colors=TICK_COL, length=0)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)

    if show_zero_line:
        ax.axhline(0, color=GRID_COL, linewidth=1.0, alpha=0.8, zorder=0)

    leg = ax.legend(loc="upper left", frameon=False)
    for t in leg.get_texts():
        t.set_color(TEXT_COL)

    return ax


# -----------------------------------------------------------------------------
# Sector universe mapping (SPDR sector ETFs)
# -----------------------------------------------------------------------------
SECTOR_ETFS: Dict[str, str] = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def load_chapter3_data(data_dir: str | os.PathLike = "data_processed"):
    """Load parquet files used in Question 3 and parse dates.

    Returns
    -------
    meta : pd.DataFrame
        Must include columns: ['ticker', 'sector', 'etf'] (etf can be bool-like).
    ret_tidy : pd.DataFrame
        Must include columns: ['Date', 'Symbol', 'log_return'].
    stocks : pd.DataFrame
        Must include columns: ['Date', 'Symbol', 'Volume'].
    etf_ret : pd.DataFrame
        Wide dataframe of ETF returns, datetime index, columns are tickers.
    """
    data_dir = Path(data_dir)

    meta = pd.read_parquet(data_dir / "meta_clean.parquet")
    ret_tidy = pd.read_parquet(
        data_dir / "returns_tidy.parquet",
        columns=["Date", "Symbol", "log_return"],
    )
    stocks = pd.read_parquet(
        data_dir / "stocks_clean.parquet",
        columns=["Date", "Symbol", "Volume"],
    )
    etf_ret = pd.read_parquet(data_dir / "etf_returns.parquet")

    ret_tidy["Date"] = pd.to_datetime(ret_tidy["Date"])
    stocks["Date"] = pd.to_datetime(stocks["Date"])
    etf_ret.index = pd.to_datetime(etf_ret.index)

    return meta, ret_tidy, stocks, etf_ret


def is_etf_flag(x) -> bool:
    """Interpret different 'truthy' ETF flags found in metadata."""
    if pd.isna(x):
        return False
    if isinstance(x, str):
        return x.strip().upper() in {"Y", "YES", "TRUE", "T", "1", "ETF"}
    return bool(x)


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------
def analyze_sector(
    *,
    meta: pd.DataFrame,
    ret_tidy: pd.DataFrame,
    stocks: pd.DataFrame,
    etf_ret: pd.DataFrame,
    sector_name: str,
    etf_ticker: str,
    cap: Optional[float] = 0.10,
    lag: int = 1,
    windows: Sequence[int] = (60, 120),
    bad_ret_thresh: float = 1.0,
    min_good_frac: float = 0.98,
) -> Tuple[Optional[dict], pd.DataFrame | str]:
    """Build a volume-weighted sector proxy and compare it to the sector ETF.

    Notes
    -----
    - Builds a volume-share weighted basket of sector constituents (non-ETFs).
    - Uses lagged weights (t-1) and optional weight cap, then re-normalizes.
    - Returns a metrics dict and a dataframe with columns ['ETF','Basket','Active'].

    Returns
    -------
    (out, df) if successful
    (None, reason) if not possible for this sector
    """
    # 1) universe = stocks in sector (exclude ETFs)
    tickers = (
        meta.loc[
            meta["sector"].astype(str).str.lower().eq(sector_name.lower())
            & (~meta["etf"].map(is_etf_flag)),
            "ticker",
        ]
        .dropna()
        .unique()
    )

    # 2) pull returns + volume for those tickers (long -> wide)
    r = ret_tidy[ret_tidy["Symbol"].isin(tickers)].dropna()
    v = stocks[stocks["Symbol"].isin(tickers)].dropna()

    r_w = r.pivot(index="Date", columns="Symbol", values="log_return").sort_index()
    v_w = v.pivot(index="Date", columns="Symbol", values="Volume").sort_index()

    cols = r_w.columns.intersection(v_w.columns)
    if len(cols) < 2:
        return None, (
            f"Too few tickers with both returns+volume for sector='{sector_name}' (got {len(cols)})."
        )

    r_w = r_w[cols]
    v_w = v_w[cols]

    # 3) ETF series
    if etf_ticker not in etf_ret.columns:
        return None, f"ETF '{etf_ticker}' not in etf_returns columns."

    r_etf = etf_ret[etf_ticker].dropna()

    # 4) align dates for basket construction & ETF comparison
    dates = r_w.index.intersection(v_w.index).intersection(r_etf.index)
    r_w = r_w.loc[dates]
    v_w = v_w.loc[dates]
    r_etf = r_etf.loc[dates]

    # optional filtering (bad returns or too-much missing)
    bad_any = (r_w.abs() > bad_ret_thresh).any(axis=0)
    good_frac = r_w.notna().mean(axis=0)
    bad_missing = good_frac < min_good_frac

    bad = bad_any | bad_missing
    if bad.any():
        r_w = r_w.loc[:, ~bad]
        v_w = v_w.loc[:, r_w.columns]  # keep aligned

    # 5) weights (volume share), then lag + cap
    vol_sum = v_w.sum(axis=1).replace(0, np.nan)
    w = v_w.div(vol_sum, axis=0).fillna(0.0)

    w_use = w.shift(lag).fillna(0.0)

    if cap is not None:
        w_use = w_use.clip(upper=cap)
        w_use = w_use.div(w_use.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # remove days where weights are meaningless (all zero)
    valid = w_use.sum(axis=1) > 0
    w_use = w_use.loc[valid]
    r_w = r_w.loc[valid]
    r_etf = r_etf.loc[valid]

    # 6) basket + df
    r_basket = (w_use * r_w).sum(axis=1)
    df = pd.DataFrame({"ETF": r_etf, "Basket": r_basket}).dropna()
    df["Active"] = df["ETF"] - df["Basket"]

    # 7) metrics
    out: dict = {}
    out["sector"] = sector_name
    out["etf"] = etf_ticker
    out["N_constituents"] = int(r_w.shape[1])  # filtered count

    den = w_use.pow(2).sum(axis=1)
    neff = (1.0 / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    out["Neff_median"] = float(neff.median()) if len(neff) else np.nan

    max_w = w_use.max(axis=1)
    out["max_weight_median"] = float(max_w.median()) if len(max_w) else np.nan
    out["max_weight_99pct"] = float(max_w.quantile(0.99)) if len(max_w) else np.nan

    out["corr_full"] = float(df["ETF"].corr(df["Basket"]))
    out["active_mean_daily"] = float(df["Active"].mean())
    out["active_std_daily"] = float(df["Active"].std())
    out["te_ann"] = float(df["Active"].std() * np.sqrt(252))

    for wlen in windows:
        out[f"te_{wlen}_ann_median"] = float(
            df["Active"].rolling(wlen).std().median() * np.sqrt(252)
        )
        out[f"corr_{wlen}_median"] = float(
            df["ETF"].rolling(wlen).corr(df["Basket"]).median()
        )

    return out, df


def build_sector_series(
    meta: pd.DataFrame,
    ret_tidy: pd.DataFrame,
    stocks: pd.DataFrame,
    etf_ret: pd.DataFrame,
    sector_etfs: Mapping[str, str] = SECTOR_ETFS,
    **analyze_kwargs,
):
    """Run analyze_sector across all sectors.

    Returns
    -------
    results_df : pd.DataFrame
        One row per sector (metrics).
    series : Dict[str, pd.DataFrame]
        Per-sector dataframe with columns ['ETF','Basket','Active'].
    """
    results = []
    series: Dict[str, pd.DataFrame] = {}

    for sector_name, etf_ticker in sector_etfs.items():
        out, df_or_reason = analyze_sector(
            meta=meta,
            ret_tidy=ret_tidy,
            stocks=stocks,
            etf_ret=etf_ret,
            sector_name=sector_name,
            etf_ticker=etf_ticker,
            **analyze_kwargs,
        )
        if out is None:
            # keep a record (optional)
            results.append({"sector": sector_name, "etf": etf_ticker, "error": df_or_reason})
            continue

        results.append(out)
        series[sector_name] = df_or_reason  # type: ignore[assignment]

    results_df = pd.DataFrame(results)
    return results_df, series


# -----------------------------------------------------------------------------
# Plotting helpers (pretty versions used in your notebook)
# -----------------------------------------------------------------------------
def plot_volatility_pretty(
    sector_name: str,
    df: pd.DataFrame,
    win: int = 60,
    etf_label: Optional[str] = None,
    sector_etfs: Mapping[str, str] = SECTOR_ETFS,
):
    """Rolling (annualized) volatility comparison: ETF vs proxy basket."""
    df = df.dropna().copy()
    etf = df["ETF"]
    bsk = df["Basket"]

    vol_etf = etf.rolling(win).std() * np.sqrt(252)
    vol_bsk = bsk.rolling(win).std() * np.sqrt(252)

    if etf_label is None:
        etf_label = sector_etfs.get(sector_name, "ETF")

    fig, ax = plt.subplots(figsize=(13, 5.2))

    ax.plot(
        vol_bsk.index,
        vol_bsk.values,
        linewidth=2.6,
        linestyle="--",
        label=f"{sector_name} proxy vol",
        color=C_60D,
    )
    ax.plot(
        vol_etf.index,
        vol_etf.values,
        linewidth=2.6,
        linestyle="--",
        label=f"{etf_label} vol",
        color=C_120D,
    )

    ax.set_title(f"{sector_name}: Rolling Volatility Comparison ({win}d, annualized)", pad=14)
    ax.set_ylabel("Annualized volatility")
    ax.set_xlabel("")

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    apply_dark_style(fig, ax, show_zero_line=False)
    plt.tight_layout()
    return fig, ax


def plot_performance_pretty(
    sector_name: str,
    df: pd.DataFrame,
    etf_label: Optional[str] = None,
    title_suffix: str = "(Index=100)",
    sector_etfs: Mapping[str, str] = SECTOR_ETFS,
):
    """Cumulative performance (Index=100): ETF vs proxy basket."""
    df = df.dropna().copy()
    idx = 100 * np.exp(df[["ETF", "Basket"]].cumsum())

    if etf_label is None:
        etf_label = sector_etfs.get(sector_name, "ETF")

    fig, ax = plt.subplots(figsize=(13, 5.2))

    ax.plot(idx.index, idx["Basket"], linewidth=2.6, label=f"{sector_name} proxy", color=C_60D)
    ax.plot(idx.index, idx["ETF"], linewidth=2.6, label=f"{etf_label} (ETF)", color=C_120D)

    ax.set_title(f"{sector_name}: ETF vs DIY Sector Basket {title_suffix}", pad=14)
    ax.set_ylabel("Index (base = 100)")
    ax.set_xlabel("")

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    apply_dark_style(fig, ax, show_zero_line=False)
    plt.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# File/plot utilities
# -----------------------------------------------------------------------------
def safe_name(s: str) -> str:
    """Filesystem-safe name used for plot saving."""
    return s.replace("&", "and").replace("/", "-").replace(" ", "_").lower()


def save_pretty_plots(
    series: Mapping[str, pd.DataFrame],
    outroot: str | os.PathLike = "chapter3_sector_png_split_pretty",
    win: int = 60,
    sector_etfs: Mapping[str, str] = SECTOR_ETFS,
):
    """Save volatility + performance plots for each sector in `series`."""
    outroot = Path(outroot)
    vol_dir = outroot / "volatility"
    perf_dir = outroot / "performance"
    vol_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    for sector_name, df in series.items():
        if df.dropna().empty:
            continue

        fig, _ = plot_volatility_pretty(
            sector_name, df, win=win, etf_label=sector_etfs.get(sector_name), sector_etfs=sector_etfs
        )
        fig.savefig(vol_dir / f"{safe_name(sector_name)}_vol{win}.png", bbox_inches="tight")
        plt.close(fig)

        fig, _ = plot_performance_pretty(
            sector_name, df, etf_label=sector_etfs.get(sector_name), sector_etfs=sector_etfs
        )
        fig.savefig(perf_dir / f"{safe_name(sector_name)}.png", bbox_inches="tight")
        plt.close(fig)

    return str(vol_dir), str(perf_dir)
