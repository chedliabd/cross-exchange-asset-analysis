"""Corrected Strategy Builder (portfolio profiling) logic.

The original notebook version (results.ipynb, "Strategy Builder") had three bugs:

1. **Basket index from a volume-weighted *price*** sum(P*V)/sum(V). When a stock
   enters or leaves the basket the average price jumps (e.g. a $500 stock joining
   $20 stocks), and those jumps were counted as returns, which gave annualised
   volatilities of 100-500 %. Fix: compound *weighted returns*, with weights fixed
   at the previous close (no look-ahead), as in Chapters 4-5.
2. **Regime rows were empty.** 396 of the 528 rows (bull/neutral/bear) had
   n_days = 0 and NaN metrics. Fix: compute regime statistics from the daily returns
   on regime days. CAGR from first/last index value is meaningless on
   non-contiguous days, so we annualise the mean daily log return.
3. **NaN rows were labelled "gambler".** `NaN < x` is False, so a missing value fell
   into the (high, high) bucket. Fix: NaN -> "insufficient data", and tertiles are
   computed *within each regime* so that a bear-market configuration is compared with
   other bear-market configurations rather than with bull markets.

ETF rows are also no longer duplicated across market categories: Q/G/S do not
apply to sector ETFs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

PROFILE_GRID = {
    ("high", "low"): "genius",
    ("high", "medium"): "champion",
    ("high", "high"): "gambler",
    ("medium", "low"): "steady",
    ("medium", "medium"): "balanced",
    ("medium", "high"): "adventurous",
    ("low", "low"): "dormant",
    ("low", "medium"): "fragile",
    ("low", "high"): "disaster",
}


def dollar_volume_weights(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """Daily weights proportional to dollar volume (P*V), the same weighting idea as the original builder."""
    dv = (prices * volumes).where(volumes > 0)
    return dv.div(dv.sum(axis=1), axis=0)


def basket_returns(returns: pd.DataFrame, weights: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Daily basket return = sum_i w_{i,t-lag} * r_{i,t}.

    Weights are renormalised each day over the stocks that actually have a return
    that day, so listings and delistings change the *weights*, never the index *level*.
    """
    w = weights.shift(lag).where(returns.notna())
    w = w.div(w.sum(axis=1), axis=0)
    return (w * returns).sum(axis=1, min_count=1).dropna()


def basket_index(returns: pd.DataFrame, weights: pd.DataFrame, lag: int = 1, base: float = 100.0) -> pd.Series:
    return base * (1 + basket_returns(returns, weights, lag)).cumprod()


def regime_stats(
    daily_returns: pd.Series,
    regime: pd.Series,
    rf_daily: pd.Series | None = None,
    min_days: int = 60,
) -> pd.DataFrame:
    """Annualised return, volatility and Sharpe for all days and for each regime."""
    r = daily_returns.dropna()
    reg = regime.reindex(r.index)
    rf = rf_daily.reindex(r.index).fillna(0.0) if rf_daily is not None else pd.Series(0.0, index=r.index)
    rows = []
    for name, mask in [("all", pd.Series(True, index=r.index))] + [
        (g, reg == g) for g in ("bull", "neutral", "bear")
    ]:
        x, ex = r[mask], (r - rf)[mask]
        n = int(mask.sum())
        if n < min_days:
            rows.append({"regime": name, "ann_return": np.nan, "ann_volatility": np.nan, "sharpe": np.nan, "n_days": n})
            continue
        rows.append({
            "regime": name,
            "ann_return": float(np.expm1(np.log1p(x).mean() * TRADING_DAYS)),
            "ann_volatility": float(x.std() * np.sqrt(TRADING_DAYS)),
            "sharpe": float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)),
            "n_days": n,
        })
    return pd.DataFrame(rows)


def _bucket(x: float, lo: float, hi: float) -> str:
    return "low" if x < lo else ("medium" if x < hi else "high")


def classify_profiles(
    df: pd.DataFrame,
    group_col: str = "regime",
    ret_col: str = "ann_return",
    vol_col: str = "ann_volatility",
) -> pd.Series:
    """3x3 tertile profile, with thresholds computed within each regime; NaN -> 'insufficient data'."""
    labels = pd.Series("insufficient data", index=df.index, dtype=object)
    for _, g in df.groupby(group_col):
        valid = g.dropna(subset=[ret_col, vol_col])
        if len(valid) < 3:
            continue
        r_lo, r_hi = valid[ret_col].quantile([1 / 3, 2 / 3])
        v_lo, v_hi = valid[vol_col].quantile([1 / 3, 2 / 3])
        for i, row in valid.iterrows():
            labels[i] = PROFILE_GRID[(_bucket(row[ret_col], r_lo, r_hi), _bucket(row[vol_col], v_lo, v_hi))]
    return labels
