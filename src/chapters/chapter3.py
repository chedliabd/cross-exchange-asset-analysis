"""
Chapter 3 (market categories inside sectors) helpers for `chapter3_market_categories.ipynb`.

Market categories are NASDAQ's listing tiers, so only NASDAQ-listed stocks have one:
Q = Global Select Market (largest, strictest), G = Global Market, S = Capital Market (smallest).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.utils.figures import PLOTLY_THEMES

TIERS = {"Q": "Global Select", "G": "Global Market", "S": "Capital Market"}
TIER_SYMBOLS = {"Q": "circle", "G": "square", "S": "triangle-up"}


def attach_tiers(stocks_sect: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Keep the NASDAQ-listed stocks that have a market category and add it as `Tier`."""
    tiers = meta.loc[
        (meta["listing_exchange"] == "Q") & meta["market_category"].isin(list(TIERS)),
        ["symbol", "market_category"],
    ].rename(columns={"symbol": "Symbol", "market_category": "Tier"})
    tiers["Tier"] = tiers["Tier"].astype(str)
    return stocks_sect.merge(tiers.drop_duplicates("Symbol"), on="Symbol", how="inner")


def cell_series(df: pd.DataFrame, value_col: str, min_stocks: int) -> pd.DataFrame:
    """
    Dollar-volume weighted average of `value_col` per (Date, Sector, Tier), using
    the stocks that have a value that day. Cells with fewer than `min_stocks`
    stocks that day are NaN.

    Returns a long DataFrame: Date, Sector, Tier, value, n_stocks.
    """
    d = df.loc[df[value_col].notna(), ["Date", "Sector", "Tier", "DollarVolume", value_col]]
    d = d.assign(_wx=d["DollarVolume"] * d[value_col])
    g = d.groupby(["Date", "Sector", "Tier"])
    out = pd.DataFrame({
        "value": g["_wx"].sum() / g["DollarVolume"].sum(),
        "n_stocks": g.size(),
    }).reset_index()
    out.loc[out["n_stocks"] < min_stocks, "value"] = float("nan")
    return out


def smooth(cells: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean of `value` over `window` days, separately for each (Sector, Tier)."""
    wide = cells.pivot_table(index="Date", columns=["Sector", "Tier"], values="value")
    return wide.rolling(window, min_periods=window).mean()


def weekly_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available date of each week."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("W")).max().values)


def category_map_figure(
    vol: pd.DataFrame, ret: pd.DataFrame, n_stocks: pd.DataFrame, frame_dates, theme: str = "light"
) -> go.Figure:
    """
    Animated Sector × Tier map: x = smoothed volatility, y = smoothed return,
    colour = sector, marker = tier, one frame per date in `frame_dates`.

    `vol`, `ret` and `n_stocks` are wide tables: index = Date, columns = (Sector, Tier).
    """
    if len(frame_dates) == 0:
        raise ValueError("No date has a sector × tier point: check MIN_STOCKS_PER_CELL and the date range")
    t = PLOTLY_THEMES[theme]
    cells = [c for c in vol.columns if vol.loc[frame_dates, c].notna().any() and ret.loc[frame_dates, c].notna().any()]
    sectors = sorted({s for s, _ in cells})
    palette = px.colors.qualitative.Vivid if theme == "dark" else px.colors.qualitative.Bold
    colors = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}

    def point(c, d):
        x, y = vol.at[d, c], ret.at[d, c]
        ok = pd.notna(x) and pd.notna(y)
        n = n_stocks.at[d, c] if c in n_stocks.columns and d in n_stocks.index else None
        return ([x] if ok else [None]), ([y] if ok else [None]), [[int(n) if pd.notna(n) else 0, d.strftime("%Y-%m-%d")]]

    first = frame_dates[0]
    fig = go.Figure()
    for sector, tier in cells:
        x, y, cd = point((sector, tier), first)
        fig.add_trace(go.Scatter(
            x=x, y=y, customdata=cd, mode="markers", name=f"{sector} · {TIERS[tier]}",
            legendgroup=sector,
            marker=dict(color=colors[sector], symbol=TIER_SYMBOLS[tier], size=12, opacity=0.9,
                        line=dict(width=0.5, color=t["bg"] if theme == "light" else "#0E1630")),
            hovertemplate=(f"<b>{sector}</b><br>{TIERS[tier]} ({tier})<br>Date: %{{customdata[1]}}"
                           "<br>Volatility: %{x:.4f}<br>Return: %{y:.4f}<br>Stocks: %{customdata[0]}<extra></extra>"),
        ))

    frames = []
    for d in frame_dates:
        data = []
        for c in cells:
            x, y, cd = point(c, d)
            data.append(go.Scatter(x=x, y=y, customdata=cd))
        frames.append(go.Frame(name=d.strftime("%Y-%m-%d"), data=data, traces=list(range(len(cells)))))
    fig.frames = frames

    xs = pd.Series(vol.loc[frame_dates, cells].values.ravel()).dropna()
    ys = pd.Series(ret.loc[frame_dates, cells].values.ravel()).dropna()
    pad = lambda s: (s.max() - s.min()) * 0.05
    anim = dict(frame=dict(duration=60, redraw=False), mode="immediate", fromcurrent=True, transition=dict(duration=0))
    fig.update_layout(
        title="Sector × NASDAQ tier: risk and return over time",
        xaxis=dict(title="Volatility (20-day, 60-day smoothed)", range=[xs.min() - pad(xs), xs.max() + pad(xs)],
                   gridcolor=t["grid"], zeroline=False),
        yaxis=dict(title="Daily return (60-day smoothed)", range=[ys.min() - pad(ys), ys.max() + pad(ys)],
                   gridcolor=t["grid"], zeroline=True, zerolinecolor=t["muted"]),
        legend=dict(title="Sector · Tier (● Global Select, ■ Global Market, ▲ Capital Market)", font=dict(size=10)),
        paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font=dict(color=t["text"]),
        margin=dict(l=60, r=20, t=60, b=40),
        updatemenus=[dict(type="buttons", direction="left", x=0.1, xanchor="right", y=0, yanchor="top",
                          pad=dict(r=10, t=70), showactive=False,
                          buttons=[dict(label="▶", method="animate", args=[None, anim]),
                                   dict(label="◼", method="animate", args=[[None], {**anim, "frame": dict(duration=0, redraw=False)}])])],
        sliders=[dict(x=0.1, len=0.9, y=0, yanchor="top", pad=dict(t=50), currentvalue=dict(prefix="Week of "),
                      steps=[dict(label=f.name, method="animate", args=[[f.name], {**anim, "frame": dict(duration=0, redraw=False)}])
                             for f in frames])],
    )
    return fig
