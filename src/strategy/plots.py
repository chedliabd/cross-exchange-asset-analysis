"""Chapter 6 figures, in a light theme (notebook / README) and a dark transparent theme (website)."""
from __future__ import annotations

from contextlib import contextmanager

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sector_rotation import drawdown

# Validated categorical palettes (dataviz reference slots 1-3), one step set per surface
THEMES = {
    "light": {
        "strategy": "#2a78d6", "spy": "#eb6834", "ew": "#1baf7a",
        "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1", "shade": "#e6e5e1",
        "surface": "white", "transparent": False,
    },
    "dark": {  # matches the website (dark navy cards, light text)
        "strategy": "#3987e5", "spy": "#d95926", "ew": "#199e70",
        "text": "#E8EEFF", "muted": "#A9B3D8", "grid": "#2a3350", "shade": "#1f2942",
        "surface": "#0E1630", "transparent": True,
    },
}


@contextmanager
def themed(theme: str):
    t = THEMES[theme]
    rc = {
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": t["muted"], "axes.labelcolor": t["muted"], "axes.titlecolor": t["text"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "text.color": t["text"],
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "legend.frameon": False, "legend.labelcolor": t["text"],
        "axes.facecolor": "none" if t["transparent"] else t["surface"],
        "figure.facecolor": "none" if t["transparent"] else t["surface"],
        "savefig.transparent": t["transparent"],
    }
    with plt.rc_context(rc):
        yield t


def _end_label(ax, series, text, color, t):
    ax.annotate(text, (series.index[-1], series.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=t["text"])
    ax.plot(series.index[-1], series.iloc[-1], "o", ms=5, color=color, mec=t["surface"], mew=1.5)


def _shade_bear(ax, regime, t, label="bear regime"):
    bear = (regime == "bear").astype(int)
    starts = bear.index[(bear.diff() == 1) | ((bear == 1) & (bear.index == bear.index[0]))]
    ends = bear.index[bear.diff() == -1]
    for i, s in enumerate(starts):
        e = ends[ends > s][0] if (ends > s).any() else bear.index[-1]
        ax.axvspan(s, e, color=t["shade"], lw=0, label=label if i == 0 else None, zorder=0)


def plot_regime_definitions(spy, regimes: dict[str, pd.Series], start, end, theme="light"):
    with themed(theme) as t:
        fig, axes = plt.subplots(len(regimes), 1, figsize=(11, 2.75 * len(regimes)), sharex=True)
        for ax, (title, reg) in zip(np.atleast_1d(axes), regimes.items()):
            p = spy.loc[start:end]
            r = reg.loc[start:end]
            _shade_bear(ax, r, t)
            ax.plot(p.index, p, color=t["spy"], label="SPY (total return)")
            ax.set_title(f"{title}  ·  bear share {(r == 'bear').mean():.0%}", loc="left", fontsize=10)
            ax.set_yscale("log")
            ax.legend(loc="upper left")
        fig.suptitle("Design period: when does each regime definition say 'bear'?", x=0.01, ha="left",
                     fontweight="bold", color=t["text"])
        fig.tight_layout()
    return fig


def plot_equity_drawdown(series: dict[str, pd.Series], regime, boundaries: dict[str, str], theme="light"):
    """`series` maps label -> (daily returns, theme color key)."""
    with themed(theme) as t:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        _shade_bear(ax1, regime, t, "trend regime = bear")
        for label, (r, ckey) in series.items():
            eq = (1 + r).cumprod()
            ax1.plot(eq.index, eq, color=t[ckey], label=label)
            _end_label(ax1, eq, f"{label} ×{eq.iloc[-1]:.1f}", t[ckey], t)
            ax2.plot(eq.index, drawdown(eq), color=t[ckey], lw=1.4)
        for name, date in boundaries.items():
            for ax in (ax1, ax2):
                ax.axvline(pd.Timestamp(date), color=t["muted"], lw=0.8, ls="--")
            ax1.text(pd.Timestamp(date), ax1.get_ylim()[1] * 0.9, f" {name} →", fontsize=9, color=t["muted"])
        ax1.set_yscale("log")
        ax1.set_ylabel("growth of $1 (log)")
        ax1.legend(loc="upper left")
        ax2.set_ylabel("drawdown")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax1.set_title("Final strategy vs benchmarks, 1999-2026 (net of costs)", loc="left", fontweight="bold")
        fig.tight_layout()
    return fig


def plot_weights(monthly_weights: pd.DataFrame, theme="light"):
    """Rows = sectors, columns = month ends."""
    with themed(theme) as t:
        w = monthly_weights
        fig, ax = plt.subplots(figsize=(11, 4.2))
        cmap = plt.get_cmap("Blues" if theme == "light" else "Blues_r").copy()
        if theme == "dark":  # dark surface: near-zero recedes into the surface, high weight is bright
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("dark_blues", ["#16244a", "#2a5fae", "#86b6ef"])
        cmap.set_bad(t["surface"] if theme == "light" else (0, 0, 0, 0))
        im = ax.imshow(np.ma.masked_where(w.values < 1e-4, w.values), aspect="auto", cmap=cmap, vmin=0, vmax=0.5,
                       interpolation="nearest",
                       extent=[mdates.date2num(w.columns[0]), mdates.date2num(w.columns[-1]), len(w) - 0.5, -0.5])
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(w.index)
        ax.grid(False)
        ax.xaxis_date()
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label("weight")
        cb.outline.set_edgecolor(t["muted"])
        ax.set_title("Monthly sector weights (blank = not held / not listed yet)", loc="left", fontweight="bold")
        fig.tight_layout()
    return fig


def plot_sensitivity(heat: pd.DataFrame, spy_sharpe: float, theme="light"):
    with themed(theme) as t:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        cmap = "Blues"
        if theme == "dark":
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list("dark_blues", ["#16244a", "#2a5fae", "#86b6ef"])
        im = ax.imshow(heat.values, cmap=cmap, vmin=0.15, vmax=0.30)
        for (i, j), v in np.ndenumerate(heat.values):
            if theme == "light":
                color = "white" if v > 0.245 else t["text"]
            else:
                color = "#0b0b0b" if v > 0.255 else t["text"]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(heat.shape[1]))
        ax.set_xticklabels([f"{c:.0%}" for c in heat.columns])
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        ax.grid(False)
        ax.set_xlabel("volatility target")
        ax.set_ylabel("# defensive sectors")
        ax.set_title(f"Design-period Sharpe across settings (SPY: {spy_sharpe:.2f})", loc="left", fontweight="bold")
        fig.tight_layout()
    return fig
