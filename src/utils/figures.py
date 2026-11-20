"""
Figures: one folder and one look for the whole project.

- Every figure is saved under `website/figures/<section>/<name>.png`. The website
  reads its images from there directly, so a figure exists in one place only.
- Two themes: "light" for notebooks, "dark" (transparent, light text) for the
  website's dark cards.
"""

from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path("website/figures")

THEMES = {
    "light": {
        "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1", "shade": "#e6e5e1",
        "surface": "white", "transparent": False,
        "accent": "#6d4ae6", "accent2": "#0f9f8f",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "strategy": "#2a78d6", "spy": "#eb6834", "ew": "#1baf7a",
        "diverging": ["#2166ac", "#f7f7f7", "#b2182b"],
    },
    "dark": {  # matches the website (dark navy cards, light text)
        "text": "#E8EEFF", "muted": "#A9B3D8", "grid": "#2a3350", "shade": "#1f2942",
        "surface": "#0E1630", "transparent": True,
        "accent": "#7C5CFF", "accent2": "#2DD4BF",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "strategy": "#3987e5", "spy": "#d95926", "ew": "#199e70",
        "diverging": ["#2DD4BF", "#16244a", "#7C5CFF"],
    },
}


@contextmanager
def themed(theme: str = "light"):
    """Matplotlib settings for a theme; yields the theme's colors."""
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


def save_figure(fig: plt.Figure, section: str, name: str, dpi: int = 150, transparent: bool = False) -> Path:
    """Save `fig` to website/figures/<section>/<name>.png and return the path."""
    out_dir = FIGURES_DIR / section
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    return path


def site_figure(plot_fn, section: str, name: str, **kwargs) -> None:
    """
    Draw a figure twice with `plot_fn(..., theme=...)`: the dark version is saved
    for the website, the light version is shown in the notebook. Both are closed
    afterwards, so a loop over many figures does not keep them all in memory.
    """
    from IPython.display import display

    dark = plot_fn(**kwargs, theme="dark")
    path = save_figure(dark, section, name, transparent=True)
    plt.close(dark)
    print("Saved", path)
    light = plot_fn(**kwargs, theme="light")
    display(light)
    plt.close(light)


# ---------- Interactive (Plotly) figures ----------

PLOTLY_THEMES = {
    "light": {"bg": "white", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1"},
    "dark": {"bg": "rgba(0,0,0,0)", "text": "#E8EEFF", "muted": "#A9B3D8", "grid": "rgba(255,255,255,0.12)"},
}


def save_html_figure(fig, section: str, name: str) -> Path:
    """
    Save a Plotly figure to website/figures/<section>/<name>.html. plotly.js is
    loaded from its CDN instead of being embedded, which keeps the file small.
    """
    out_dir = FIGURES_DIR / section
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True, config={"displaylogo": False, "responsive": True})
    return path
