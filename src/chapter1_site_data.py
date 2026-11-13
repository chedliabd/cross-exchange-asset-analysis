"""Build the data behind the Chapter 1 website charts (website/content/chapter1.json).

The raw stock data and the `chapter1/` helper package are not in this repository, so the
sector correlation matrices are taken from the saved outputs of `results.ipynb`
(cells "Sector-Sector Correlation" and "Volatility Correlation", values printed to 2
decimals). The transcription is checked two ways before anything is exported:

1. every matrix must be symmetric with a unit diagonal;
2. re-applying the notebook's diversifier rule (average correlation with the other 10
   sectors below 0.25 for returns / 0.15 for volatility) must reproduce the notebook's
   saved diversifier tables exactly.

We also export a sector order from average-linkage hierarchical clustering on the
correlation distance d = sqrt(2 (1 - rho)) of the full-sample matrix, so the heatmap
shows related sectors as blocks. (A 2-D PCA/MDS "sector map" of these matrices keeps only
~25-35 % of the structure, so the website uses the exact clustered heatmap instead.)

Run from the repo root:  python -m src.chapter1_site_data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

SECTORS = [
    "Basic Materials", "Communication Services", "Consumer Cyclical", "Consumer Defensive",
    "Energy", "Financial Services", "Healthcare", "Industrials", "Real Estate",
    "Technology", "Utilities",
]
WINDOWS = {
    "full": "Full sample (1973–2020)",
    "20y": "Last 20 years (2000–2020)",
    "5y": "Last 5 years (2015–2020)",
}
THRESHOLDS = {"returns": 0.25, "volatility": 0.15}

# Rows/columns in SECTORS order, transcribed from results.ipynb outputs.
CORR = {
    "returns": {
        "full": """
1.00 .36 .20 .31 .52 .43 .25 .32 .32 .38 .33
.36 1.00 .17 .27 .30 .33 .24 .26 .25 .37 .24
.20 .17 1.00 .16 .17 .21 .14 .16 .15 .21 .12
.31 .27 .16 1.00 .26 .30 .23 .23 .22 .29 .25
.52 .30 .17 .26 1.00 .37 .24 .29 .25 .32 .32
.43 .33 .21 .30 .37 1.00 .27 .30 .32 .35 .29
.25 .24 .14 .23 .24 .27 1.00 .22 .19 .29 .17
.32 .26 .16 .23 .29 .30 .22 1.00 .21 .29 .21
.32 .25 .15 .22 .25 .32 .19 .21 1.00 .28 .22
.38 .37 .21 .29 .32 .35 .29 .29 .28 1.00 .23
.33 .24 .12 .25 .32 .29 .17 .21 .22 .23 1.00
""",
        "20y": """
1.00 .32 .15 .33 .57 .47 .21 .25 .29 .40 .35
.32 1.00 .10 .27 .29 .34 .19 .19 .22 .38 .22
.15 .10 1.00 .12 .14 .18 .09 .10 .11 .17 .09
.33 .27 .12 1.00 .29 .34 .20 .18 .22 .31 .27
.57 .29 .14 .29 1.00 .42 .22 .24 .23 .34 .34
.47 .34 .18 .34 .42 1.00 .24 .26 .34 .41 .31
.21 .19 .09 .20 .22 .24 1.00 .15 .14 .25 .14
.25 .19 .10 .18 .24 .26 .15 1.00 .15 .24 .16
.29 .22 .11 .22 .23 .34 .14 .15 1.00 .27 .21
.40 .38 .17 .31 .34 .41 .25 .24 .27 1.00 .25
.35 .22 .09 .27 .34 .31 .14 .16 .21 .25 1.00
""",
        "5y": """
1.00 .21 .36 .27 .49 .49 .10 .14 .18 .35 .23
.21 1.00 .33 .21 .26 .40 .10 .10 .16 .29 .14
.36 .33 1.00 .31 .37 .58 .17 .19 .25 .42 .20
.27 .21 .31 1.00 .27 .42 .11 .09 .21 .28 .23
.49 .26 .37 .27 1.00 .56 .16 .18 .19 .36 .22
.49 .40 .58 .42 .56 1.00 .20 .22 .33 .52 .31
.10 .10 .17 .11 .16 .20 1.00 .07 .07 .17 .02
.14 .10 .19 .09 .18 .22 .07 1.00 .07 .13 .07
.18 .16 .25 .21 .19 .33 .07 .07 1.00 .24 .17
.35 .29 .42 .28 .36 .52 .17 .13 .24 1.00 .19
.23 .14 .20 .23 .22 .31 .02 .07 .17 .19 1.00
""",
    },
    "volatility": {  # correlation of 60-day rolling volatility
        "full": """
1.00 .49 .10 .26 .79 .57 .30 .18 .35 .20 .38
.49 1.00 .08 .26 .42 .52 .20 .11 .21 .36 .29
.10 .08 1.00 -.00 .11 .34 .06 -.03 -.03 -.05 .03
.26 .26 -.00 1.00 .26 .18 .17 .11 .11 .32 .22
.79 .42 .11 .26 1.00 .51 .34 .19 .21 .25 .39
.57 .52 .34 .18 .51 1.00 .17 .08 .22 .21 .26
.30 .20 .06 .17 .34 .17 1.00 .24 .04 .20 .33
.18 .11 -.03 .11 .19 .08 .24 1.00 -.00 .09 .18
.35 .21 -.03 .11 .21 .22 .04 -.00 1.00 .10 .15
.20 .36 -.05 .32 .25 .21 .20 .09 .10 1.00 .31
.38 .29 .03 .22 .39 .26 .33 .18 .15 .31 1.00
""",
        "20y": """
1.00 .39 -.06 .23 .75 .60 .16 .07 .29 .23 .25
.39 1.00 -.03 .20 .33 .49 .05 .02 .16 .39 .20
-.06 -.03 1.00 -.08 -.04 .30 -.08 -.10 -.08 -.12 -.09
.23 .20 -.08 1.00 .19 .14 .21 .03 .09 .37 .28
.75 .33 -.04 .19 1.00 .49 .20 .09 .12 .22 .27
.60 .49 .30 .14 .49 1.00 .04 .00 .18 .20 .19
.16 .05 -.08 .21 .20 .04 1.00 .23 -.01 .18 .30
.07 .02 -.10 .03 .09 .00 .23 1.00 -.05 .05 .14
.29 .16 -.08 .09 .12 .18 -.01 -.05 1.00 .13 .10
.23 .39 -.12 .37 .22 .20 .18 .05 .13 1.00 .40
.25 .20 -.09 .28 .27 .19 .30 .14 .10 .40 1.00
""",
        "5y": """
1.00 .07 -.00 -.09 .68 .48 -.08 -.18 .19 .24 -.12
.07 1.00 .13 .12 .13 .24 -.11 -.09 -.03 .37 .02
-.00 .13 1.00 .04 .20 .46 .19 -.13 -.06 .11 .14
-.09 .12 .04 1.00 -.00 .14 .26 -.08 -.08 .13 .13
.68 .13 .20 -.00 1.00 .59 .07 -.10 .05 .20 .05
.48 .24 .46 .14 .59 1.00 .15 -.11 .04 .54 .15
-.08 -.11 .19 .26 .07 .15 1.00 .02 -.16 .08 .19
-.18 -.09 -.13 -.08 -.10 -.11 .02 1.00 -.10 -.14 -.07
.19 -.03 -.06 -.08 .05 .04 -.16 -.10 1.00 -.06 -.09
.24 .37 .11 .13 .20 .54 .08 -.14 -.06 1.00 .07
-.12 .02 .14 .13 .05 .15 .19 -.07 -.09 .07 1.00
""",
    },
}

# Diversifier tables saved in results.ipynb (1 = qualifies in that window)
NOTEBOOK_DIVERSIFIERS = {
    "returns": {
        "Healthcare": (1, 1, 1), "Industrials": (1, 1, 1), "Real Estate": (1, 1, 1),
        "Utilities": (1, 1, 1), "Consumer Cyclical": (1, 1, 0), "Communication Services": (0, 0, 1),
        "Consumer Defensive": (0, 0, 1), "Basic Materials": (0, 0, 0), "Energy": (0, 0, 0),
        "Financial Services": (0, 0, 0), "Technology": (0, 0, 0),
    },
    "volatility": {
        "Consumer Cyclical": (1, 1, 1), "Industrials": (1, 1, 1), "Real Estate": (1, 1, 1),
        "Healthcare": (0, 1, 1), "Basic Materials": (0, 0, 1), "Communication Services": (0, 0, 1),
        "Consumer Defensive": (0, 0, 1), "Utilities": (0, 0, 1), "Energy": (0, 0, 0),
        "Financial Services": (0, 0, 0), "Technology": (0, 0, 0),
    },
}


def parse(block: str) -> np.ndarray:
    m = np.array([[float(x) for x in row.split()] for row in block.strip().splitlines()])
    assert m.shape == (11, 11), m.shape
    return m


def check(m: np.ndarray, name: str) -> None:
    assert np.allclose(np.diag(m), 1.0), f"{name}: diagonal"
    bad = np.argwhere(~np.isclose(m, m.T))
    assert len(bad) == 0, f"{name}: asymmetric at {[(SECTORS[i], SECTORS[j]) for i, j in bad]}"


def avg_corr(m: np.ndarray) -> np.ndarray:
    """Average correlation of each sector with the 10 others."""
    return (m.sum(axis=1) - 1.0) / (m.shape[0] - 1)


def cluster_order(m: np.ndarray) -> list[int]:
    d = np.sqrt(np.clip(2.0 * (1.0 - m), 0, None))
    np.fill_diagonal(d, 0.0)
    return leaves_list(linkage(squareform(d, checks=False), "average", optimal_ordering=True)).tolist()


def build() -> dict:
    out = {"sectors": SECTORS, "windows": WINDOWS, "thresholds": THRESHOLDS, "metrics": {}}
    for metric, blocks in CORR.items():
        mats = {w: parse(b) for w, b in blocks.items()}
        for w, m in mats.items():
            check(m, f"{metric}/{w}")

        avgs = {w: avg_corr(m) for w, m in mats.items()}
        passes = {w: avgs[w] < THRESHOLDS[metric] for w in mats}
        for i, s in enumerate(SECTORS):
            got = tuple(int(passes[w][i]) for w in ("full", "20y", "5y"))
            assert got == NOTEBOOK_DIVERSIFIERS[metric][s], f"{metric}/{s}: {got} != notebook"

        out["metrics"][metric] = {
            "corr": {w: m.round(2).tolist() for w, m in mats.items()},
            "avg_corr": {w: a.round(3).tolist() for w, a in avgs.items()},
            "diversifier": {w: p.astype(int).tolist() for w, p in passes.items()},
            "order": cluster_order(mats["full"]),
            "mean_pair_corr": {w: round(float(m[np.triu_indices(11, 1)].mean()), 3) for w, m in mats.items()},
        }
    return out


if __name__ == "__main__":
    data = build()
    path = Path("website/content/chapter1.json")
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"checks passed, wrote {path} ({path.stat().st_size / 1024:.1f} KB)")
    for metric, d in data["metrics"].items():
        print(metric, "mean pair correlation:", d["mean_pair_corr"])
