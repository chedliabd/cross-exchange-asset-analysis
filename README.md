# ada-2025-project-thedatacollectors

# Exploring Market Dynamics: Time-Based Comparisons and Strategy Design from OHLCV Data

## Abstract

In this project, we perform a large-scale comparative analysis of financial assets over time, focusing on both **stocks** and **ETFs**.  
Our objective is to identify **structural and behavioral differences** across multiple market dimensions — such as listing exchange, NASDAQ market category, and economic sector.

We analyze how these differences **evolve through time** using key indicators like **returns, volatility, and trading volume**, leveraging historical **OHLCV** (Open, High, Low, Close, Volume) data.  
The goal is to uncover consistent market behaviors and test whether they can be **exploited for predictive or investment purposes** through quantitative modeling and backtesting.

---

## Research Questions

1. Do industries that share similar volatility and volume dynamics correspond to the same NASDAQ market categories (Global Select, Global Market, Capital Market)?
2. Which indicator — volatility or trading volume — better differentiates industries in terms of market behavior and clustering?
3. Does volatility contain predictive information about short-term future returns within or across market categories?
4. How do mismatches between industry-based clusters and official market categories reflect market efficiency or investor perception?
5. Do data-driven volatility/volume clusters align with traditional sector classifications (Tech, Biotech, Utilities, etc.)?
6. Which indicators are most influential in defining behavioral clusters across industries?
7. How stable are industry-level volatility regimes over time? Are transitions linked to macroeconomic or stress events?
8. Does past volatility or trading activity predict short-term (1–5 day) return direction or magnitude?
9. Do financially weaker firms exhibit stronger volatility or liquidity anomalies compared to stable ones?
10. Do stocks in the same sectors move similarly, or are their price dynamics independent from sector membership?

---

## Additional Datasets

- **NASDAQ Metadata:** Market category enrichment for NASDAQ-traded symbols (Global Select, Global Market, Capital Market).  
- **Sector Classification Data:** Sector and industry tags for each ticker retrieved from Yahoo Finance.

---

## Methods

### 1. Data Cleaning

**Goal:** Ensure continuous, consistent, and interpretable OHLCV series with a strong emphasis on the reliability of `Adj Close`.

- **Adjusted Close (`Adj Close`):**
  - Drop tickers with all-zero `Adj Close` (corrupted or incomplete data).
  - Fill isolated NaN or short gaps via **bidirectional interpolation** (mean between previous and next valid values).
  - Apply **forward fill** for single-day NaNs to maintain continuity.
  - This affects less than **0.002%** of rows and preserves temporal structure.

- **Open Prices:**
  - Impute missing or zero values using the **previous day’s `Adj Close`** to ensure day-to-day consistency.

- **Volume:**
  - Zero volumes (non-trading days, suspensions) are **retained and flagged** for potential weighting or later handling.

- **Constant (Stale) Ranges:**
  - Periods with constant prices/volumes are **flagged**, not removed.
  - They may reflect market suspensions, corporate actions, or placeholder imputations.
  - These can later be **down-weighted or smoothed** in modeling.

---

### 2. Data Preprocessing

- Merge **industry and sector metadata** with OHLCV data.
- Merge **NASDAQ market category** information.
- Compute **daily log returns** from `Adj Close`.
- Compute **rolling volatility** (30-day window) to capture short-term risk and stability patterns.

---

### 3. Exploratory Analysis

- **Volatility and Volume Patterns:**  
  Scatterplots, rolling correlations, and cross-sector mean/standard deviation tracking.  
  Highlight outlier sectors with error bars (mean ± std).

- **Category Comparisons:**  
  Compare distributions (histograms, KDEs, boxplots) of returns and volatilities across NASDAQ categories and sectors.

- **Collinearity Checks:**  
  Compute pairwise correlations between main features.

- **Market Breadth & Regimes:**  
  Visualize cross-sectional volatility and the percentage of advancing/declining assets over time (heatmaps).

---

### 4. Clustering Analysis

We cluster stocks using their **correlation structure of log returns** to study sectoral co-movement.

- Build a **correlation matrix** and convert it into a **k-nearest-neighbor graph**.  
- Apply **Louvain** and **Leiden** community detection algorithms to find clusters that maximize modularity.  
- Compare cluster outputs with known sector and market classifications to assess alignment between data-driven and fundamental groupings.

---

### 5. Hypothesis Testing & Regression

Hypotheses to evaluate:

- **H1:** Higher relative volume → higher realized volatility.  
- **H2:** Volume shocks predict next-day volatility.  
- **H3:** After extreme volume/range spikes → mean reversion in returns.  
- **H4:** Momentum vs. mean reversion patterns differ by NASDAQ category (Q/G/S).  
- **H5:** IPO cohorts show distinct volatility/liquidity and weaker predictability.

Analytical approach:
- Run regressions, decile sorts, and out-of-sample validation.
- Visualize results via **forest plots**, **bar charts**, and **violin plots**.

---

### 6. Trading Strategy & Backtesting

Develop and evaluate trading strategies based on volatility and volume signals:

- **Signal Types:**
  - Relative Volume–Shock Reversal  
  - Range–Breakout  
  - Cross-Sectional Momentum (12–1)  
  - ETF vs. Basket Hedge (optional)

- **Backtesting Metrics:**
  - Daily and monthly backtests  
  - Performance measured via **PnL**, **Sharpe ratio**, and **maximum drawdown**  
  - Compare signal performance across market categories and asset types (stocks vs ETFs)

---

## Draft Timeline

| Date | Milestone |
|------|------------|
| 17/11 | Exploratory Analysis |
| 24/11 | Clustering |
| 01/12 | Hypothesis Testing & Regression |
| 08/12 | Trading & Backtesting |
| 17/12 | Final Review & Submission |

---

## Team Organization

- **Efe:** Correlation & Clustering  
- **Hugo:** Data Cleaning, Preprocessing, and Exploratory Analysis  
- **Irem:** Hypothesis Testing, Regression, and Backtesting  
- **Ozan:** Visualization, Reporting, and Results Integration, Data Augmentation with sectors
- **Chedly:** Visualization, Reporting, Data Cleaning, Results Integration, Data Augmentation with market categories
