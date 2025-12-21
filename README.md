
# An Investor’s Guide to Building a Smarter Portfolio

## Abstract

This project is built around one practical goal: help investors get the biggest market understanding and risk control with the least effort. Instead of giving abstract statistics, we build a step-by-step guide that answers real portfolio questions:

- Do sectors actually diversify each other, or is it fake diversification?
- How do market regimes affect the risk–return profiles of equities?
- Do market categories (NASDAQ Global Select, Global Market, Capital Market) exhibit distinct behavior across sectors?
- Within a sector, is one representative stock enough, or do you need a basket?
- Should you buy the sector ETF, or is it worth recreating the exposure yourself?



To do that, our analysis follows a logical pipeline: sector relationships, market regimes and categories, sector representation (single stocks vs baskets), ETF versus DIY exposure, and finally investor-specific portfolio guidance. The website will be a portfolio decision toolkit. The objective is to make sector investing more transparent, avoid misleading shortcuts, and provide investors with a simple way to choose between market exposure, sector exposure, ETFs, or DIY baskets, depending on both their risk tolerance and the market environment.




## Data Preparation & Cleaning Policy (For Both Stocks And ETFs)

This project relies on **time-consistent financial time series**.  
Because downstream analyses (rolling statistics, regimes, clustering) depend on
continuous timelines, the cleaning strategy prioritizes **symbol-level consistency**
over row-level fixes.

The same policy is applied to both **stocks** and **ETFs**.

### Core Principle
> If a symbol is structurally broken, it is removed.  
> If a value can be safely repaired without breaking time consistency, it is fixed.

As a result:
- Rows are not dropped blindly.
- Gaps inside time series are avoided.
- When issues are systematic, the **entire symbol** is removed.
- Visual diagnostics guide all major decisions.

### Main Cleaning Rules
- Dates are validated (ordering, coverage, trading days).
- Missing values are inspected globally, per symbol, and over time.
- If NaNs or zeros coincide across multiple columns within the same symbols,
  those symbols are removed.
- Zeros in price columns are treated as invalid.
- `Open` prices may be repaired using the previous day’s `Adj Close`.
- `Close` prices may be repaired using same-day `Adj Close`.
- Volume is handled separately (trim early zeros, repair conservatively, remove if unresolved).

Cleaning is **iterative**, with diagnostics repeated after each step.

---

## Metadata Cleaning & Enrichment

Metadata links price dynamics to structural attributes (sector, exchange,
market category, financial status). The goal is **correctness**, not completeness.

### Policy
- Metadata is never blindly trusted.
- Existing valid values are never overwritten.
- Missing values are filled only from reliable external sources.
- When information cannot be safely inferred, it is left missing.

### Cleaning & Enrichment
- Non-essential columns (e.g. `cqs_symbol`) are removed.
- Symbols and strings are normalized.
- Empty strings are converted to missing values.
- Market category and financial status are enriched using:

---


## Project Structure & Analysis Pipeline

### Chapter 1 — Sector-Sector Structure
- Sector–sector correlation 
- Volatility correlation
- Rolling correlations 
- Sector clustering 

### Chapter 2 — Market Regimes and Sector Behavior

**Goal:**  
Understand how **sector returns, volatility, and correlations change across market regimes**.

In this chapter, we define market regimes using **SPY drawdowns** (Bull, Neutral, Bear) and study how sector behavior depends on the prevailing market environment. Instead of looking at sectors in aggregate, we condition all analyses on the regime to capture how risk and return evolve over time.


### What is analyzed
- **Sector returns** computed as weighted averages of stock returns
- **Sector volatility** using rolling volatility measures
- **Correlation structure** between sectors
- **Stability of sector rankings** across regimes

### Key outputs
- Static sector correlation matrix
- Rolling average correlation over time
- Average sector returns by regime (Bear / Neutral / Bull)
- Clustered correlation heatmaps for each regime
- Sector volatility comparisons across regimes
- Statistical tests (Kruskal-Wallis, Spearman rank correlations)

### Why it matters
Sector behavior is not constant.  
The same sector can have very different risk–return profiles depending on whether markets are stressed, neutral, or expanding. This chapter shows that:
- diversification benefits vary across regimes,
- volatility concentrates differently in stress periods,
- sector attractiveness is regime-dependent.

**All final plots are generated from `results.ipynb`.**  
Development notebooks: `Chapter2_returns.ipynb`, `Chapter2_volat.ipynb`  
Helper modules: `chapter2/returns.py`, `chapter2/volatilities.py`

---

### Chapter 3 — Market Categories Within Sectors

**Goal:**  
Answer a practical question:  
**Within a given sector, does choosing a market category really change risk and return?**

Market categories are often treated as independent performance drivers. In this chapter, we test whether those distinctions remain meaningful **once sector exposure is fixed**, and whether they are stable through time.

### Market categories studied
- **Growth (G)** — a style dimension linked to high expected growth
- **Quality (Q)** — a style dimension linked to financial strength and stability
- **Size (S)** — a firm characteristic (not a style), reflecting firm scale and liquidity

### What is analyzed
- Average returns by category within each sector
- Average volatility by category within each sector
- Joint return–volatility behavior through time
- Interaction between sector structure and category choice

### Key outputs
- Static bar plots comparing categories inside sectors
- Feature tables combining return and volatility
- **Final animated return–volatility map** tracking  
  *Sector × Market Category* through time

The animation shows:
- when category distinctions collapse (sector dominates),
- when they separate (category choice matters),
- how these dynamics evolve across market conditions.

### Why it matters
Category effects are **not constant**:
- they can disappear during market stress,
- re-emerge in calmer periods,
- and behave very differently across sectors.

This chapter shows that category-based allocation only makes sense when considered **together with sector structure and market conditions**.

**All final outputs are generated from `results.ipynb`.**  
Development notebooks: `Chapter3_discovering.ipynb`, `Chapter3_final.ipynb`  
Helper modules: `chapter3/discovering.py`, `chapter3/final_clustering.py`

---
### Chapter 4 — ETF vs Sector Baskets 
- Volume-weighted sector basket returns
- ETF–basket correlation
- Rolling correlation
- Tracking error 
- ETF vs basket performance plots

### Chapter 5 — Sector Proxy: Is One Stock Enough?

This chapter examines whether a single stock can reliably represent a sector, or whether investors need a basket of stocks to avoid idiosyncratic risk.

For each sector, we construct a **sector proxy** as a volume-weighted basket of liquid stocks, used as a reference for true sector behavior. Individual stocks are evaluated based on **rolling correlations** between their returns and the sector proxy, after filtering out illiquid names.

Representative candidates are selected using a **lifetime-aware ranking rule**, which favors stocks that maintain high representativeness consistently over long periods, rather than those that rank highly only during short or favorable windows.

We then assess the quality of these candidates through **tracking error analysis**, which measures the volatility of the return difference between a stock (or basket) and the sector proxy over time. Lower tracking error indicates closer and more stable sector replication.

The results show that:
- Some sectors are cohesive enough to be well approximated by one carefully chosen stock.
- Other sectors display strong internal heterogeneity, making single-stock exposure unreliable.
- In most cases, a small basket captures most diversification benefits with limited added complexity.

This chapter provides practical guidance on when sector exposure can be simplified and when intra-sector diversification is essential.



### Strategy Builder
In the final part we create an interactive strategy builder where users can select sectors, market categories and type of portfolios to create custom portfolios and see their performance. They will see the average return their portfolio gives and how volatile it is. They will then get a description of how well the portfolio is diversified based on the sectors and market categories selected.

## How to execute the code

### 1. Clone the repository
```console
git clone https://github.com/epfl-ada/ada-2025-project-thedatacollectors.git
cd ada-2025-project-thedatacollectors
```
## 2. Create/activate a virtual environment and install the requirements
```console
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## 3. Dowload the dataset from https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset and place it in the data/raw folder 

## 4. Run preprocessing 

Run the preprocessing pipeline from the project root with:
```console
python src/preprocessing/preparing_dataset.py
```
## 5. Run the results.ipynb notebook to generate all the results and plots


## Contributions within the team

Chedly: worked on data inspection, did all of the data cleaning, worked on Chapter 2 (Market Regimes), Chapter 3 (Market Categories) and contributed to README, created the the website and added my part on it.

İrem: Worked on Chapter 1, focusing on sector–sector relationships. Prepared the README, added my part on the website

Efe: Worked on Chapter 4 (ETF analysis) and contributed to README.

Hugo: Worked on Chapter 5, analyzing sector proxy representativeness and contributed to data prepocessing and README.

Ozan: Worked on Chapter 6, developing the interactive investor profile mapping, and contributed to data preprocessing(added the secotrs) and README.


## Additional Datasets

- **NASDAQ Metadata:** Market category enrichment for NASDAQ-traded symbols (Global Select, Global Market, Capital Market).  
- **Sector Classification Data:** Sector and industry tags for each ticker retrieved from Yahoo Finance.
- **SPY data:**: [SPY Historical Prices on Kaggle](https://www.kaggle.com/datasets/benjaminbtang/spy-historical-prices)
