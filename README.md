
# An Investor's Guide to Building a Smarter Portfolio

**Live website:** https://epfl-ada.github.io/ada-2025-project-thedatacollectors/
(The `website/` folder is also deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`.)

---

## Table of contents

1. [Abstract](#abstract)
2. [Research questions](#research-questions)
3. [Data sources](#data-sources)
4. [End-to-end pipeline at a glance](#end-to-end-pipeline-at-a-glance)
5. [Data preparation & cleaning (stocks and ETFs)](#data-preparation--cleaning-stocks-and-etfs)
6. [Metadata cleaning & enrichment](#metadata-cleaning--enrichment)
7. [Feature engineering: returns, volatility, sector aggregation](#feature-engineering-returns-volatility-sector-aggregation)
8. [Chapter 1: Sector–sector structure (correlation, volatility, clustering)](#chapter-1--sectorsector-structure)
9. [Chapter 2: Market regimes and sector behaviour](#chapter-2--market-regimes-and-sector-behaviour)
10. [Chapter 3: Market categories within sectors](#chapter-3--market-categories-within-sectors)
11. [Chapter 4: Sector ETF vs DIY basket](#chapter-4--sector-etf-vs-diy-basket)
12. [Chapter 5: Sector proxy, or is one stock enough?](#chapter-5--sector-proxy-is-one-stock-enough)
13. [Chapter 6: Systematic sector strategy](#chapter-6-systematic-sector-strategy)
14. [Additional ML experiments](#additional-ml-experiments)
15. [Visualisation approach](#visualisation-approach)
16. [Design decisions, limitations & possible improvements](#design-decisions-limitations--possible-improvements)
17. [Repository map](#repository-map)
18. [How to run the code](#how-to-run-the-code)
19. [Tech stack](#tech-stack)

---

## Abstract

This project has one practical goal: to help investors gain the most market understanding and risk control for the least effort. We avoid abstract statistics and build a step-by-step guide that answers real portfolio questions. It works on ~20+ years of daily NASDAQ-traded stock and ETF prices and covers:

- **sector structure** (who moves with whom, and is diversification real or fake?),
- **market regimes** (how sector risk and return change in Bull / Neutral / Bear markets),
- **market categories** (does the NASDAQ listing tier matter once the sector is fixed?),
- **implementation** (sector ETF vs a do-it-yourself basket; one stock vs a small basket),
- a **systematic, backtested sector strategy** that turns these findings into trading rules (trend-switched defensive rotation with volatility targeting), validated out of sample up to 2026, plus a corrected version of the website's Strategy Builder profiling tool.

The output is a portfolio decision toolkit published as a static website.

## Research questions

| # | Question | Chapter |
|---|----------|---------|
| 1 | Do sectors actually diversify each other, or is it *fake diversification*? | Ch. 1 |
| 2 | How do market regimes (Bull / Neutral / Bear) change sector returns, volatility and correlations? | Ch. 2 |
| 3 | Do NASDAQ market categories (Global Select / Global Market / Capital Market) behave differently **within** the same sector? | Ch. 3 |
| 4 | Should you buy the sector ETF, or recreate the exposure yourself with a basket of stocks? | Ch. 4 |
| 5 | Within a sector, is one representative stock enough, or do you need a basket? | Ch. 5 |
| 6 | Can the findings be turned into a rule-based strategy that survives an honest out-of-sample test? | Ch. 6 |

---

## Data sources

| Dataset | Content | Use |
|--------|---------|-----|
| [Kaggle – Stock Market Dataset (jacksoncrow)](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) | One CSV per symbol with daily `Date, Open, High, Low, Close, Adj Close, Volume` for all NASDAQ-traded **stocks** and **ETFs** (history up to 1 April 2020), plus `symbols_valid_meta.csv` | Core price/volume data and base metadata |
| NASDAQ symbol directory (`nasdaqlisted.txt`) | Official `Market Category` and `Financial Status` per NASDAQ-listed symbol | Filling missing metadata |
| Yahoo Finance via `yfinance` | `sector` and `industry` per ticker | Sector classification (11 GICS-like sectors) |
| [SPY Historical Prices (Kaggle)](https://www.kaggle.com/datasets/benjaminbtang/spy-historical-prices) | S&P 500 ETF daily prices | Market proxy used to define regimes |

**The 11 sectors used:** Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities.

**Sector ETF mapping (SPDR Select Sector ETFs):** XLB, XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLRE, XLK, XLU (same order as above).

---

## End-to-end pipeline at a glance

```
raw CSVs (stocks/, etfs/)  ──►  preparing_stocks.ipynb / preparing_etfs.ipynb
                                 • date checks, missing/zero diagnostics
                                 • symbol removal + conservative repairs
                                 • volume imputation
                                        │
symbols_valid_meta.csv ──► preparing_metadata.ipynb
nasdaqlisted.txt           • normalise, enrich market category / financial status
yfinance sectors           • add Sector / Industry
                                        │
                                        ▼
                      data_cleaned/  (one CSV per ticker)
                      data_processed/ (aggregated parquet)
                        stocks_clean.parquet, etfs_clean.parquet, metadata_clean.parquet
                        returns_matrix.parquet (Date × Symbol log-returns)
                        returns_tidy.parquet   (long format + metadata)
                        etf_returns.parquet
                        stocks_sect(_clean).parquet (returns + within-sector weights + rolling vols)
                                        │
          ┌──────────────┬──────────────┼──────────────┬──────────────┐
          ▼              ▼              ▼              ▼              ▼
       Ch.1 sector    Ch.2 regimes   Ch.3 market    Ch.4 ETF vs    Ch.5 sector
       structure      (SPY drawdown) categories     DIY basket     proxy
          └──────────────┴──────────────┴──────────────┴──────────────┘
                                        │
                                        ▼
      Ch.6 systematic strategy (src/strategy/) + Strategy Builder JSON
                                        │
                                        ▼
                         website/ (static site on GitHub Pages)
```

---

## Data preparation & cleaning (stocks and ETFs)

Notebooks: `preparing_stocks.ipynb`, `preparing_etfs.ipynb`
Helpers: `src/utils/loading_data.py`, `src/utils/data_inspection.py`, `src/utils/data_inspection_viz.py`, `src/utils/data_cleaning.py`

### Core principle

Every downstream analysis (rolling windows, regimes, correlations, clustering) needs **continuous, time-consistent series**. So the cleaning works at the **symbol level** first and the **row level** second:

> If a symbol is structurally broken, it is removed.
> If a value can be safely repaired without breaking time consistency, it is fixed.

That means rows are never dropped blindly (that would create holes inside series and bias rolling statistics), and systematic problems remove the **whole symbol**. Cleaning is iterative: diagnostics are re-run after every step to check its effect.

### Step 1: Loading

- Each CSV is loaded into a `dict[symbol → DataFrame]` with **no cleaning at load time** (`load_symbol_csv`), so every later decision is explicit and traceable.
- The symbol is taken from the filename and `Date` is parsed with `errors="coerce"`.

### Step 2: Date sanity checks

- **Monotonicity:** check that dates are strictly increasing within every symbol (`dates_in_increasing_order`).
- **Coverage:** first/last date per symbol (`show_date_ranges`), symbols that **end before the 2020-04-01 cutoff** (`ends_before`, which detects delisted or truncated histories).
- **Start-date distribution** plot (`plot_start_date_distribution`), which shows that symbols enter the dataset at very different times (IPOs) and motivates the *pairwise-overlap* correlation design used later.
- A `Weekday` column (0 = Mon … 4 = Fri) is added for the volume imputation.

### Step 3: Missing-value analysis

- Missing rate **per symbol and per column**, plus a **global row-weighted** missing rate (`missing_rates_from_dict`).
- **Missing counts over time** plot (`plot_missing_counts_over_time`) to see whether gaps cluster in specific periods.
- Key finding: NaNs appear in **exactly the same symbols across all OHLCV columns**, which points to a structural problem with those symbols rather than scattered gaps.
- **Decision:** these symbols are removed: **≈4.35 % of stock symbols** and **≈0.5 % of ETF symbols**. Imputation would have meant inventing whole stretches of prices.

### Step 4: Zero-value analysis

- Zero rate per symbol/column (`zero_rates_from_dict`) and **zeros over time** for `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` (`plot_zero_counts_over_time`).
- A price of 0 is **economically invalid**: it is a data error, not a real observation.
- Symbols with zeros in `High`/`Low` are the same set, and so are symbols with zeros in `Close`/`Adj Close`. Both sets are **removed entirely**, because they indicate a corrupted series.

### Step 5: Conservative repairs

| Problem | Fix | Rationale |
|---------|-----|-----------|
| `Open == 0` (first row) | Drop that row; if the new first row also has `Open == 0`, fill it with the dropped row's `Adj Close` | No previous day exists to fill from |
| `Open == 0` (other rows) | Replace with **previous day's `Adj Close`** | The open is usually close to yesterday's close, and only **past** information is used |
| `Close` is NaN/0 | Replace with **same-day `Adj Close`** | `Adj Close` is valid and very close to `Close` (they differ only by split/dividend adjustments) |

### Step 6: Volume handling (most careful step)

1. **Intraday volatility proxy:** `Volatility = (High − Low) / Open` (a normalised daily range).
2. **Trim leading zero volumes:** remove rows at the *start* of a series until the first non-zero volume (pre-trading / listing artefacts).
3. **Impute remaining zero volumes, past-only (no look-ahead leakage):**
   - baseline = **median of the last 10 non-zero volumes on the same weekday** (volume has weekday seasonality); if there are none, use the median of the last 10 non-zero volumes on any day;
   - adjust for that day's activity using the **z-score of the volatility proxy**:
     $$\text{Volume}_t = \text{baseline} \times \max\left(0.01,\; 1 + 0.7 \cdot z_t^{\text{vol}}\right)$$
     (higher-than-usual price range ⇒ higher-than-usual volume, a well-known volume–volatility relationship).
4. Symbols whose zero volumes **still cannot be resolved** (no past info) are removed.

### Step 7: Final checks & export

- Missing and zero rates are recomputed to confirm the dataset is clean.
- Export:
  - `data_cleaned/{stocks,etfs}/<TICKER>.csv` (same layout as the raw data, one file per ticker)
  - `data_processed/stocks_clean.parquet` / `etfs_clean.parquet` (one long table with a `ticker` column; columnar and fast to load)

---

## Metadata cleaning & enrichment

Notebook: `preparing_metadata.ipynb`. Helpers: `src/utils/loading_data.py::load_nasdaq_metadata`, `src/utils/metadata_enrich.py`

Metadata links price dynamics to structural attributes (sector, exchange, market category, financial status). The goal is **correctness, not completeness**.

**Policy**
- Metadata is never blindly trusted.
- Existing valid values are **never overwritten**.
- Missing values are filled **only from reliable external sources**.
- If a value cannot be safely inferred, it stays missing.

**Steps**
1. **Typed loading:** everything is read as string, then whitespace is stripped, `Y/N` columns become nullable booleans, `Round Lot Size` becomes numeric, empty strings become `NA`, and columns are renamed to snake_case.
2. **Drop non-essential columns** such as `cqs_symbol`.
3. **Enrich `market_category` and `financial_status`** with a case-insensitive **left join** on the symbol against the official NASDAQ directory (`nasdaqlisted.txt`). Values are filled **only where they were missing** (`where(~isna, from_dir)`).
4. **Readable labels:**
   - Market category: `Q` = NASDAQ **Global Select** Market, `G` = NASDAQ **Global** Market, `S` = NASDAQ **Capital** Market. These are listing tiers ordered by size/listing requirements (Q strictest, S smallest companies).
   - Financial status: `N` Normal, `D` Deficient, `E` Delinquent, `Q` Bankrupt, and combinations.
5. **Diagnostics plots:** symbols per listing exchange, missing rate per exchange, distribution of market categories and financial status. Enrichment fills the market category for essentially **all NASDAQ-listed** symbols. Other exchanges (NYSE, etc.) have no NASDAQ tier by definition.
6. **Sector enrichment (yfinance):**
   - `check_valid_securities` pings each ticker (`history(period="1d")`) to measure Yahoo coverage. It is **resumable** (it saves progress to CSV and skips tickers already processed).
   - `create_update_sector_csv` fetches `info["sector"]` and `info["industry"]` for valid tickers, with rate limiting (`sleep(0.2)`), also resumable.
   - `add_sector_column` merges the sector into the final metadata.
7. Financial status was studied but **not used further**, because almost all symbols are "Normal" and the variable has no discriminative power.
8. Export: `data_processed/metadata_clean.parquet` (+ CSV copy).

---

## Feature engineering: returns, volatility, sector aggregation

### Log returns
$$r_{i,t} = \ln\left(\frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}}\right)$$
Computed on **`Adj Close`** (adjusted for splits and dividends). Log returns are **time-additive** (multi-day return = sum of daily returns) and closer to symmetric than simple returns.

- `returns_matrix.parquet`: wide `Date × Symbol` matrix. Only stocks with **≥ 3 years (756 days)** of data are kept. **No NaN filtering** is done, so each stock keeps its own history.
- `returns_tidy.parquet`: long format merged with metadata.
- `etf_returns.parquet`: same for ETFs.

### Volatility
- **Rolling volatility** = rolling standard deviation of log returns over windows of **20, 60, 120** trading days (≈ 1, 3, 6 months), annualised with $\sqrt{252}$ when needed.
- Window choice is a **bias–variance trade-off**: short windows react fast but are noisy, long windows are smooth but lag.

### Sector aggregation (volume-weighted)
Sector series are built from constituents with **volume-share weights** within each (Date, Sector):
$$w_{i,t} = \frac{V_{i,t}}{\sum_{j \in S} V_{j,t}}, \qquad R_{S,t} = \sum_{i \in S} w_{i,t}\, r_{i,t}$$
Weights sum to 1 per (Date, Sector), and this is checked in the notebooks. Volume weighting lets **actively traded names dominate** sector dynamics, which is a proxy for economic importance and liquidity when market capitalisation is not available in the dataset. Sector volatility is aggregated the same way: `SectorVol20 = Σ WeightedVol20`.

---

## Chapter 1: Sector–sector structure

*Question: how can investors truly diversify across sectors in the short and long term?*

### 1.0 Time windows
Three horizons are used to separate long-run structure from recent behaviour:
- **Full sample**: the common-coverage window where all sectors exist.
- **20Y**: 2000-04-01 → 2020-04-01.
- **5Y**: 2015-04-01 → 2020-04-01.

### 1.1 Return correlation (Pearson)
- Sector–sector **Pearson correlation** matrices on volume-weighted sector log returns, for each window.
- **Minimum overlap of 2 years (504 trading days)** per pair, so that unstable estimates are not reported.
- **Diversifier rule:** a sector is a "return diversifier" in a window if its **mean correlation with the other 10 sectors is < 0.25**.

**Findings**
- Full sample: correlations are mostly **0.15–0.40**, so sectors are never independent.
- **Diversification is polarizing, not disappearing.** The *average* pair correlation barely changes (0.27 → 0.25 → 0.24), but its spread widens:
  - **the cyclical core tightens:** Financials–Consumer Cyclical goes 0.21 → 0.58, Financials–Energy 0.37 → 0.56, Financials–Technology 0.35 → 0.52;
  - **the defensives decouple:** Healthcare–Utilities goes 0.17 → 0.02.
- **Consistent return diversifiers:** Healthcare, Industrials, Real Estate, Utilities. Healthcare's average correlation falls from 0.22 to 0.12.
- **Consumer Cyclical** was the best diversifier for decades (0.17, then 0.13) but lost this status in the last 5Y (0.32).
- **Never diversifiers:** Basic Materials, Energy, Financial Services, Technology. Financial Services becomes the most connected sector in 2015-2020 (average 0.40).

### 1.2 Volatility correlation (do sectors become risky together?)
- Rolling **60-day volatility** per sector, then the correlation of those volatility series.
- Return correlation tells you whether sectors **move in the same direction**. Volatility correlation tells you whether they **become risky at the same time**, which is the systemic-risk view.
- A stricter threshold (**< 0.15**) defines volatility diversifiers.
- **Findings:**
  - volatility correlations are *lower* than return correlations on average (full sample 0.22 vs 0.27), but more extreme for a few pairs: Energy–Basic Materials is 0.79 / 0.75 / 0.68.
  - **Consumer Cyclical, Industrials and Real Estate** are consistent volatility diversifiers.
  - In the last 5Y, Financials' risk is tied to Energy (0.59) and Technology (0.54, up from 0.21).
  - Caveat: average volatility correlation drops to 0.09 in the 5Y window, partly because 5 years hold few volatility cycles, so "5Y-only" diversifiers are weak evidence.

### 1.3 Rolling correlations
- **60-day** (reactive) and **120-day** (smoother) rolling Pearson correlations for selected pairs: a high-correlation pair (Energy–Financial Services), a low one (Consumer Cyclical–Utilities) and a "surprise" pair (Communication Services–Real Estate). (Notebook only.)
- **Finding:** correlations are strongly **time-varying** and spike in crises. A single static number hides exactly the moments when diversification fails.

### 1.4 PCA sector map
- **PCA** on each correlation matrix (returns and volatility, three windows), projecting the 11 sectors on **PC1/PC2** with **shared axis limits** across windows so the maps are comparable.
- PC1 typically captures the **common market factor**. Sectors far from the main cluster have distinct behaviour.
- **Findings:** **Utilities and Healthcare** are consistently isolated (return-based diversifiers). **Consumer Cyclical and Industrials** are isolated in volatility space.
- **Caveat:** a 2-D projection keeps only about 25-35 % of the structure of these matrices, because sector correlations are fairly uniform. The website therefore uses a clustered heatmap instead (see 1.5).

### 1.5 Website version
The raw data and the `chapter1/` helper package are not in this repository. So the website charts are rebuilt from the six correlation matrices saved in the `results` notebook outputs, using `src/chapter1_site_data.py` → `website/content/chapter1.json`. That script checks the transcription in two ways: each matrix must be symmetric, and re-applying the diversifier rule must reproduce the notebook's saved diversifier tables exactly. The site (`website/chapter1.js`, interactive SVG) shows two charts:
- a **dot plot** of each sector's average correlation in the 3 windows, against the threshold. It replaces the unlabeled Sankey diagram.
- a **clustered lower-triangle heatmap** with Full / 20Y / 5Y views and a "5Y minus Full" change view. Returns use a sequential scale (all positive); volatility and change use a diverging scale.

---

## Chapter 2: Market regimes and sector behaviour

*Question: how do sector returns, volatility and correlations change across market regimes?*

### 2.1 Regime definition (rule-based, using SPY drawdowns)
1. SPY log returns → cumulative index $I_t = 100\cdot e^{\sum r}$.
2. Running maximum $M_t = \max_{s\le t} I_s$ → **drawdown** $DD_t = I_t / M_t - 1$.
3. Labels:
   - **Bear:** $DD_t \le -20\%$ (the standard "bear market" convention),
   - **Neutral:** $-20\% < DD_t \le -5\%$,
   - **Bull:** $DD_t > -5\%$.
4. Timeline plot of SPY with regime-shaded background.

Why drawdown? It is **transparent, interpretable, and has no fitted parameters**, so it is easy to explain to investors. *(An alternative definition was also explored in `Clustering_H.ipynb`: Bear = high 60-day volatility (top quartile) **and** deep drawdown (bottom decile), Bull = low volatility (bottom quartile) **and** positive 60-day drift, Neutral = everything else.)*

### 2.2 Returns block
- **Sector daily return** = Σ (weight × stock return) per (Date, Sector). Dates with fewer than 10 sectors available are dropped for the matrix analysis.
- **Static correlation heatmap:** all pairs positive. Strongest pairs: Financial Services–Real Estate (≈ 0.71), Consumer Cyclical–Financial Services (≈ 0.63), Technology–Consumer Cyclical (≈ 0.58). Healthcare and Utilities are the least correlated.
- **60-day rolling average correlation** (mean of the off-diagonal upper triangle of the rolling correlation matrix): ranges from ≈ 0.15–0.30 in calm periods to **> 0.8** in crises (2008–09, 2011–12). *Systemic risk is mainly a correlation phenomenon.*
- **Boxplots of sector returns by regime:** Bull median > 0, Neutral ≈ 0, Bear slightly < 0. The Bear regime has the **widest dispersion** and the most extreme negative outliers.
- **Ranked bar charts of mean sector return per regime:**
  - *Bear:* Consumer Cyclical and Communication Services are the strongest. Defensives (Utilities, Consumer Defensive) are slightly negative (a counter-intuitive result, likely from rebound days inside bear phases).
  - *Neutral:* Healthcare leads, then Basic Materials and Energy. Consumer Cyclical is negative.
  - *Bull:* all sectors positive. Healthcare, Technology and Communication Services lead.
- **Regime-conditional hierarchical clustering:** for each regime, correlation → distance $d = 1 - \rho$ → **average-linkage** hierarchical clustering → heatmap reordered by dendrogram leaves (`leaves_list`).
  - *Bear:* one big, highly correlated block ("everything sells together").
  - *Neutral:* lower correlation, with clear sub-blocks (defensives vs cyclicals/financials).
  - *Bull:* moderate correlation and broad participation.

### 2.3 Volatility block
- `SectorVol20` (20-day, weight-aggregated) merged with regime labels.
- **Mean volatility per regime:** Bear volatility is **≈ 2× Bull**, and Neutral sits in between.
- **Top-8 most volatile sectors per regime** and a **Sector × Regime heatmap**, sorted by Bear volatility:
  - *Bear:* Financial Services and Real Estate (leverage/credit sensitive), then Materials, Consumer Cyclical, Energy.
  - *Neutral:* Materials, Energy (commodity/macro uncertainty), Tech.
  - *Bull:* volatility becomes **idiosyncratic**: Healthcare, Materials, Energy.

### 2.4 Statistical tests
| Test | Hypotheses | Why this test | Result |
|------|-----------|---------------|--------|
| **Kruskal–Wallis H** | H₀: sector volatility has the same distribution in Bear / Neutral / Bull | Non-parametric. Volatility is positive, right-skewed and non-normal, so ANOVA assumptions fail | **H = 4105.2, p ≈ 0** → reject H₀ |
| **Spearman rank correlation** of sector volatility rankings between regimes | H₀: no association between the rankings | Rank-based, measures whether the *same* sectors lead risk | Bear–Neutral ρ = 0.45 (p = 0.17), **Bear–Bull ρ = 0.07 (p = 0.83)**, **Neutral–Bull ρ = 0.82 (p = 0.002)** |

**Conclusion:** regimes change **both how much risk there is and which sectors carry it**. The risk structure moves smoothly between Neutral and Bull, then **breaks** in Bear markets. There is no universally "best" sector, so allocation needs to be **regime-aware**.

---

## Chapter 3: Market categories within sectors

*Question: once the sector is fixed, does choosing a NASDAQ market category change risk and return?*

### Market categories (NASDAQ listing tiers)
- **Q: Global Select Market**: the strictest listing standards, mostly large, established companies.
- **G: Global Market**: mid-tier companies.
- **S: Capital Market**: smaller, typically younger, less liquid companies.

The tiers act as a **size/maturity proxy**. The chapter tests whether that proxy still matters after controlling for sector.

### Part A: Static exploration
- Average return by category (overall).
- Average return by category **within each sector** (a grouped comparison that controls for sector).
- Average **60-day rolling volatility** by category within each sector.
- Combined return/volatility feature tables per (Sector × Category).

Static averages pool very different market environments into one number, so they can mislead.

### Part B: Dynamic return–volatility map (main result)
- For every **Sector × Category** cell: aggregate returns and volatility across stocks, **smooth over time** (rolling) to reduce noise, and track them through time.
- **Animated Plotly scatter:** x = smoothed volatility, y = smoothed return, colour = sector, marker = category, with an animation frame per time step and fixed axis ranges so frames are comparable.
- **Reading it:** when the categories of a sector **cluster together**, the sector dominates and the category choice adds little. When they **separate**, the category tilt materially changes risk/return.

### Takeaways
- Category effects are **not constant**: distinctions **collapse in stress** (sector and market factors dominate) and **re-emerge in calmer periods**.
- Category-based allocation only makes sense **together with sector structure and the market regime**.

---

## Chapter 4: Sector ETF vs DIY basket

*Question: should you buy the sector ETF or replicate the sector yourself?*

Code: `Helper_4.py` (`analyze_sector`, `build_sector_series`), `Question_3.ipynb`, `Run_file_3_modular.ipynb`

### DIY basket construction
1. **Universe:** all non-ETF stocks of the sector (from metadata).
2. **Data quality filter:** drop tickers with any |daily log return| > 1 (data errors) or with < 98 % non-missing returns on the common dates.
3. **Weights:** volume share $w_{i,t} = V_{i,t} / \sum_j V_{j,t}$.
4. **Lag by one day:** $w_{t-1}$ is used to weight $r_t$. This **avoids look-ahead bias**, because today's volume is not known before today's return is realised.
5. **Cap at 10 %** per stock, then **renormalise** to sum to 1. This limits concentration in a few heavily traded names, similar to how capped indices work.
6. Basket return $R^{B}_t = \sum_i w_{i,t-1} r_{i,t}$. ETF and basket are **aligned on identical trading dates**.

### Comparison metrics
| Metric | Definition | Meaning |
|--------|-----------|---------|
| Active return | $a_t = r^{ETF}_t - R^{B}_t$ | Daily difference between the two exposures |
| Correlation | Full sample + **median rolling 60/120-day** correlation | How similar the exposures are |
| **Tracking error (annualised)** | $TE = \text{std}(a_t)\cdot\sqrt{252}$, plus the median rolling 60/120-day TE | How far the basket strays from the ETF |
| **Effective number of stocks** | $N_{eff} = 1 / \sum_i w_i^2$ (inverse Herfindahl index), median over time | Real diversification of the basket |
| Concentration | Median and 99th percentile of the max weight | Single-name risk |

### Visualisations
- **Performance:** both series rebased to **100** on the same start date (comparing growth paths, not prices).
- **Rolling annualised 60-day volatility:** ETF vs basket.
- Dark-themed PNGs per sector (`website/chapter3_sector_png_split_pretty/`), plus a written per-sector interpretation and "trading takeaway" on the website (`website/content/chapter3_writeups.json`).

### Findings
- The **DIY basket tends to outperform** the ETF over the sample. **Real Estate is the exception**: there the ETF does better.
- **ETFs are generally less volatile**, thanks to their broader and capped constituent set.
- Rule of thumb: if volatility spikes while ETF–basket tracking error widens, treat it as a regime shift and prefer the ETF until tracking stabilises.

---

## Chapter 5: Sector proxy, or is one stock enough?

*Question: can a single stock represent a sector, or do you need a basket?*

Code: `chapter5.py` (`run_sector_analysis`), `Chapter_5.ipynb`

### Pipeline (per sector)
1. **Sector panel:** wide returns (`Date × ticker`), volume, and the **sector proxy** $R_{S,t} = \sum_i w_{i,t} r_{i,t}$ (volume-weighted).
2. **Liquidity filter:** keep stocks with **≥ 80 % of observed days with Volume > 0**, then among those keep stocks whose **average volume is above the sector's 20th percentile**. Only *observed* days count, so recent IPOs are not penalised.
3. **Representativeness score:** the **mean rolling correlation** (90-day window in the final run, min 40 obs.) between each stock and the sector proxy.
4. **Lifetime-aware ranking:**
   $$\text{effective\_score}_i = \text{rep\_score}_i \times \left(\text{lifetime\_fraction}_i\right)^{\alpha}, \quad \alpha = 2$$
   where `lifetime_fraction` = share of the panel's dates on which the stock has data. This **penalises short histories**: a recent stock can look highly correlated over a short, favourable window, but it cannot represent the sector over decades. Ties are broken by **lower volatility**.
5. **Selection:** the **single best stock** (rank 1) and an **equal-weighted Top-5 basket**. Weights are renormalised over stocks with data each day.
6. **Risk metrics per stock:** volatility, **maximum drawdown**, and **Expected Shortfall (ES 5 %)**, the mean of returns below the 5 % VaR quantile.
7. **Intra-sector structure (unsupervised ML):**
   - Stock–stock **pairwise-overlap correlation**: each pair uses all dates where both stocks exist.
   - **Angular distance** $d = s\sqrt{2(1-\rho)}$. Undefined correlations are filled with a **large distance** (max), which discourages merging unknown pairs. Tickers with > 20 % missing pairwise correlations are flagged as a separate **"MissingData" (-99)** group and pushed far away on the plot.
   - **UMAP** (`metric="precomputed"`, `n_neighbors=10`, `min_dist=0.01`) → 2-D embedding.
   - **HDBSCAN** on the UMAP coordinates (`min_cluster_size=5`), which finds dense clusters and labels the rest as noise. Agglomerative clustering (complete linkage, distance threshold at the 30th percentile) and HDBSCAN on raw distances are also implemented as alternatives.
8. **Validation:**
   - **Normalised index comparison:** sector proxy vs single stock vs Top-5 basket, each rebased.
   - **Rolling tracking error** $TE_t = \text{std}_{w}(r^{proxy}_t - R_{S,t})$ over 60- and 120-day windows, single vs basket, on a shared y-axis across sectors.
9. Interactive Plotly figures exported to HTML and embedded via iframes on the website (`website/chapter_5_plots/`).

### Findings
- **Energy** (and almost Communication Services) is dominated by a common factor, so **one well-chosen stock is enough**.
- **Technology, Healthcare, Financial Services, Industrials, Consumer Defensive, Utilities, Basic Materials:** a single stock captures the direction but has large tracking error. A **Top-5 basket** clearly improves correlation and stability (e.g. Basic Materials: single AA ρ ≈ 0.68 → basket ρ ≈ 0.80. Consumer Defensive: PG ρ ≈ 0.60 → basket ρ ≈ 0.79).
- **Consumer Cyclical** is strongly **regime-dependent**: the basket helps but cannot remove large deviations.
- **Structural cohesion in UMAP does not guarantee good tracking.** Some sectors look concentrated in UMAP space but still show TE spikes. **Rolling tracking error is the decisive validation metric.**
- A small, behaviourally selected basket captures most of the diversification benefit at a fraction of the complexity of full replication.

---

## Chapter 6: Systematic sector strategy

**Code:** `src/strategy/` (`sector_rotation.py`, `profiles.py`, `data.py`) · **Notebook:** `Chapter_6_strategy.ipynb` · **Tests:** `tests/test_strategy.py` · **Figures:** `plots_ch6/`

### Why the original Strategy Builder was replaced
The first version (in `results.ipynb`, still live on the website) was a **lookup table** of historical return/volatility for each configuration (sector × market category × ETF/all/top-5 × regime = 528 rows), each sorted into one of 9 profiles. It was not a strategy, because it had no rule for *when* to hold *what*. It also had three bugs:

1. **Volume-weighted *price* index** $\sum P\cdot V/\sum V$: a $500 stock joining a basket of $20 stocks shows up as a +2 400 % "return". Median annualised volatility in `strategy_stats.json` is **508 %** for "all" baskets and **276 %** for "top5", against 22 % for ETFs.
2. **All 396 regime rows are empty** (`n_days = 0`, NaN metrics), so the bull/neutral/bear dropdown never showed real numbers.
3. **NaN → "gambler":** `NaN < threshold` is `False`, so empty rows fell into the high-return/high-volatility bucket. That is why **412 of 528** rows were "gambler".

`src/strategy/profiles.py` fixes all three:
- basket returns $= \sum_i w_{i,t-1} r_{i,t}$, with weights renormalised over the stocks trading that day, so listings change the weights and never the index level;
- regime statistics come from regime days only, using the annualised mean log return because regime days are not contiguous;
- tertiles are computed **within each regime**;
- NaN → "insufficient data";
- ETF rows are no longer duplicated across Q/G/S.

The notebook applies the fix to the sector ETFs. Regenerating the stock-basket rows needs the full Kaggle stock data, and the website still reads the old JSON.

### The strategy: trend-switched defensive rotation with risk targeting
Every rule maps to an earlier finding:

| Component | Rule | Comes from |
|---|---|---|
| Universe | 11 SPDR sector ETFs; each becomes eligible after 1 year of history (XLRE 2016, XLC 2019) | Ch. 4/5: ETFs are cheap, liquid, less volatile than stock baskets |
| Base weights | inverse 63-day volatility | Ch. 1: sector volatilities differ a lot |
| Regime switch | if SPY < its 200-day moving average → hold only the **3 sectors with the lowest trailing 1-year beta** to SPY | Ch. 2: defensive, weakly correlated sectors hold up in bad regimes |
| Risk sizing | scale the book to **12 % ex-ante volatility** using the 63-day **covariance matrix** (so correlation spikes count); max 100 % invested, rest in T-bills | Ch. 1/2: volatility and correlations spike together, so diversification fails when it's needed |
| Rebalancing | monthly | keeps turnover low (~2.4× NAV/yr) |

The defensive sleeve is chosen **by data** (trailing beta), not hard-coded from our own full-sample Chapter 2 results, which would be look-ahead. It still ends up holding Consumer Defensive (~97 % of bear months), Utilities (~96 %) and Healthcare (~60 %), the same sectors Chapter 2 found most resilient.

### Backtest hygiene (no look-ahead)
- Signals use data up to the **month-end close**; trades execute at the **next close** (`exec_lag=1`).
- Weights **drift** with prices between rebalances (no free daily rebalancing).
- **10 bps** cost per unit of traded notional; cash earns the 13-week T-bill (`^IRX`); Sharpe uses **excess** returns.
- Prices: total-return adjusted ETF/SPY prices from Yahoo Finance (same convention as `Adj Close`).
- Unit tests check that changing *future* prices does not change today's weights, that the weight drift matches an analytical buy-and-hold, and that the execution lag and costs behave correctly.

### Design / validation / holdout protocol
| Period | Dates | Role |
|---|---|---|
| Design | 1999-12 → 2012-12 | the **only** period used to make choices |
| Validation | 2013 → 2020-04-01 | rest of the project's dataset; not used for choices |
| Holdout | 2020-04 → 2026-11 | data that **did not exist** in the project dataset |

**Choices made on the design period.** 6 pre-declared variants were compared: 3 regime definitions × with/without a 12-1 momentum sleeve.
- **The Ch. 2 regime (drawdown from all-time high ≤ −20 %) is a poor *trading* signal.** After the 2000 peak, SPY stayed more than 20 % below its high until late 2004 (in total-return terms), so 42 % of the design period, including most of the 2003-04 recovery, is labelled "bear". The 200-day trend filter reacts within weeks and won with every selection rule (Sharpe 0.23 vs 0.20).
- **Momentum was dropped.** The 12-1 top-4 momentum sleeve had a lower Sharpe (0.20 vs 0.23) at more than 2× the turnover. Sector momentum is known to have struggled in the 2000s and crashed in 2009.

### Results (net of costs)
| | Period | CAGR | Vol | Sharpe | Max DD | Calmar | Beta |
|---|---|---|---|---|---|---|---|
| **Strategy** | full 1999-2026 | 7.2 % | 12.4 % | **0.46** | **−31 %** | **0.23** | 0.53 |
| EW sectors | full | 8.8 % | 18.0 % | 0.45 | −53 % | 0.17 | 0.90 |
| SPY | full | 8.2 % | 19.2 % | 0.41 | −55 % | 0.15 | 1.00 |
| **Strategy** | validation 2013-20 | 7.6 % | 13.6 % | 0.55 | −30 % | 0.26 | 0.79 |
| SPY | validation | 10.0 % | 16.2 % | 0.62 | −34 % | 0.30 | 1.00 |
| **Strategy** | holdout 2020-26 | 12.1 % | 11.0 % | 0.82 | **−11 %** | **1.07** | 0.52 |
| SPY | holdout | 20.2 % | 17.3 % | 0.98 | −25 % | 0.82 | 1.00 |

**Ablation** (full period, each row adds one idea):

| Variant | Sharpe | Max DD | Turnover / yr |
|---|---|---|---|
| Equal-weight sectors | 0.45 | −53 % | 0.4 |
| + inverse-vol | 0.46 | −50 % | 0.8 |
| + trend regime switch | 0.46 | −40 % | 2.6 |
| + vol target (final) | 0.46 | −31 % | 2.4 |

Each layer mainly buys **drawdown protection**, not Sharpe.

**Robustness checks**
- **Vol-matched benchmark** (static SPY + T-bills at the same volatility) tests whether the rules do more than "hold less stock". Over the full period the strategy wins: Sharpe 0.46 vs 0.41, max DD −31 % vs −39 %. That edge comes from the 2000-2012 bear markets, though. In 2013-2026 the static mix had the higher Sharpe, and the strategy only kept a smaller drawdown (holdout −11 % vs −16 %).
- **Stationary block bootstrap** (Politis-Romano, mean block 21 days, which keeps autocorrelation and volatility clustering) of the Sharpe difference: **no 90 % interval excludes 0** (full period vs SPY: +0.06, CI [−0.10, +0.22]).
- **Parameter sensitivity** (design period, 60 settings of vol target 8-20 %, 2-5 defensive sectors, 42/63/126-day windows): Sharpe 0.16-0.28, **all** far above SPY's 0.09, so the result does not depend on one lucky setting. 4-5 defensive sectors would have scored slightly higher than the pre-declared 3, but we kept 3 so as not to tune on the grid.
- **Costs:** Sharpe 0.48 at 0 bps → 0.46 at 10 bps → 0.39 at 50 bps.

### Honest conclusion
It is a **risk-managed sector allocation**: about **half the drawdown of the market** with a similar Sharpe. It is **not** an alpha generator. It lags SPY on return, mostly in the 2013-2026 bull market dominated by mega-cap tech, and no Sharpe difference is statistically significant.

**Remaining limitations**
- one market and one asset class;
- the ETF list and sector definitions are known ex post;
- 6 variants were compared on the design period, which biases its Sharpe upward;
- the trend filter whipsaws in sideways markets (2011, 2015-16, 2018);
- no taxes.

**Next steps**
- a bond/gold sleeve for the bear regime instead of low-beta equities;
- the Ch. 5 top-5 baskets as an alternative implementation;
- walk-forward re-estimation of the parameters;
- regenerating the website's Strategy Builder JSON with the corrected `profiles.py`.

---

## Additional ML experiments

### Stock-level market structure (`Clustering_H.ipynb`)
Goal: build data-driven "correlation risk groups" and check whether they recover sectors.
1. **Pairwise-overlap correlation** across all stocks (min overlap 2 years) → **correlation distance** $d = \sqrt{2(1-\rho)}$. This is the Euclidean distance between standardised return series, so it is a proper metric and suits Ward linkage.
2. **Hierarchical clustering (Ward)**, with NaN distances filled by the median, cut at k = 5, 10, 15, plus a truncated dendrogram.
3. **UMAP** (`n_neighbors=30`, `min_dist=0.05`, precomputed distances) → 2-D market map, coloured by sector.
4. **HDBSCAN** on the embedding (`min_cluster_size = max(20, n/200)`, EOM selection): finds clusters without fixing k and flags **outliers/noise**.
5. **K-Means** on the UMAP coordinates for k = 5, 8, 10, 15.
6. **Regime-specific clustering:** correlations recomputed on bull / neutral / bear days only (min overlap 252 / 252 / 60 days), then Ward dendrograms on a **shared y-scale** so cluster tightness is comparable.
7. **External evaluation against sectors:**
   - **Purity**: share of stocks belonging to their cluster's majority sector,
   - **NMI** (normalised mutual information),
   - **ARI** (adjusted Rand index, chance-corrected),
   - **pairwise ARI between methods** (stability/agreement), and cluster × cluster contingency tables.

### Sector clustering on volatility profiles (`clustering_rolling_volatilitties.ipynb`)
Each sector is described by the mean/median/etc. of its 20/60/120-day rolling volatilities, then **StandardScaler → DBSCAN** (`eps=1.3`, `min_samples=2`), with a **PCA** 2-D projection for visualisation.

### Content-based stock recommender (`exploration.ipynb`)
A prototype "find stocks similar to my portfolio":
- **Features per stock (last 365 days):** mean daily return, volatility, momentum (1 m / 3 m / 12 m), average dollar volume (liquidity), and a one-hot sector.
- **StandardScaler** → user "style vector" = **mean of the holdings' vectors** → **cosine similarity** to every stock → top-k, excluding current holdings.
- **Explainability:** each recommendation gets a generated sentence ("higher volatility, similar 3-month momentum, lower liquidity…") comparing the stock's raw features to the user's average with a ±15 % tolerance.
- **PCA** 2-D plots show the user's style point, the recommended stocks and the sector layout.

---

## Visualisation approach

- **Diagnostics first:** time-series plots of missing/zero counts and start-date histograms drove the cleaning decisions.
- **Heatmaps** (custom diverging colour maps, fixed range −1 to 1) for correlation matrices. **Clustered heatmaps** reorder rows by dendrogram leaves so that blocks become visible.
- **Dendrograms** truncated to the last p merges for readability, with a shared y-scale when comparing regimes.
- **Dot plot** (website) of each sector's average correlation per window against the diversifier threshold. It replaced the notebook's Sankey diagram, which could not show *which* sector was in each band.
- **PCA maps** (notebook) with shared axis limits across windows. On the website they are replaced by a clustered heatmap, because 2-D maps kept only ~25-35 % of the structure.
- **Regime-shaded timelines** (SPY index with a Bull/Neutral/Bear background).
- **Boxplots and ranked bar charts** for return distributions per regime.
- **Animated Plotly scatter** for the Sector × Category return–volatility map (Ch. 3).
- **Interactive Plotly** UMAP maps, rebased index comparisons and rolling tracking-error charts, exported to HTML and embedded via iframes (Ch. 5).
- A consistent **dark theme** across the matplotlib and Plotly outputs to match the website.

---

## Design decisions, limitations & possible improvements

**Key design decisions (and why)**
- **Log returns on Adj Close:** time-additive, and corrected for splits and dividends.
- **Pairwise-overlap correlations** instead of a common date window: they keep decades of history and all stocks (IPOs from the 1980s to 2015+) instead of truncating to the shortest series. The cost is that the matrix may not be positive semi-definite, which is acceptable for distances and visual maps.
- **Volume weighting** (no market-cap data in the dataset): a liquidity/importance proxy, **lagged** by one day and **capped** at 10 % in the ETF comparison.
- **Non-parametric tests** (Kruskal–Wallis, Spearman) because financial returns and volatilities are skewed and heavy-tailed.
- **Rule-based regimes** for transparency. Alternatives would be Hidden Markov Models, Markov-switching models or clustering on volatility/return features.
- **Unsupervised ML (UMAP + HDBSCAN)** because there are no labels for "true" sector structure. Sectors are used only for **external validation** (purity, NMI, ARI).

**Known limitations**
- **Survivorship bias:** the dataset only contains symbols that were still listed in 2020, and broken symbols were removed during cleaning.
- **Sample ends 2020-04-01**, so it includes only the start of the COVID crash.
- **Sector labels come from today's yfinance classification** and are applied backwards in time (companies can change sector).
- **Statistical tests on daily pooled data:** observations are autocorrelated (rolling volatility in particular), so the Kruskal–Wallis p-value is overstated. Spearman uses only 11 sectors, so its power is low. Block bootstrap or tests on non-overlapping windows would be more rigorous.
- **Website Strategy Builder JSON is still the old version:** it has the volume-weighted-price, empty-regime and NaN→"gambler" bugs described in Chapter 6. The fix is in `src/strategy/profiles.py`, but regenerating the stock-basket rows needs the full Kaggle data.
- **No transaction costs, rebalancing costs or slippage** in the basket comparisons.
- **Volume imputation** is heuristic (weekday median scaled by the volatility z-score). It only affects a very small share of rows.

**Natural extensions**
- HMM-based regimes, and forecasting regime probabilities.
- Minimum-variance or risk-parity weights for the DIY basket, and comparing them to volume weighting.
- Out-of-sample backtests of the "use the ETF when tracking error widens" rule (Ch. 6 does this for the sector-rotation rules).
- Market-cap weighting with an external data source.

---

## Repository map

```
├── preparing_stocks.ipynb            # stock cleaning pipeline
├── preparing_etfs.ipynb              # ETF cleaning pipeline (same policy)
├── preparing_metadata.ipynb          # metadata normalisation + NASDAQ / yfinance enrichment
├── src/utils/
│   ├── loading_data.py               # raw CSV + NASDAQ metadata loaders (no cleaning at load)
│   ├── data_inspection.py            # date / missing / zero diagnostics
│   ├── data_inspection_viz.py        # diagnostic plots over time
│   ├── data_cleaning.py              # symbol removal, Open/Close repairs
│   └── metadata_enrich.py            # enrichment, yfinance sector fetch, log-return / volatility helpers
├── Clustering_H.ipynb                # stock-level correlation, Ward, UMAP+HDBSCAN, K-Means, regime clustering, NMI/ARI
├── chapter4_returns.ipynb            # (Ch. 2) regimes, sector returns, rolling corr, regime clustering
├── chapter4_volat.ipynb              # (Ch. 2) sector volatility by regime, Kruskal–Wallis, Spearman
├── rolling_volatiltiy_by_sectors_over_time.ipynb   # 20/60/120-day rolling volatility by sector
├── clustering_rolling_volatilitties.ipynb          # DBSCAN + PCA on sector volatility profiles
├── Question_3.ipynb, Run_file_3_modular.ipynb, Helper_4.py   # (Ch. 4) ETF vs DIY basket
├── Chapter_5.ipynb, chapter5.py      # (Ch. 5) sector proxy, representativeness, UMAP/HDBSCAN, tracking error
├── Chapter_6_strategy.ipynb          # (Ch. 6) systematic strategy: design/validation/holdout, ablation, robustness
├── src/strategy/
│   ├── data.py                       # ETF/SPY/T-bill download + cache, study periods
│   ├── sector_rotation.py            # regimes, signals, backtest engine, metrics, block bootstrap
│   └── profiles.py                   # corrected Strategy Builder (return-compounded baskets, regime stats)
├── tests/test_strategy.py            # no-look-ahead, costs, execution lag, builder bug regressions
├── src/chapter1_site_data.py         # (Ch. 1) website data: verified correlation matrices → website/content/chapter1.json
├── exploration.ipynb                 # content-based stock recommender prototype
├── results.ipynb.zip                 # consolidated results notebook (all chapters + Strategy Builder)
├── plots/, plots_ch2/, plots_ch6/    # exported figures
├── website/                          # static site (HTML/CSS/JS, JSON content, embedded Plotly HTML)
└── .github/workflows/pages.yml       # GitHub Pages deployment of website/
```

> Note: the notebook file names come from an earlier chapter numbering (e.g. `chapter4_*` = Chapter 2 on the website, `Question_3`/`Helper_4` = Chapter 4). The consolidated `results` notebook imports helper packages (`chapter1/`, `chapter2/`, `chapter3/`) that are not included in this repository.

---

## How to run the code

### 1. Clone the repository
```console
git clone https://github.com/chedliabd/cross-exchange-asset-analysis.git
cd cross-exchange-asset-analysis
```

### 2. Create a virtual environment and install the requirements
```console
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the data
- Download the [Kaggle stock market dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) and place `stocks/`, `etfs/` and `symbols_valid_meta.csv` under `data/`.
- Download `nasdaqlisted.txt` (NASDAQ Trader symbol directory) into `data/`.
- Download the [SPY prices](https://www.kaggle.com/datasets/benjaminbtang/spy-historical-prices) as `spy.csv` at the project root.

### 4. Run the preprocessing notebooks (from the project root)
1. `preparing_stocks.ipynb`
2. `preparing_etfs.ipynb`
3. `preparing_metadata.ipynb` (the yfinance sector fetch is slow but resumable)

These write `data_cleaned/` and `data_processed/`.

### 5. Generate results
Unzip `results.ipynb.zip` and run it to reproduce all chapters, or run the individual chapter notebooks listed in the repository map.

### 6. Chapter 6 strategy (independent of steps 3-5)
It only needs ETF prices, which are downloaded from Yahoo Finance on first run and cached in `data/strategy/prices.csv`:
```console
python -m pytest tests/test_strategy.py
jupyter nbconvert --to notebook --execute --inplace Chapter_6_strategy.ipynb
```
Delete the cache, or call `download_prices(refresh=True)`, to extend the holdout to today.

---

## Tech stack

**Python:** pandas, NumPy, SciPy (hierarchical clustering, Kruskal–Wallis, Spearman), scikit-learn (StandardScaler, PCA, K-Means, DBSCAN, Agglomerative, cosine similarity, NMI/ARI), umap-learn, hdbscan, yfinance, matplotlib, seaborn, Plotly, tqdm, PyArrow/Parquet, pytest.
**Web:** HTML/CSS/vanilla JavaScript, embedded Plotly HTML, JSON content files, GitHub Actions → GitHub Pages.
