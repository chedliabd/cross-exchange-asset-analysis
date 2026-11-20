
# An Investor's Guide to Building a Smarter Portfolio

**Live website:** https://chedliabd.github.io/cross-exchange-asset-analysis/
(The `website/` folder is deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`.)

---

## Table of contents

1. [Abstract](#abstract)
2. [Research questions](#research-questions)
3. [Data sources](#data-sources)
4. [End-to-end pipeline at a glance](#end-to-end-pipeline-at-a-glance)
5. [Data preparation & cleaning (stocks and ETFs)](#data-preparation--cleaning-stocks-and-etfs)
6. [Metadata cleaning & enrichment](#metadata-cleaning--enrichment)
7. [Feature engineering: returns, volatility, sector aggregation](#feature-engineering-returns-volatility-sector-aggregation)
8. [Chapter 1: Sector–sector structure](#chapter-1-sectorsector-structure)
9. [Chapter 2: Market regimes and sector behaviour](#chapter-2-market-regimes-and-sector-behaviour)
10. [Chapter 3: Market categories within sectors](#chapter-3-market-categories-within-sectors)
11. [Chapter 4: Sector ETF vs DIY basket](#chapter-4-sector-etf-vs-diy-basket)
12. [Chapter 5: Sector proxy, or is one stock enough?](#chapter-5-sector-proxy-or-is-one-stock-enough)
13. [Chapter 6: Systematic sector strategy](#chapter-6-systematic-sector-strategy)
14. [Visualisation approach](#visualisation-approach)
15. [Design decisions, limitations & possible improvements](#design-decisions-limitations--possible-improvements)
16. [Repository map](#repository-map)
17. [How to run the code](#how-to-run-the-code)
18. [Tech stack](#tech-stack)



---

## Abstract

This project has one practical goal: to help investors gain the most market understanding and risk control for the least effort. 
We avoid abstract statistics and build a step-by-step guide that answers real portfolio questions. 
It works on ~20+ years of daily NASDAQ-traded stock and ETF prices and covers:

- **sector structure** (who moves with whom, and is diversification real or fake?),
- **market regimes** (how sector risk and return change in Bull / Neutral / Bear / Recovery markets),
- **market categories** (does the NASDAQ listing tier matter once the sector is fixed?),
- **implementation** (sector ETF vs a do-it-yourself basket; one stock vs a small basket),
- a **systematic, backtested sector strategy** that turns these findings into trading rules (trend-switched defensive rotation with volatility targeting), validated out of sample up to 2026, plus a corrected version of the website's Strategy Builder profiling tool.

The output is a portfolio decision toolkit published as a static website.

## Research questions

| # | Question | Chapter |
|---|----------|---------|
| 1 | Do sectors actually diversify each other, or is it *fake diversification*? | Ch. 1 |
| 2 | How do market regimes (Bull / Neutral / Bear / Recovery) change sector returns, volatility and correlations? | Ch. 2 |
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

**The 11 sectors used:** Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities.

**Sector ETF mapping (SPDR Select Sector ETFs):** XLB, XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLRE, XLK, XLU (same order as above).

---

## End-to-end pipeline at a glance

```
data/ (raw)                              data_processed/ (one version of each dataset)
───────────                              ─────────────────────────────────────────────
stocks/*.csv ──► preparing_stocks.ipynb ──► stocks_clean.parquet
etfs/*.csv   ──► preparing_etfs.ipynb   ──► etfs_clean.parquet (includes SPY)
symbols_valid_meta.csv ┐
nasdaqlisted.txt       ├► preparing_metadata.ipynb ──► metadata_clean.parquet
Yahoo Finance (cached) ┘
                         preparing_sectors.ipynb ──► stocks_sect.parquet    (sector, returns, rolling vols,
                                                                             dollar-volume weights)
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
Helpers: `src/utils/loading_data.py`, `src/utils/data_inspection.py`, `src/utils/data_inspection_viz.py`, `src/utils/data_cleaning.py`, `src/utils/data_export.py`

Both notebooks follow the same sections and import all their helpers from `src/utils/`, so stocks and ETFs are cleaned with exactly the same code.

### Core principle

Every downstream analysis (rolling windows, regimes, correlations, clustering) needs **continuous, time-consistent series**. So the cleaning works at the **symbol level** first and the **row level** second:

> If a symbol is structurally broken, it is removed.
> If a value can be safely repaired without breaking time consistency, it is fixed.

That means rows are never dropped blindly (that would create holes inside series and bias rolling statistics), and systematic problems remove the **whole symbol**. Cleaning is iterative: diagnostics are re-run after every step to check its effect.

### Step 1: Loading

- Each CSV is loaded into a `dict[symbol → DataFrame]` with **no cleaning at load time** (`load_all_symbols`), so every later decision is explicit and traceable.
- The symbol is taken from the filename and `Date` is parsed with `errors="coerce"`.

### Step 2: Date sanity checks

- **Monotonicity:** check that dates are strictly increasing within every symbol (`symbols_with_unsorted_dates`).
- **Coverage:** first/last date and row count per symbol (`date_range_table`).
- **Truncated histories:** symbols that **end before the 2020-04-01 cutoff** (`ends_before`, e.g. delisted symbols) are **removed**, since they can't be compared with the others over the same period.
- **Start-date distribution** plot (`plot_start_date_distribution`), which shows that symbols enter the dataset at very different times (IPOs) and motivates the *pairwise-overlap* correlation design used later.
- A `Weekday` column (0 = Mon … 4 = Fri) is added for the volume imputation.

### Step 3: Missing-value analysis

- Missing rate **per symbol and per column**, plus a **global row-weighted** missing rate, on the six raw price/volume columns (`missing_rates_from_dict`).
- **Missing counts over time** plot (`plot_missing_counts_over_time`) to see whether gaps cluster in specific periods.
- Key finding: NaNs appear in **exactly the same symbols across all OHLCV columns**, which points to a structural problem with those symbols rather than scattered gaps.
- **Decision:** these symbols are removed: **≈4.35 % of stock symbols** and **≈0.5 % of ETF symbols**. Imputation would have meant inventing whole stretches of prices.

### Step 4: Zero-value analysis

- Zero rate per symbol/column on the six raw price/volume columns only (`zero_rates_from_dict`; `Weekday` and `Volatility` are left out because 0 is a valid value there) and **zeros over time** for `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` (`plot_zero_counts_over_time`).
- A price of 0 is **economically invalid**: it is a data error, not a real observation.
- Symbols with zeros in `High`/`Low` are the same set, and so are symbols with zeros in `Close`/`Adj Close`. Both sets are **removed entirely**, because they indicate a corrupted series.
- Any **negative** raw price (`Open`, `High`, `Low`, `Close`) or volume is impossible, so those symbols are removed too (`get_symbols_with_negatives`).
- A non-positive **`Adj Close`** is a known Yahoo artifact on old data of high-dividend stocks (the backward dividend adjustment pushes it below zero; e.g. Novo Nordisk, Unilever, Santander). The rest of their history is valid, so instead of removing them, `trim_before_last_nonpositive` starts their series the day after the last non-positive value.

### Step 5: Conservative repairs

| Problem | Fix | Rationale |
|---------|-----|-----------|
| `Open == 0` (first row) | Drop that row | No previous day exists to fill from |
| `Open == 0` (other rows) | Replace with **previous day's `Close`** | The open is usually close to yesterday's close, both are raw (unadjusted) prices, and only **past** information is used. `Adj Close` is not used because split/dividend adjustments can put it far from the raw price on older data |

### Step 6: Volume handling (most careful step)

1. **Trim leading zero volumes:** remove rows at the *start* of a series until the first non-zero volume (pre-trading / listing artefacts).
2. **Intraday volatility proxy:** `Volatility = (High − Low) / Open` (a normalised daily range).
3. **Impute remaining zero volumes** (`fill_zero_volumes`):
   - baseline = **median of the last 10 non-zero volumes on the same weekday**, using earlier days only (volume has weekday seasonality); if there are none, use the median of the last 10 non-zero volumes on any day;
   - adjust for that day's activity using the **z-score of the volatility proxy** (mean and standard deviation over the symbol's history):
     $$\text{Volume}_t = \text{baseline} \times \max\left(0.01,\; 1 + 0.7 \cdot z_t^{\text{vol}}\right)$$
     (higher-than-usual price range ⇒ higher-than-usual volume, a well-known volume–volatility relationship).
4. Symbols whose zero volumes **still cannot be resolved** (no past info) are removed.

### Step 7: Final checks & export

- Missing and zero rates are recomputed to confirm the dataset is clean, and the number of symbols kept is reported.
- Export (`export_clean_parquet`): `data_processed/stocks_clean.parquet` / `etfs_clean.parquet`, one long table sorted by `Symbol` and `Date`. This is the **only** saved version of the cleaned prices; every later notebook reads it.

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
3. **Enrich `market_category` and `financial_status`** with a case-insensitive **left join** on the symbol against the official NASDAQ directory (`nasdaqlisted.txt`). Values are filled **only where they were missing**.
4. **Readable labels** ([NASDAQ Trader definitions](https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs)):
   - Listing exchange: `Q` NASDAQ, `N` NYSE, `A` NYSE American, `P` NYSE Arca, `Z` Cboe BZX (BATS), `V` IEX.
   - Market category: `Q` = NASDAQ **Global Select** Market, `G` = NASDAQ **Global** Market, `S` = NASDAQ **Capital** Market. These are listing tiers ordered by size/listing requirements (Q strictest, S smallest companies).
   - Financial status: `N` Normal, `D` Deficient, `E` Delinquent, `Q` Bankrupt, `G` Deficient & Bankrupt, `H` Deficient & Delinquent, `J` Delinquent & Bankrupt, `K` all three.
   - Note that `Q` means *NASDAQ* as a listing exchange but *Global Select* as a market category, so the notebook filters on the readable labels.
5. **Diagnostics plots:** symbols per listing exchange, missing rate per exchange, distribution of market categories and financial status. Market category and financial status only exist for **NASDAQ-listed** symbols; other exchanges (NYSE, etc.) have no NASDAQ tier by definition.
6. Financial status was studied but **not used further**, because almost all symbols are "Normal" and the variable has no discriminative power.
7. **Sector and industry (yfinance), stocks only:**
   - `fetch_yfinance_profiles` looks up each stock's company name, `sector` and `industry` (Kaggle `BRK.B` → Yahoo `BRK-B`). Results are cached in `data/yfinance_profiles.csv` with the lookup date (`fetched_on`); the fetch is **resumable** and rate-limited.
   - Yahoo returns a ticker's *current* profile, not its 2020 one. To avoid tickers reused by another company since 2020, `add_sector_industry` only keeps a sector when Yahoo's company name matches the 2020 `security_name` (at least one significant word in common, `yf_name_match`). Otherwise Sector and Industry stay missing.
   - `sector_coverage` reports how many stocks have a sector, per exchange.
8. Export: `data_processed/metadata_clean.parquet`, the only saved version of the metadata.

---

## Feature engineering: returns, volatility, sector aggregation

### Daily returns
$$r_{i,t} = \frac{P^{adj}_{i,t}}{P^{adj}_{i,t-1}} - 1$$
Computed on **`Adj Close`** (adjusted for splits and dividends, so a return includes dividends). **Simple** returns are used because every sector series and basket in the project is a weighted average of stock returns, and a portfolio's return is the weighted sum of its stocks' simple returns.

### Sector table (`preparing_sectors.ipynb`, helpers in `src/utils/sector_panel.py`)
`stocks_sect.parquet` has one row per stock and day, for stocks with a verified sector, over each stock's **full history** (up to 2020-04-01), keeping stocks with at least **252 days** of prices. It is read by Chapters 1, 2, 3, 4 and 5 and by the Strategy Picker, and each chapter picks its own time window.

- **`Return`**: simple daily return of `Adj Close` (a sector return is a weighted average of simple returns), capped at **±50 %** (the notebook reports how many returns are capped).
- **`RollingVol{20,60,120}`**: rolling standard deviation of daily returns over **20, 60, 120** trading days (≈ 1, 3, 6 months). Window choice is a **bias–variance trade-off**: short windows react fast but are noisy, long windows are smooth but lag.
- Returns and rolling volatilities are computed on each stock's **full history** before the window is cut, so they are defined from the first day of the window.

### Sector aggregation (dollar-volume weighted)
Sector series are weighted averages of their stocks, with **dollar-volume weights** within each (Date, Sector):
$$w_{i,t} = \frac{P_{i,t} V_{i,t}}{\sum_{j \in S_t} P_{j,t} V_{j,t}}, \qquad R_{S,t} = \sum_{i \in S_t} w_{i,t}\, r_{i,t}$$
where $P$ is `Close` and $V$ is `Volume` (both split-adjusted, so $P \cdot V$ is the money traded), and $S_t$ is the set of the sector's stocks that **have a value that day**. Weights therefore always sum to 1, and this is checked in the notebook. Dollar volume reflects economic importance and liquidity when market capitalisation is not available; share volume alone would let a cheap stock with many shares traded outweigh a large company. Sector volatility is aggregated the same way: `SectorVol20 = Σ WeightedVol20`.

---

## Chapter 1: Sector–sector structure

*Question: how can investors truly diversify across sectors in the short and long term?*

**Notebook:** `chapter1_sector_diversification.ipynb` · **Helpers:** `src/chapters/chapter1.py` · **Input:** `stocks_sect.parquet` · **Outputs:** `website/content/chapter1.json` (data behind the website charts) and `website/figures/chapter1/` (static versions of the charts)

### 1.0 Sector series and time windows
- One daily series per sector from `stocks_sect.parquet`: the **return** is the dollar-volume weighted average of its stocks' daily returns, the **volatility** is the dollar-volume weighted average of its stocks' 60-day volatility.
- Three horizons separate long-run structure from recent behaviour:
  - **Full sample**: from the first day where every sector has at least **5 stocks**, so each sector series averages several companies;
  - **20Y**: 2000-04-01 → 2020-04-01;
  - **5Y**: 2015-04-01 → 2020-04-01.

### 1.1 Return correlation (Pearson)
- Sector–sector **Pearson correlation** matrices of daily sector returns, for each window.
- **Minimum overlap of 2 years (504 trading days)** per pair, so that unstable estimates are not reported.
- Each sector is summarised by its **average correlation with the other 10 sectors** (lower = more independent = better diversifier).

**Findings**
- Full sample: correlations are mostly **0.15–0.40**, so sectors are never independent.
- **Diversification is polarizing, not disappearing.** The *average* pair correlation barely changes (0.27 → 0.25 → 0.24), but its spread widens:
  - **the cyclical core tightens:** Financials–Consumer Cyclical goes 0.21 → 0.58, Financials–Energy 0.37 → 0.56, Financials–Technology 0.35 → 0.52;
  - **the defensives decouple:** Healthcare–Utilities goes 0.17 → 0.02.
- **Least connected in every window:** Healthcare, Industrials, Real Estate, Utilities. Healthcare's average correlation falls from 0.22 to 0.12.
- **Consumer Cyclical** was the most independent sector for decades (0.17, then 0.13) but not in the last 5Y (0.32).
- **Always among the most connected:** Basic Materials, Energy, Financial Services, Technology. Financial Services becomes the most connected sector in 2015-2020 (average 0.40).

### 1.2 Volatility correlation (do sectors become risky together?)
- Correlation of the sectors' **60-day volatility** series, per window.
- Return correlation tells you whether sectors **move in the same direction**. Volatility correlation tells you whether they **become risky at the same time**, which is the systemic-risk view.
- **Findings:**
  - volatility correlations are *lower* than return correlations on average (full sample 0.22 vs 0.27), but more extreme for a few pairs: Energy–Basic Materials is 0.79 / 0.75 / 0.68.
  - **Consumer Cyclical, Industrials and Real Estate** have the most independent volatility in every window.
  - In the last 5Y, Financials' risk is tied to Energy (0.59) and Technology (0.54, up from 0.21).
  - Caveat: average volatility correlation drops to 0.09 in the 5Y window, partly because 5 years hold few volatility cycles, so sectors that look independent only in the 5Y window are weak evidence.


### 1.3 Charts (website and `website/figures/chapter1/`)
- **Which sectors diversify?** A dot plot of each sector's average correlation with the other sectors in the 3 windows (returns and volatility).
- **Every pair of sectors:** clustered lower-triangle heatmaps for the 3 windows plus a "5Y minus Full" change view. Sectors are ordered by average-linkage hierarchical clustering on the full-sample correlation distance $\sqrt{2(1-\rho)}$, so sectors that move alike form blocks.
- The notebook writes `website/content/chapter1.json`, which `website/chapter1.js` draws as interactive SVG charts, and saves the static versions to `website/figures/chapter1/`. Its last section prints the numbers quoted in the website text, so the text can be updated after a new run.

> The findings above come from the published run. Numbers will move slightly after the full rerun (new cleaning, dollar-volume weights, verified sectors).

---

## Chapter 2: Market regimes and sector behaviour

*Question: how do sector returns, volatility and correlations change across market regimes?*

**Notebook:** `chapter2_market_regimes.ipynb` · **Helpers:** `src/chapters/chapter2.py` · **Inputs:** `stocks_sect.parquet`, `etfs_clean.parquet` (SPY) · **Outputs:** the 15 website figures in `website/figures/chapter2/` (dark transparent versions for the site; the notebook shows light versions)

### 2.1 Regime definition (rule-based, using SPY drawdowns)
1. **Drawdown** of SPY's adjusted price: $DD_t = P_t / \max_{s\le t} P_s - 1$.
2. Labels:
   - **Bear:** $DD_t \le -20\%$ (the standard "bear market" convention) and falling: from the peak down to the bottom of that decline,
   - **Recovery:** $DD_t \le -20\%$ but after the bottom, while the market rebounds,
   - **Neutral:** $-20\% < DD_t \le -5\%$,
   - **Bull:** $DD_t > -5\%$.
3. **Study period:** from the first day where every sector has at least 5 stocks (same rule as Chapter 1) and SPY has prices, to 2020-04-01.

Why drawdown? It is **transparent, interpretable, and has no fitted parameters**, so it is easy to explain to investors. Bear and Recovery are split because a drawdown level alone mixes two opposite markets: about two thirds of the days 20 % below the peak come after the bottom, while the market rebounds strongly, which made the average "Bear" return positive in the first version. Finding the bottom uses hindsight, which is fine to describe the past but not as a trading signal (Chapter 6 uses a different rule).

### 2.2 Sector series
One daily series per sector from `stocks_sect.parquet`, with dollar-volume weights: the **return** is the weighted average of its stocks' daily returns, the **volatility** the weighted average of their 20-day volatility.

### 2.3 Website figures
- **Static view (no regime):**
  - sector correlation heatmap over the whole period;
  - **60-day rolling average correlation** (mean of the off-diagonal correlations): low in calm periods, very high in crises, so *systemic risk is mainly a correlation phenomenon*;
  - average sector volatility per regime (bar chart).
- **For each regime (Bear / Recovery / Neutral / Bull):**
  - ranked average daily return per sector;
  - correlation heatmap computed on that regime's days only, reordered by **average-linkage** hierarchical clustering on $d = 1 - \rho$;
  - the 8 most volatile sectors (average 20-day volatility).
- The notebook's last section prints the rankings and levels quoted in the website text, so the text can be updated after a new run.

**Findings (published run; numbers will move after the full rerun)**
- *Volatility:* Bear volatility is **≈ 2× Bull**, and Neutral sits in between. In Bear markets the most volatile sectors are Financial Services and Real Estate (leverage/credit sensitive), then Materials, Consumer Cyclical and Energy; in Bull markets volatility becomes more sector-specific (Healthcare, Materials, Energy).
- *Returns:* in Bear, Consumer Cyclical and Communication Services were the strongest and the defensives slightly negative (likely the rebound days above); in Neutral, Healthcare leads; in Bull, all sectors are positive, led by Healthcare, Technology and Communication Services.
- *Correlations:* Bear is one big highly correlated block ("everything sells together"); Neutral shows sub-blocks (defensives vs cyclicals/financials); Bull has moderate correlation and broad participation.

**Conclusion:** regimes change **both how much risk there is and which sectors carry it**. There is no universally "best" sector, so allocation needs to be **regime-aware**.

---

## Chapter 3: Market categories within sectors

*Question: once the sector is fixed, does choosing a NASDAQ market category change risk and return?*

**Notebook:** `chapter3_market_categories.ipynb` · **Helpers:** `src/chapters/chapter3.py` · **Inputs:** `stocks_sect.parquet`, `metadata_clean.parquet` · **Output:** the interactive website map `website/figures/chapter3/sector_category_map.html`

### Market categories (NASDAQ listing tiers)
- **Q: Global Select Market**: the strictest listing standards, mostly large, established companies.
- **G: Global Market**: mid-tier companies.
- **S: Capital Market**: smaller, typically younger, less liquid companies.

The tiers act as a **size/maturity proxy**, and only NASDAQ-listed stocks have one. The chapter tests whether that proxy still matters after controlling for sector. The study period starts in **July 2006**, when NASDAQ's current three-tier structure began.

### Dynamic return–volatility map
- For every **Sector × Tier** cell and day: dollar-volume weighted average of the stocks' daily returns and of their 20-day volatility (at least **5 stocks** per cell), then a **60-day rolling mean** to see trends rather than daily noise.
- **Animated Plotly scatter:** x = smoothed volatility, y = smoothed return, colour = sector, marker = tier, one frame per week, with fixed axis ranges so frames are comparable. plotly.js is loaded from its CDN, which keeps the file at a few MB.
- **Reading it:** when the tiers of a sector **cluster together**, the sector dominates and the tier choice adds little. When they **separate**, the tier tilt materially changes risk/return.

### Takeaways
- Tier effects are **not constant**: distinctions **collapse in stress** (sector and market factors dominate) and **re-emerge in calmer periods**.
- Tier-based allocation only makes sense **together with sector structure and the market regime**.

---

## Chapter 4: Sector ETF vs DIY basket

*Question: should you buy the sector ETF or replicate the sector yourself?*

**Notebook:** `chapter4_etf_vs_basket.ipynb` · **Helpers:** `src/chapters/chapter4.py` · **Inputs:** `stocks_sect.parquet`, `etfs_clean.parquet` · **Outputs:** the 22 website figures in `website/figures/chapter4/` (performance and volatility per sector). The per-sector interpretations on the website are hand-written in `website/content/chapter4_writeups.json`.

### DIY basket construction
1. **Universe:** the sector's stocks that have a return on day $t$ (no survivorship filter: stocks that list or delist during the period are included while they trade).
2. **Weights:** dollar volume $P_{i,t-1} V_{i,t-1}$ (`Close × Volume`), **lagged by one day** so that $w_{t-1}$ weights $r_t$. This **avoids look-ahead bias**, because today's volume is not known before today's return is realised.
3. **Cap at 10 %** per stock: the excess is spread over the other stocks, repeated until no weight is above the cap (on days with fewer than 10 stocks, equal weights). This limits concentration in a few heavily traded names, similar to how capped indices work.
4. Basket return $R^{B}_t = \sum_i w_{i,t-1} r_{i,t}$ with **simple** daily returns (a portfolio's return is the weighted sum of simple returns, not of log returns).
5. The ETF's return comes from its `Adj Close` (dividends included, like the stocks). ETF and basket are **aligned on identical trading dates**, from the ETF's first day to 2020-04-01.

### Website figures
- **Performance:** both series rebased to **100** on the same start date (comparing growth paths, not prices).
- **Rolling annualised 60-day volatility:** ETF vs basket.

### Findings (published run; numbers will move after the full rerun)
- The **DIY basket tended to outperform** the ETF over the sample. **Real Estate was the exception**: there the ETF did better.
- **ETFs are generally less volatile**, thanks to their broader, market-cap weighted and capped constituent set.
- Rule of thumb: if volatility spikes while ETF and basket drift apart, treat it as a regime shift and prefer the ETF until they track again.

---

## Chapter 5: Sector proxy, or is one stock enough?

*Question: can a single stock represent a sector, or do you need a basket?*

**Notebook:** `chapter5_sector_proxy.ipynb` · **Helpers:** `src/chapters/chapter5.py` · **Input:** `stocks_sect.parquet` · **Outputs:** the 33 interactive website figures in `website/figures/chapter5/` (UMAP map, index comparison and tracking error per sector)

### Pipeline (per sector, 2005-01-01 → 2020-04-01)
1. **Sector proxy:** $R_{S,t} = \sum_i w_{i,t} r_{i,t}$, the dollar-volume weighted sector return from `stocks_sect` (same series as the other chapters).
2. **Liquidity filter:** keep stocks that trade on **≥ 80 % of their observed days**, then those whose **average dollar volume is above the sector's 20th percentile**. Only *observed* days count, so recent IPOs are not penalised.
3. **Representativeness score:** the **mean rolling 90-day correlation** (min 60 obs.) between each stock and the sector proxy.
4. **Lifetime-aware ranking:**
   $$\text{score}_i = \text{rep\_score}_i \times \left(\text{lifetime}_i\right)^{\alpha}, \quad \alpha = 2$$
   where `lifetime` = share of the period on which the stock has a return. This **penalises short histories**: a recent stock can look highly correlated over a short, favourable window, but it cannot represent the sector over 15 years. Ties are broken by **lower volatility**.
5. **Selection:** the **single best stock** (rank 1) and an **equal-weighted Top-5 basket** (weights renormalised over the stocks with a return each day).
6. **Validation:**
   - **Normalised index comparison:** sector proxy vs single stock vs Top-5 basket, each rebased to 100.
   - **Tracking error** $\text{std}(r_t - R_{S,t}) \cdot \sqrt{252}$, over the whole period and on rolling 60- and 120-day windows, single vs basket.
7. **Intra-sector structure (unsupervised ML):**
   - Stock–stock **pairwise-overlap correlation** (each pair uses all days where both stocks trade, at least **one year** in common).
   - **Angular distance** $d = \sqrt{2(1-\rho)}$; missing correlations get the maximum distance (2) so unknown pairs are never pulled together. Stocks missing more than 20 % of their correlations are left off the map (the notebook reports how many).
   - **UMAP** (`metric="precomputed"`, `n_neighbors=10`, `min_dist=0.01`) → 2-D map, then **HDBSCAN** (scikit-learn, `min_cluster_size=5`) on the map coordinates; clusters are named by size, unassigned points are "Outlier".
8. Interactive Plotly figures saved as HTML (plotly.js from its CDN) and embedded via iframes on the website. The notebook's last table holds the numbers quoted in each sector's write-up.

Caveat: the representative and the basket are selected and evaluated on the same period, so their fit is an in-sample best case.

### Findings (published run; numbers will move after the full rerun)
- **Energy** (and almost Communication Services) is dominated by a common factor, so **one well-chosen stock is enough**.
- **Technology, Healthcare, Financial Services, Industrials, Consumer Defensive, Utilities, Basic Materials:** a single stock captures the direction but has large tracking error. A **Top-5 basket** clearly improves correlation and stability (e.g. Basic Materials: single AA ρ ≈ 0.68 → basket ρ ≈ 0.80. Consumer Defensive: PG ρ ≈ 0.60 → basket ρ ≈ 0.79).
- **Consumer Cyclical** is strongly **regime-dependent**: the basket helps but cannot remove large deviations.
- **Structural cohesion in UMAP does not guarantee good tracking.** Some sectors look concentrated in UMAP space but still show TE spikes. **Rolling tracking error is the decisive validation metric.**
- A small, behaviourally selected basket captures most of the diversification benefit at a fraction of the complexity of full replication.

---

## Chapter 6: Systematic sector strategy

**Code:** `src/strategy/` (`sector_rotation.py`, `profiles.py`, `data.py`, `plots.py`) · **Notebook:** `chapter6_sector_strategy.ipynb` · **Tests:** `tests/test_strategy.py` · **Outputs:** the 4 website figures in `website/figures/chapter6/` and the Strategy Picker data `website/content/strategy_stats.json`

### Why the original Strategy Builder was replaced
The first version (in the project's original results notebook) was a **lookup table** of historical return/volatility for each configuration (sector × market category × ETF/all/top-5 × regime = 528 rows), each sorted into one of 9 profiles. It was not a strategy, because it had no rule for *when* to hold *what*. It also had these bugs:

1. **Volume-weighted *price* index** $\sum P\cdot V/\sum V$: a $500 stock joining a basket of $20 stocks shows up as a +2 400 % "return". Median annualised volatility in `strategy_stats.json` is **508 %** for "all" baskets and **276 %** for "top5", against 22 % for ETFs.
2. **All 396 regime rows are empty** (`n_days = 0`, NaN metrics), so the bull/neutral/bear dropdown never showed real numbers.
3. **NaN → "gambler":** `NaN < threshold` is `False`, so empty rows fell into the high-return/high-volatility bucket. That is why **412 of 528** rows were "gambler".
4. **"Top 5" chosen on the last day** of the data: today's most traded stocks, held back in time (look-ahead).

`src/strategy/profiles.py` fixes them, and Part 5 of the notebook regenerates the Strategy Picker data (`picker_table`):
- basket returns $= \sum_i w_{i,t-1} r_{i,t}$ with dollar-volume weights from the previous day, renormalised over the stocks trading that day, so listings change the weights and never the index level;
- **top5:** each day, the 5 stocks that traded the most money the day before;
- every row covers the same period (1999 → 2020-04-01) so rows are comparable; the picker shows the whole period (the unused per-regime rows were dropped);
- NaN → "insufficient data";
- one row per **sector × type** (all / top5 / ETF), without the NASDAQ-tier split: the tiers are those of 2020, and a stock in the smallest tier in 2020 is often a company that shrank, so a tier split would pick winners and losers after the fact.

The website's Strategy Picker reads the regenerated `website/content/strategy_stats.json`.

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
- walk-forward re-estimation of the parameters.

---

## Visualisation approach

- **Diagnostics first:** time-series plots of missing/zero counts and start-date histograms drove the cleaning decisions.
- **Heatmaps** (custom diverging colour maps, fixed range −1 to 1) for correlation matrices. **Clustered heatmaps** reorder rows by dendrogram leaves so that blocks become visible.
- **Dendrograms** truncated to the last p merges for readability, with a shared y-scale when comparing regimes.
- **Dot plot** (website) of each sector's average correlation with the other sectors, one dot per window.
- **Clustered correlation heatmaps** (Ch. 1 and 2), ordered by hierarchical clustering so related sectors form blocks.
- **Ranked bar charts** of sector returns and volatility per regime (Ch. 2).
- **Animated Plotly scatter** for the Sector × Category return–volatility map (Ch. 3).
- **Interactive Plotly** UMAP maps, rebased index comparisons and rolling tracking-error charts, exported to HTML and embedded via iframes (Ch. 5).
- A consistent **dark theme** across the matplotlib and Plotly outputs to match the website: matplotlib figures are drawn with the shared themes in `src/utils/figures.py` (light in the notebooks, dark transparent for the site).

---

## Design decisions, limitations & possible improvements

**Key design decisions (and why)**
- **Simple daily returns on Adj Close:** corrected for splits and dividends, and they aggregate exactly into sector and basket returns.
- **Pairwise-overlap correlations** instead of a common date window: they keep decades of history and all stocks (IPOs from the 1980s to 2015+) instead of truncating to the shortest series. The cost is that the matrix may not be positive semi-definite, which is acceptable for distances and visual maps.
- **Dollar-volume weighting** of sector series and baskets (no market-cap data in the dataset): `Close × Volume` is a liquidity/importance proxy. In the Chapter 4 basket the weights are **lagged** by one day and **capped** at 10 %.
- **Rule-based regimes** for transparency. Alternatives would be Hidden Markov Models, Markov-switching models or clustering on volatility/return features.
- **Unsupervised ML (UMAP + HDBSCAN)** in Chapter 5, because there are no labels for the sub-groups of stocks inside a sector.

**Known limitations**
- **Survivorship bias:** the dataset only contains symbols that were still listed in 2020, and broken symbols were removed during cleaning.
- **Sample ends 2020-04-01**, so it includes only the start of the COVID crash.
- **Sector labels come from today's yfinance classification** and are applied backwards in time (companies can change sector). Tickers whose Yahoo company name no longer matches the 2020 name are left without a sector, but a company that changed sector under the same name is not detected.
- **Bear vs Recovery uses hindsight:** the bottom of a decline is only known afterwards, so these regimes describe the past and cannot be used as a live signal.
- **Website Strategy Picker data is still the old version until the full rerun:** it has the bugs described in Chapter 6. Part 5 of `chapter6_sector_strategy.ipynb` regenerates `website/content/strategy_stats.json` with the corrected `src/strategy/profiles.py`.
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
├── preparing_sectors.ipynb           # stock-level sector table (returns, rolling vols, dollar-volume weights)
├── src/utils/
│   ├── loading_data.py               # raw CSV + NASDAQ metadata loaders (no cleaning at load)
│   ├── data_inspection.py            # date / missing / zero diagnostics
│   ├── data_inspection_viz.py        # diagnostic plots over time
│   ├── data_cleaning.py              # symbol removal, Open repair, volume trimming/imputation
│   ├── data_export.py                # parquet export of the cleaned prices
│   ├── sector_panel.py               # sector table: returns, rolling volatility, dollar-volume weights, sector series
│   ├── figures.py                    # shared light/dark themes; every figure goes to website/figures/<section>/
│   └── metadata_enrich.py            # NASDAQ directory enrichment, code labels, cached yfinance sector/industry fetch
├── chapter1_sector_diversification.ipynb  # (Ch. 1) sector correlations, diversifiers, website data
├── chapter2_market_regimes.ipynb     # (Ch. 2) SPY regimes, sector returns / volatility / correlations by regime
├── src/chapters/chapter2.py          # (Ch. 2) regimes, rolling correlation, clustering, figures
├── chapter3_market_categories.ipynb  # (Ch. 3) Sector × NASDAQ tier risk–return map
├── src/chapters/chapter3.py          # (Ch. 3) tier series, smoothing, animated Plotly map
├── chapter4_etf_vs_basket.ipynb      # (Ch. 4) sector ETF vs dollar-volume weighted DIY basket
├── src/chapters/chapter4.py          # (Ch. 4) capped lagged weights, basket and ETF returns, figures
├── chapter5_sector_proxy.ipynb       # (Ch. 5) sector proxy vs single stock / Top-5 basket, UMAP + HDBSCAN map
├── src/chapters/chapter5.py          # (Ch. 5) liquidity filter, ranking, tracking error, UMAP/HDBSCAN, figures
├── chapter6_sector_strategy.ipynb    # (Ch. 6) systematic strategy (design/validation/holdout, ablation, robustness) + Strategy Picker data
├── src/strategy/
│   ├── data.py                       # ETF/SPY/T-bill download + cache, study periods
│   ├── sector_rotation.py            # regimes, signals, backtest engine, metrics, block bootstrap
│   ├── plots.py                      # Chapter 6 figures (shared light/dark themes)
│   └── profiles.py                   # Strategy Picker table: return-compounded baskets, top-5 without look-ahead, profiles
├── tests/test_strategy.py            # no-look-ahead, costs, execution lag, builder bug regressions
├── src/chapters/chapter1.py          # (Ch. 1) windows, correlations, diversifiers, website JSON, figures
├── website/                          # static site (HTML/CSS/JS, JSON content, embedded Plotly HTML)
│   └── figures/                      # every figure produced by the notebooks, one subfolder per chapter (the site reads them here)
└── .github/workflows/pages.yml       # GitHub Pages deployment of website/
```

---

## How to run the code

Everything is run **from the project root**, in this order. Each notebook only reads files written by the steps before it.

### 1. Clone the repository
```console
git clone https://github.com/chedliabd/cross-exchange-asset-analysis.git
cd cross-exchange-asset-analysis
```

### 2. Create a virtual environment and install the requirements
Python 3.11+ is required.
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Select the `.venv` interpreter as the notebook kernel.

### 3. Download the raw data into `data/`
Needs a Kaggle API token in `~/.kaggle/kaggle.json`.
```console
kaggle datasets download -d jacksoncrow/stock-market-dataset -p data --unzip
curl -L -A "Mozilla/5.0" -o data/nasdaqlisted.txt https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt
```
This gives `data/stocks/`, `data/etfs/` (SPY included), `data/symbols_valid_meta.csv` and `data/nasdaqlisted.txt`.

### 4. Prepare the data (writes `data_processed/`)
1. `preparing_stocks.ipynb` → `stocks_clean.parquet`
2. `preparing_etfs.ipynb` → `etfs_clean.parquet`
3. `preparing_metadata.ipynb` → `metadata_clean.parquet` (the Yahoo Finance sector lookup takes about 30 minutes; it is cached in `data/yfinance_profiles.csv` and resumes if interrupted)
4. `preparing_sectors.ipynb` → `stocks_sect.parquet`

### 5. Run the chapters (writes `website/figures/` and `website/content/`)
1. `chapter1_sector_diversification.ipynb`
2. `chapter2_market_regimes.ipynb`
3. `chapter3_market_categories.ipynb`
4. `chapter4_etf_vs_basket.ipynb`
5. `chapter5_sector_proxy.ipynb`
6. `chapter6_sector_strategy.ipynb`: first run the tests with `python -m pytest tests`. Parts 1-4 download ETF prices from Yahoo Finance up to today (cached in `data/strategy/prices.csv`); Part 5 writes the Strategy Picker data.

### 6. Look at the website locally
```console
python -m http.server 8000 --directory website
```
then open http://localhost:8000. Pushing to `main` deploys `website/` to GitHub Pages.

---

## Tech stack

**Python:** pandas, NumPy, SciPy (hierarchical clustering), scikit-learn (HDBSCAN), umap-learn, yfinance, matplotlib, seaborn, Plotly, tqdm, PyArrow/Parquet, pytest.
**Web:** HTML/CSS/vanilla JavaScript, embedded Plotly HTML, JSON content files, GitHub Actions → GitHub Pages.
