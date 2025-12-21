
# An Investor’s Guide to Building a Smarter Portfolio

Data Story : put the link 

## Abstract

This project is built around one practical goal: help investors get the biggest market understanding and risk control with the least effort. Instead of giving abstract statistics, we build a step-by-step guide that answers real portfolio questions:

- Do sectors actually diversify each other, or is it fake diversification?
- Within a sector, is one representative stock enough, or do you need a basket?
- Should you buy the sector ETF, or is it worth recreating the exposure yourself?
- How do all of these answers change in bull vs bear markets?
- Given my risk tolerance, what is a sensible way to allocate today?


To do that, our analysis follows a logical pipeline: sector relationships, market regimes, sector representation (single stocks vs baskets), ETF versus DIY exposure, and finally investor-specific portfolio guidance. The website will be a portfolio decision toolkit. The objective is to make sector investing more transparent, avoid misleading shortcuts, and provide investors with a simple way to choose between market exposure, sector exposure, ETFs, or DIY baskets, depending on both their risk tolerance and the market environment.

## Research Questions

1. How can investors truly diversify their portfolio across sectors in the short term and the long term ?
2. Is investing in one stock enough to represent a sector, or do investors need to diversify among multiple stocks within the same sector to reduce risk?
3. Should an investor buy the sector ETF, or construct a sector basket manually to obtain similar exposure with better cost/precision?
4. When do bull/bear/neutral regimes occur, and which sectors perform best (or worst) inside each regime?
5. Given my risk tolerance and today’s market conditions, what exposure should I take: market-wide, sector rotation, ETFs, or a DIY basket?



## Additional Datasets

- **NASDAQ Metadata:** Market category enrichment for NASDAQ-traded symbols (Global Select, Global Market, Capital Market).  
- **Sector Classification Data:** Sector and industry tags for each ticker retrieved from Yahoo Finance.



## Data Cleaning and Preprocessing 

## Methods 

We combine financial econometrics, unsupervised learning, and risk analysis:

- Volume-weighted sector log returns
- Pearson and rolling correlations
- Rolling volatility
- Clustering
- PCA


## Project Structure & Analysis Pipeline

### Chapter 1 — Sector-Sector Structure
- Sector–sector correlation 
- Volatility correlation
- Rolling correlations 
- Sector clustering 

### Chapter 2 — Market Regimes
- Identification of bull, bear, and neutral regimes
- Analysis of sector behavior conditional on regimes

### Chapter 3 — Market Categories
- Comparison of sector behavior across market categories
- Cross-category correlation and volatility analysis

### Chapter 4 — ETF vs Sector Basket
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


## Repository Structure 
```console
ada-2025-project-thedatacollectors/
├── data/                     # Raw and processed datasets
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing/        # Data cleaning & alignment (we should add the preparing_*.ipynb files here i guess and make them py files)
│   ├── chapter1/             # Sector–sector analysis
│   ├── chapter2/             # Market regime analysis
│   ├── chapter3/             # Market category & sector behavior
│   ├── chapter4/             # Sector proxy (single stock vs basket)
│   ├── chapter5/             # ETF vs DIY sector basket
│   └── chapter6/             # Investor profile mapping
│
├── plots/                    # Folder for plots (will contain the plots generated by the scripts 
│                             # and used in the data story when generated by the results notebook)
├── results.ipynb             # Main analysis notebook
├── requirements.txt          # Required packages
├── README.md                 # Project description and instructions
└── results.ipynb             # Jupyter notebook with the results
├── .gitignore
└── README.md
```
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

Chedly: Worked on initial data cleaning and preprocessing; worked on Chapter 2 (Market Regimes), Chapter 3 (Market Categories) and contributed to README.

İrem: Contributed to data preprocessing and worked on Chapter 1, focusing on sector–sector relationships. Prepared the README.

Efe: Worked on Chapter 4 (ETF analysis) and contributed to data preprocessing and README.

Hugo: Worked on Chapter 5, analyzing sector proxy representativeness and contributed to data preprocessing and README.

Ozan: Worked on Chapter 6, developing the interactive investor profile mapping, and contributed to data preprocessing and README.
