"""Regime-aware sector rotation: the Chapter 6 systematic strategy.

Every building block comes from a finding earlier in the project:

* **Selection: 12-1 month momentum** (Ch. 1/3). Sector return dispersion is
  persistent over months, so we hold the `top_k` sectors with the strongest
  trailing return, skipping the most recent month (short-term reversal).
* **Regime switch: SPY drawdown** (Ch. 2). Same thresholds as the regime analysis
  (Bull > -5 %, Neutral -5..-20 %, Bear <= -20 %). Ch. 2 showed that the defensive,
  weakly correlated sectors hold up best in Neutral/Bear markets. Instead of
  hard-coding "Healthcare / Staples / Utilities" (which would be look-ahead from
  our own full-sample results), we pick the `defensive_k` sectors with the lowest
  trailing 1-year beta to SPY at each rebalance.
* **Sizing: inverse volatility + volatility targeting** (Ch. 1/2). Correlations and
  volatility co-move and spike in stress, so diversification fails exactly when it
  is needed. Scaling the book to a target ex-ante volatility (using the trailing
  covariance matrix, so correlation spikes are captured) cuts exposure in those
  periods. No leverage: exposure is capped at 100 %, the rest earns the T-bill rate.
* **Implementation: sector ETFs** (Ch. 4/5). They are cheap and liquid and had
  lower volatility than the stock baskets.

Backtest hygiene: signals use data up to the close of the rebalance day and trades
execute at the next close (`exec_lag=1`). Weights drift between monthly rebalances,
and every trade pays `cost_bps` on traded notional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REGIMES = ("bull", "neutral", "bear")


# --------------------------------------------------------------------------- #
# Regimes                                                                      #
# --------------------------------------------------------------------------- #
def drawdown(prices: pd.Series) -> pd.Series:
    """Drawdown from the running peak (causal: uses only past prices)."""
    return prices / prices.cummax() - 1.0


def drawdown_regime(
    spy: pd.Series, neutral: float = -0.05, bear: float = -0.20, window: int | None = None
) -> pd.Series:
    """Chapter 2 regime labels: bull / neutral / bear from the SPY drawdown.

    `window=None` measures the drawdown from the all-time high (the Ch. 2 definition).
    A finite window (e.g. 252) measures it from the trailing 1-year high instead.
    """
    s = spy.dropna()
    peak = s.cummax() if window is None else s.rolling(window, min_periods=1).max()
    dd = s / peak - 1.0
    regime = pd.Series("bull", index=dd.index, name="regime")
    regime[dd <= neutral] = "neutral"
    regime[dd <= bear] = "bear"
    return regime


def trend_regime(spy: pd.Series, window: int = 200) -> pd.Series:
    """Two-state trend filter: bull above the 200-day moving average, bear below it."""
    s = spy.dropna()
    sma = s.rolling(window, min_periods=window).mean()
    regime = pd.Series("bull", index=s.index, name="regime")
    regime[s < sma] = "bear"
    return regime


def make_regime(spy: pd.Series, method: str) -> pd.Series:
    if method == "drawdown_ath":
        return drawdown_regime(spy)
    if method == "drawdown_1y":
        return drawdown_regime(spy, window=TRADING_DAYS)
    if method == "trend_200d":
        return trend_regime(spy)
    raise ValueError(f"unknown regime method {method!r}")


# --------------------------------------------------------------------------- #
# Strategy configuration and signals                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyConfig:
    momentum_lookback: int = 252   # 12 months
    momentum_skip: int = 21        # skip the last month (short-term reversal)
    top_k: int = 4                 # sectors held by the momentum sleeve
    use_momentum: bool = False     # True -> attack sleeve = top_k by 12-1 momentum (rejected on design data)
    use_regime: bool = True        # Ch. 2 regime switch into defensive sectors
    regime_method: str = "trend_200d"  # "trend_200d" (selected), "drawdown_ath" (Ch. 2), "drawdown_1y"
    defensive_k: int = 3           # sectors in the defensive sleeve
    beta_window: int = 252
    weighting: str = "inverse_vol"  # "inverse_vol" or "equal"
    vol_window: int = 63           # ~3 months for vol and covariance estimates
    vol_target: float | None = 0.12  # annualised; None disables vol targeting
    max_gross: float = 1.0         # no leverage
    cost_bps: float = 10.0         # per unit of traded notional (one way)
    exec_lag: int = 1              # trade N days after the signal close
    neutral_defensive_share: float = 0.5  # Neutral regime = 50/50 momentum/defensive


@dataclass
class Signals:
    momentum: pd.DataFrame
    vol: pd.DataFrame
    beta: pd.DataFrame
    returns: pd.DataFrame
    regime: pd.Series


def compute_signals(prices: pd.DataFrame, spy: pd.Series, cfg: StrategyConfig) -> Signals:
    """All signals are computed with trailing windows, so the value at date t only uses data <= t."""
    rets = prices.pct_change(fill_method=None)
    spy_ret = spy.pct_change(fill_method=None)

    momentum = prices.shift(cfg.momentum_skip) / prices.shift(cfg.momentum_lookback) - 1.0
    vol = rets.rolling(cfg.vol_window, min_periods=cfg.vol_window).std() * np.sqrt(TRADING_DAYS)
    cov = rets.rolling(cfg.beta_window, min_periods=cfg.beta_window).cov(spy_ret)
    beta = cov.div(spy_ret.rolling(cfg.beta_window, min_periods=cfg.beta_window).var(), axis=0)
    return Signals(momentum, vol, beta, rets, make_regime(spy, cfg.regime_method))


def _weights(names: list[str], vol: pd.Series, scheme: str) -> pd.Series:
    if not names:
        return pd.Series(dtype=float)
    if scheme == "equal":
        w = pd.Series(1.0, index=names)
    elif scheme == "inverse_vol":
        w = 1.0 / vol[names]
    else:
        raise ValueError(f"unknown weighting {scheme!r}")
    return w / w.sum()


def target_weights(date: pd.Timestamp, sig: Signals, cfg: StrategyConfig) -> pd.Series:
    """Target portfolio (fraction of NAV per ETF) using information up to `date`'s close."""
    mom, vol, beta = sig.momentum.loc[date], sig.vol.loc[date], sig.beta.loc[date]
    eligible = mom.index[mom.notna() & vol.notna() & (vol > 0)]
    if len(eligible) == 0:
        return pd.Series(0.0, index=sig.momentum.columns)

    if cfg.use_momentum:
        attack = list(mom[eligible].nlargest(cfg.top_k).index)
    else:
        attack = list(eligible)
    w = _weights(attack, vol, cfg.weighting)

    if cfg.use_regime:
        regime = sig.regime.get(date, "bull")
        b = beta[eligible].dropna()
        defensive = list(b.nsmallest(cfg.defensive_k).index)
        w_def = _weights(defensive, vol, cfg.weighting)
        share = {"bull": 0.0, "neutral": cfg.neutral_defensive_share, "bear": 1.0}[regime]
        if share > 0 and len(w_def):
            w = w.mul(1 - share).add(w_def.mul(share), fill_value=0.0)

    if cfg.vol_target is not None and len(w):
        window = sig.returns.loc[:date, w.index].tail(cfg.vol_window)
        cov = window.cov().values * TRADING_DAYS
        port_vol = float(np.sqrt(w.values @ cov @ w.values))
        if np.isfinite(port_vol) and port_vol > 0:
            w = w * min(cfg.max_gross, cfg.vol_target / port_vol)

    return w.reindex(sig.momentum.columns, fill_value=0.0)


# --------------------------------------------------------------------------- #
# Backtest engine                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    returns: pd.Series            # daily net returns
    weights: pd.DataFrame         # end-of-day weights (after drift and trades)
    turnover: pd.Series           # traded notional per day (sum |dw|)
    regime: pd.Series
    config: StrategyConfig
    targets: dict = field(default_factory=dict)  # signal date -> target weights

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def exposure(self) -> pd.Series:
        return self.weights.sum(axis=1)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def backtest(
    prices: pd.DataFrame,
    spy: pd.Series,
    rf_daily: pd.Series,
    cfg: StrategyConfig = StrategyConfig(),
    start: str | None = None,
    end: str | None = None,
    signals: Signals | None = None,
) -> BacktestResult:
    """Simulate the strategy day by day with drifting weights, lagged execution and costs.

    Signals are computed on the full history (they are trailing, so this is causal);
    `start` only controls when trading begins, which keeps the warm-up period available.
    """
    sig = signals or compute_signals(prices, spy, cfg)
    dates = prices.loc[start:end].index
    if len(dates) == 0:
        raise ValueError("empty backtest window")
    R = sig.returns.reindex(dates).fillna(0.0).to_numpy()
    rf = rf_daily.reindex(dates).ffill().fillna(0.0).to_numpy()
    rebal = set(month_end_dates(dates)) | {dates[0]}  # invest from the first day of the window

    n = prices.shape[1]
    w = np.zeros(n)
    cash = 1.0
    pending: dict[int, np.ndarray] = {}
    targets: dict = {}
    out_r = np.zeros(len(dates))
    out_w = np.zeros((len(dates), n))
    out_to = np.zeros(len(dates))

    for d, date in enumerate(dates):
        # 1) earn today's return on yesterday's holdings, then let the weights drift
        gross = 1.0 + w @ R[d] + cash * rf[d]
        w = w * (1.0 + R[d]) / gross
        cash = cash * (1.0 + rf[d]) / gross
        ret = gross - 1.0

        # 2) on a rebalance date, form the target from today's close and schedule the trade
        if date in rebal:
            tgt = target_weights(date, sig, cfg).to_numpy()
            targets[date] = tgt
            pending[d + cfg.exec_lag] = tgt

        # 3) execute a trade scheduled for today's close (exec_lag=0 trades on the signal close)
        if d in pending:
            tgt = pending.pop(d)
            traded = float(np.abs(tgt - w).sum())
            ret = (1.0 + ret) * (1.0 - traded * cfg.cost_bps / 1e4) - 1.0
            w, cash = tgt.copy(), 1.0 - tgt.sum()
            out_to[d] = traded

        out_r[d], out_w[d] = ret, w

    cols = prices.columns
    return BacktestResult(
        returns=pd.Series(out_r, index=dates, name="strategy"),
        weights=pd.DataFrame(out_w, index=dates, columns=cols),
        turnover=pd.Series(out_to, index=dates, name="turnover"),
        regime=sig.regime.reindex(dates),
        config=cfg,
        targets={k: pd.Series(v, index=cols) for k, v in targets.items()},
    )


# Pre-defined variants used for the ablation study (each adds one idea)
FINAL = StrategyConfig()

# Ablation ladder: each row adds one idea on top of the previous one
ABLATIONS: dict[str, StrategyConfig] = {
    "EW sectors": StrategyConfig(use_regime=False, weighting="equal", vol_target=None),
    "+ inverse-vol": StrategyConfig(use_regime=False, vol_target=None),
    "+ trend regime switch": StrategyConfig(vol_target=None),
    "+ vol target (final)": FINAL,
}

# Alternatives examined on the design period and rejected (reported for transparency)
REJECTED: dict[str, StrategyConfig] = {
    "v1: momentum + Ch2 drawdown regime": StrategyConfig(use_momentum=True, regime_method="drawdown_ath"),
    "momentum + 1y-high drawdown regime": StrategyConfig(use_momentum=True, regime_method="drawdown_1y"),
    "momentum + trend regime": StrategyConfig(use_momentum=True),
}


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return float((equity / equity.cummax() - 1).min())


def performance_metrics(
    returns: pd.Series,
    rf_daily: pd.Series,
    benchmark: pd.Series | None = None,
    result: BacktestResult | None = None,
) -> dict:
    r = returns.dropna()
    rf = rf_daily.reindex(r.index).fillna(0.0)
    ex = r - rf
    years = len(r) / TRADING_DAYS
    cagr = float((1 + r).prod() ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS))
    mdd = max_drawdown(r)
    out = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)) if ex.std() > 0 else np.nan,
        "Sortino": float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Worst month": float((1 + r).resample("ME").prod().sub(1).min()),
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0.0)
        out["Beta vs SPY"] = float(np.cov(r, b)[0, 1] / b.var())
        active = r - b
        out["Tracking error"] = float(active.std() * np.sqrt(TRADING_DAYS))
    if result is not None:
        out["Turnover / yr"] = float(result.turnover.reindex(r.index).sum() / years)
        out["Avg exposure"] = float(result.exposure.reindex(r.index).mean())
    return out


def metrics_table(series: dict[str, pd.Series], rf_daily, benchmark, results=None) -> pd.DataFrame:
    results = results or {}
    return pd.DataFrame(
        {k: performance_metrics(v, rf_daily, benchmark, results.get(k)) for k, v in series.items()}
    ).T


def regime_breakdown(returns: dict[str, pd.Series], regime: pd.Series) -> pd.DataFrame:
    """Annualised mean return of each series within each regime (regime days are not contiguous,
    so we annualise the mean daily log return instead of using first/last values)."""
    rows = {}
    for name, r in returns.items():
        lr = np.log1p(r)
        reg = regime.reindex(r.index)
        rows[name] = {g: float(np.expm1(lr[reg == g].mean() * TRADING_DAYS)) for g in REGIMES}
    df = pd.DataFrame(rows).T
    first = next(iter(returns.values()))
    df.loc["share of days"] = [float((regime.reindex(first.index) == g).mean()) for g in REGIMES]
    return df


def stationary_bootstrap_sharpe_diff(
    r_a: pd.Series,
    r_b: pd.Series,
    rf_daily: pd.Series,
    n_boot: int = 2000,
    mean_block: int = 21,
    seed: int = 0,
) -> dict:
    """Politis-Romano stationary bootstrap of Sharpe(a) - Sharpe(b).

    Days are resampled in blocks of random (geometric) length so that the
    autocorrelation and volatility clustering of daily returns are preserved,
    the same issue flagged for the Chapter 2 Kruskal-Wallis test.
    """
    idx = r_a.index.intersection(r_b.index)
    rf = rf_daily.reindex(idx).fillna(0.0).to_numpy()
    a = r_a.reindex(idx).to_numpy() - rf
    b = r_b.reindex(idx).to_numpy() - rf
    T = len(idx)
    rng = np.random.default_rng(seed)

    def sharpe(x):
        return x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS)

    diffs = np.empty(n_boot)
    t_idx = np.arange(T)
    for k in range(n_boot):
        # a new block starts with probability 1/mean_block; inside a block, days run consecutively
        new_block = rng.random(T) < 1.0 / mean_block
        new_block[0] = True
        block_start = np.maximum.accumulate(np.where(new_block, t_idx, 0))
        origin = rng.integers(T, size=T)[block_start]
        pos = (origin + t_idx - block_start) % T
        diffs[k] = sharpe(a[pos]) - sharpe(b[pos])

    lo, hi = np.percentile(diffs, [5, 95])
    return {
        "sharpe_diff": float(sharpe(a) - sharpe(b)),
        "ci90_low": float(lo),
        "ci90_high": float(hi),
        "p_value_diff_le_0": float((diffs <= 0).mean()),
    }


def parameter_grid(
    prices: pd.DataFrame,
    spy: pd.Series,
    rf_daily: pd.Series,
    grid: dict[str, list],
    start: str | None,
    end: str | None,
    base: StrategyConfig = StrategyConfig(),
) -> pd.DataFrame:
    """Re-run the backtest for every combination in `grid` (robustness, not tuning)."""
    from itertools import product

    keys = list(grid)
    spy_ret = spy.pct_change(fill_method=None)
    rows = []
    for values in product(*(grid[k] for k in keys)):
        cfg = StrategyConfig(**{**base.__dict__, **dict(zip(keys, values))})
        res = backtest(prices, spy, rf_daily, cfg, start, end)
        m = performance_metrics(res.returns, rf_daily, spy_ret, res)
        rows.append({**dict(zip(keys, values)), **m})
    return pd.DataFrame(rows)
