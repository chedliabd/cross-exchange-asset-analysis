import numpy as np
import pandas as pd
import pytest

from src.strategy import profiles
from src.strategy.sector_rotation import (
    StrategyConfig,
    backtest,
    compute_signals,
    drawdown_regime,
    stationary_bootstrap_sharpe_diff,
    target_weights,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2000-01-03", periods=900)
    n = 6
    common = rng.normal(0.0003, 0.01, len(dates))
    rets = common[:, None] * rng.uniform(0.5, 1.5, n) + rng.normal(0.0002, 0.008, (len(dates), n))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=dates, columns=list("ABCDEF"))
    spy = pd.Series(100 * np.cumprod(1 + common), index=dates)
    rf = pd.Series(0.02 / 252, index=dates)
    return prices, spy, rf


def test_drawdown_regime_thresholds():
    spy = pd.Series([100, 96, 94, 85, 79, 100], index=pd.bdate_range("2020-01-01", periods=6))
    assert drawdown_regime(spy).tolist() == ["bull", "bull", "neutral", "neutral", "bear", "bull"]


def test_signals_do_not_look_ahead(market):
    prices, spy, rf = market
    cfg = StrategyConfig()
    date = prices.index[600]
    before = target_weights(date, compute_signals(prices, spy, cfg), cfg)

    shocked = prices.copy()
    shocked.loc[shocked.index > date] *= np.linspace(0.2, 5, shocked.shape[1])
    spy_shocked = spy.copy()
    spy_shocked[spy_shocked.index > date] *= 0.3
    after = target_weights(date, compute_signals(shocked, spy_shocked, cfg), cfg)
    pd.testing.assert_series_equal(before, after)


def test_weights_are_long_only_and_unlevered(market):
    prices, spy, rf = market
    res = backtest(prices, spy, rf, StrategyConfig(), start="2001-01-01")
    assert (res.weights >= -1e-12).all().all()
    assert (res.exposure <= 1 + 1e-9).all()
    mom = StrategyConfig(use_momentum=True, regime_method="drawdown_ath")
    res_mom = backtest(prices, spy, rf, mom, start="2001-01-01")
    for tgt in res_mom.targets.values():
        assert (tgt > 0).sum() <= mom.top_k + mom.defensive_k


def test_costs_reduce_returns(market):
    prices, spy, rf = market
    free = backtest(prices, spy, rf, StrategyConfig(cost_bps=0), start="2001-01-01")
    costly = backtest(prices, spy, rf, StrategyConfig(cost_bps=50), start="2001-01-01")
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert free.turnover.sum() == pytest.approx(costly.turnover.sum(), rel=0.05)


def test_execution_lag_delays_trades(market):
    prices, spy, rf = market
    res = backtest(prices, spy, rf, StrategyConfig(exec_lag=1), start="2001-01-01")
    signal_day = next(iter(res.targets))
    pos = res.returns.index.get_loc(signal_day)
    assert res.turnover.iloc[pos] == 0
    assert res.turnover.iloc[pos + 1] > 0


def test_equal_weight_buy_and_hold_matches_manual(market):
    prices, spy, rf = market
    cfg = StrategyConfig(use_momentum=False, use_regime=False, weighting="equal",
                         vol_target=None, cost_bps=0, exec_lag=0)
    res = backtest(prices, spy, rf, cfg, start="2001-01-01", end="2001-01-31")
    # invested at the first close, then weights drift: NAV = average buy-and-hold growth
    window = prices.loc["2001-01-01":"2001-01-31"]
    expected = (window.iloc[-1] / window.iloc[0]).mean() * (1 + rf.iloc[0])  # day 0: still in cash
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert np.allclose(res.exposure, 1.0)


def test_bootstrap_identical_series_has_zero_diff(market):
    prices, spy, rf = market
    r = spy.pct_change().dropna()
    out = stationary_bootstrap_sharpe_diff(r, r, rf, n_boot=50)
    assert out["sharpe_diff"] == 0 and out["ci90_low"] == 0 and out["ci90_high"] == 0


def test_basket_index_has_no_composition_jumps():
    """The original builder's sum(P*V)/sum(V) jumps when an expensive stock enters the basket."""
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"CHEAP": [10.0] * 6, "PRICEY": [np.nan, np.nan, np.nan, 500.0, 500.0, 500.0]}, index=dates)
    volumes = pd.DataFrame({"CHEAP": [1e6] * 6, "PRICEY": [np.nan, np.nan, np.nan, 1e6, 1e6, 1e6]}, index=dates)

    old_vw_price = (prices * volumes).sum(axis=1) / volumes.sum(axis=1)
    assert old_vw_price.pct_change().max() > 20  # +2450 % "return" from a listing

    rets = prices.pct_change(fill_method=None)
    w = profiles.dollar_volume_weights(prices, volumes)
    idx = profiles.basket_index(rets, w)
    assert np.allclose(idx.values, 100.0)  # prices never move -> index stays flat


def test_regime_stats_and_profiles_handle_missing():
    dates = pd.bdate_range("2010-01-01", periods=300)
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300), index=dates)
    regime = pd.Series(["bull"] * 250 + ["bear"] * 50, index=dates)
    stats = profiles.regime_stats(r, regime, min_days=60).set_index("regime")
    assert stats.loc["bull", "n_days"] == 250 and np.isfinite(stats.loc["bull", "ann_return"])
    assert np.isnan(stats.loc["bear", "ann_return"])  # 50 < min_days

    df = pd.DataFrame({
        "regime": ["bull"] * 4 + ["bear"],
        "ann_return": [0.1, 0.2, 0.3, 0.05, np.nan],
        "ann_volatility": [0.1, 0.2, 0.3, 0.15, np.nan],
    })
    labels = profiles.classify_profiles(df)
    assert labels.iloc[-1] == "insufficient data"
    assert "gambler" in labels.values and labels.iloc[:4].nunique() > 1
