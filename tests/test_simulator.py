"""单元测试: 策略模拟器。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy.simulator import StrategyConfig, simulate_strategy  # noqa: E402


def _make_inputs(n=100, gold_up=True, btc_up=True):
    idx = pd.date_range("2020-01-01", periods=n)
    gold = pd.Series(np.linspace(1500, 1800 if gold_up else 1200, n), index=idx)
    btc = pd.Series(np.linspace(10000, 50000 if btc_up else 5000, n), index=idx)
    # 构造始终看涨的期望收益
    g_er = pd.Series(np.full(n, 30.0), index=idx)
    b_er = pd.Series(np.full(n, 500.0), index=idx)
    return idx, gold, btc, g_er, b_er


def test_no_negative_holdings():
    idx, gold, btc, g_er, b_er = _make_inputs()
    cfg = StrategyConfig(p=0.5, k=0.2)
    res = simulate_strategy(idx, gold, btc, g_er, b_er, cfg)
    assert (res.gold_holding >= 0).all()
    assert (res.bitcoin_holding >= 0).all()
    assert (res.cash_curve >= 0).all()
    assert res.final_wealth > 0


def test_buy_bitcoin_reallocates_p_ratio():
    """触发买比特币时, 现金与黄金各留 (1-p) 比例。"""
    idx = pd.date_range("2020-01-01", periods=5)
    gold = pd.Series([100.0] * 5, index=idx)
    btc = pd.Series([1000.0] * 5, index=idx)
    g_er = pd.Series([0.0] * 5, index=idx)
    b_er = pd.Series([500.0] * 5, index=idx)  # 远超 2% 手续费, 触发买比特币
    cfg = StrategyConfig(p=0.5, k=0.2, initial_capital=1000.0)
    res = simulate_strategy(idx, gold, btc, g_er, b_er, cfg)
    assert res.n_buys["btc"] == 5
    # 首次买入后: B = ((1000*0.5 + 0)/1000)*0.98 = 0.49
    assert np.isclose(res.bitcoin_holding.iloc[0], 0.49)
    assert np.isclose(res.cash_curve.iloc[0], 500.0)
    assert np.isclose(res.gold_holding.iloc[0], 0.0)


def test_sell_bitcoin_on_risk():
    """期望损失超过 k*价格 -> 卖出全部比特币。"""
    idx = pd.date_range("2020-01-01", periods=5)
    gold = pd.Series([100.0] * 5, index=idx)
    btc = pd.Series([1000.0] * 5, index=idx)
    g_er = pd.Series([0.0] * 5, index=idx)
    b_er = pd.Series([-300.0] * 5, index=idx)  # 损失 > 0.2*1000
    cfg = StrategyConfig(p=0.5, k=0.2, initial_capital=1000.0)
    res = simulate_strategy(idx, gold, btc, g_er, b_er, cfg)
    assert res.n_buys["sell_btc"] == 5
    assert np.isclose(res.bitcoin_holding.iloc[-1], 0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
