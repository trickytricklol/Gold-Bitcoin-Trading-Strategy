"""资金模拟器 —— 论文 3.2.3 投资措施 (Investment measures) 的逐日实现。

论文给出的伪代码 (其中 G_r(T) / B_r(T) 为当日实际价格):

    If    期望黄金收益 > 买入黄金手续费
          G = G + M / G_r(T) * 0.99;   M = 0;
    Elseif 期望黄金损失 > 卖出黄金手续费
          M = M + G * G_r(T) * 0.99;   G = 0;
    Elseif 期望比特币损失 > k * 比特币当日价格
          M = M + B * B_r(T) * 0.98;   B = 0;
    Elseif 期望比特币收益 > 买入比特币手续费
          B = B + ((M*p + p*G*G_r(T)*0.99) / B_r(T)) * 0.98;
          G = (1-p)*G;
          M = (1-p)*M;

符号: M=现金, B=比特币数量, G=黄金数量(盎司),
      p=固定资产(比特币)投资比例, k=比特币风险预警系数。

假设 (论文未明示、本实现补充并在 README 中说明):
- 手续费以"价差门槛"形式进入判定: 买黄金门槛 = commission_gold * 金价,
  卖黄金门槛 = commission_gold * 金价, 买比特币门槛 = commission_bitcoin * 币价;
- 黄金非交易日: 价格沿用上一交易日 (ffill), 期望收益沿用上一交易日
  (论文: "G_dr follows the previous")。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """策略参数。"""

    initial_capital: float = 1000.0
    commission_gold: float = 0.01     # 黄金买卖手续费比例 (留存 0.99)
    commission_bitcoin: float = 0.02  # 比特币买卖手续费比例 (留存 0.98)
    p: float = 0.54                   # 固定资产投资比例
    k: float = 0.19                   # 比特币风险预警系数


@dataclass
class SimulationResult:
    """回测结果。"""

    wealth_curve: pd.Series           # 每日总资产
    cash_curve: pd.Series             # 每日现金
    gold_holding: pd.Series           # 每日黄金持仓 (盎司)
    bitcoin_holding: pd.Series        # 每日比特币持仓
    final_wealth: float
    n_buys: dict = field(default_factory=dict)   # 各类交易次数统计


def simulate_strategy(
    dates: pd.DatetimeIndex,
    gold_price: pd.Series,
    bitcoin_price: pd.Series,
    gold_expected_return: pd.Series,
    bitcoin_expected_return: pd.Series,
    cfg: StrategyConfig,
) -> SimulationResult:
    """按论文 3.2.3 的规则逐日模拟交易。

    Args:
        dates: 回测日历 (每日)。
        gold_price / bitcoin_price: 对齐 dates 的逐日价格。
        gold_expected_return / bitcoin_expected_return: 期望收益序列
            (黄金非交易日会自动沿用上一交易日: "G_dr follows the previous")。
        cfg: 策略参数 (p, k, 手续费)。
    """
    # 期望收益对齐与黄金非交易日沿用
    g_er = gold_expected_return.reindex(dates).ffill().fillna(0.0).to_numpy()
    b_er = bitcoin_expected_return.reindex(dates).fillna(0.0).to_numpy()
    # 黄金非交易日无价格 -> 沿用上一交易日 (与数据加载层行为一致)
    g_px = gold_price.reindex(dates).ffill().bfill().to_numpy()
    b_px = bitcoin_price.reindex(dates).to_numpy()

    M, B, G = cfg.initial_capital, 0.0, 0.0
    wealth, cash, gold_hold, btc_hold = [], [], [], []
    n_buys = {"gold": 0, "sell_gold": 0, "btc": 0, "sell_btc": 0}

    for t in range(len(dates)):
        if g_px[t] <= 0 or b_px[t] <= 0:
            # 数据缺口日不交易
            wealth.append(M + B * b_px[t - 1] + G * g_px[t - 1] if t > 0 else M)
            cash.append(M)
            gold_hold.append(G)
            btc_hold.append(B)
            continue

        g_er_t, b_er_t = g_er[t], b_er[t]
        gold_comm = cfg.commission_gold * g_px[t]
        btc_comm = cfg.commission_bitcoin * b_px[t]

        # --- 逐级判定 (严格按论文 elif 顺序) ---
        if g_er_t > gold_comm:                      # 买黄金
            G += M * (1 - cfg.commission_gold) / g_px[t]
            M = 0.0
            n_buys["gold"] += 1
        elif g_er_t < -gold_comm:                   # 卖黄金 (止损)
            M += G * g_px[t] * (1 - cfg.commission_gold)
            G = 0.0
            n_buys["sell_gold"] += 1
        elif b_er_t < -cfg.k * b_px[t]:             # 卖比特币 (风险对冲)
            M += B * b_px[t] * (1 - cfg.commission_bitcoin)
            B = 0.0
            n_buys["sell_btc"] += 1
        elif b_er_t > btc_comm:                     # 买比特币
            usd_for_btc = M * cfg.p + cfg.p * G * g_px[t] * (1 - cfg.commission_gold)
            B += (usd_for_btc / b_px[t]) * (1 - cfg.commission_bitcoin)
            G *= 1 - cfg.p
            M *= 1 - cfg.p
            n_buys["btc"] += 1

        wealth.append(M + B * b_px[t] + G * g_px[t])
        cash.append(M)
        gold_hold.append(G)
        btc_hold.append(B)

    idx = pd.DatetimeIndex(dates)
    return SimulationResult(
        wealth_curve=pd.Series(wealth, index=idx),
        cash_curve=pd.Series(cash, index=idx),
        gold_holding=pd.Series(gold_hold, index=idx),
        bitcoin_holding=pd.Series(btc_hold, index=idx),
        final_wealth=wealth[-1],
        n_buys=n_buys,
    )


def grid_search_pk(
    dates: pd.DatetimeIndex,
    gold_price: pd.Series,
    bitcoin_price: pd.Series,
    gold_expected_return: pd.Series,
    bitcoin_expected_return: pd.Series,
    p_values: np.ndarray,
    k_values: np.ndarray,
    base_cfg: StrategyConfig,
) -> tuple[np.ndarray, float, float]:
    """在 (p, k) 网格上搜索最终收益最大值 (论文 3.2.4)。

    Returns:
        (final_wealth 矩阵 [p × k], 最优 p, 最优 k)。
    """
    wealth_grid = np.zeros((len(p_values), len(k_values)))
    for i, p in enumerate(p_values):
        for j, k in enumerate(k_values):
            cfg = StrategyConfig(
                initial_capital=base_cfg.initial_capital,
                commission_gold=base_cfg.commission_gold,
                commission_bitcoin=base_cfg.commission_bitcoin,
                p=float(p),
                k=float(k),
            )
            res = simulate_strategy(
                dates, gold_price, bitcoin_price,
                gold_expected_return, bitcoin_expected_return, cfg,
            )
            wealth_grid[i, j] = res.final_wealth

    best_i, best_j = np.unravel_index(np.argmax(wealth_grid), wealth_grid.shape)
    return wealth_grid, float(p_values[best_i]), float(k_values[best_j])
