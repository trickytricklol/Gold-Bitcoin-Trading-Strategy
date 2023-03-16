"""期望收益计算 (论文 3.2.1 ~ 3.2.2)。

定义 (论文公式):
    B_dr = B_p(T') - B(T)     比特币期望收益
    G_dr = G_p(T') - G(T)     黄金期望收益

其中 T 为当前交易日, T' 为预测曲线上的"转折日", B_p/G_p 为模型预测价格。

规则 (论文 3.2.2 影响因素 1、2):
- 若下一交易日预测价高于当日实际价 -> 判定为上涨(收益), 沿预测曲线
  继续看, 直到预测价格发生转向 (由涨转跌或由跌转涨), 转折日的预测价
  与当日实际价之差即期望收益;
- 黄金在非交易日没有价格, 期望收益沿用前一交易日 (G_dr follows
  the previous) —— 由回测主循环负责 (simulator 逐日调用时对非交易日
  使用上一次结果)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _turning_point_expected_return(
    price_today: float, pred_future: np.ndarray
) -> float:
    """沿预测曲线寻找转折日并计算期望收益。

    Args:
        price_today: 当日实际价格 P(T)。
        pred_future: 从 T+1 起的预测价格序列 [P_p(T+1), P_p(T+2), ...]。

    Returns:
        期望收益 P_p(T') - P(T)。若无法判定趋势, 返回 0。
    """
    if len(pred_future) == 0:
        return 0.0

    # 方向: 下一日预测高于当日 -> 看涨; 低于 -> 看跌
    first = pred_future[0]
    if abs(first - price_today) < 1e-12:
        return 0.0
    direction = 1.0 if first > price_today else -1.0

    turning_price = first
    for i in range(len(pred_future) - 1):
        step = pred_future[i + 1] - pred_future[i]
        if step * direction < 0:        # 预测价格转向
            break
        turning_price = pred_future[i + 1]

    return float(turning_price - price_today)


def expected_return_series(
    actual: pd.Series, pred: pd.Series, horizon: int = 30
) -> pd.Series:
    """对每一天计算期望收益。

    Args:
        actual: 实际价格序列 (对齐回测日历)。
        pred:   模型预测价格序列 (同一索引)。
        horizon: 最多向前看的天数 (防止无限延伸; 论文中实际受转折限制)。

    Returns:
        与 actual 同索引的期望收益序列。
    """
    aligned = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    out: list[float] = []
    idx = aligned.index

    for i, ts in enumerate(idx):
        future = aligned["pred"].to_numpy()[i + 1 : i + 1 + horizon]
        out.append(_turning_point_expected_return(aligned["actual"].iloc[i], future))

    return pd.Series(out, index=idx, name="expected_return")
