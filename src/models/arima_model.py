"""ARIMA 模型实现 (基于 statsmodels)。

对齐论文 2.1 节:
- 通过 ACF / PACF 确定阶数, 论文 Table 1 给出 Gold 与 Bitcoin 均为
  ARIMA(2, 1, 2)。
- 用于捕捉时间序列中的线性自相关结构。

滚动预测策略 (GitHub 上主流 ARIMA 时间序列仓库的标准做法):
- 训练集上拟合 ARIMA; 对回测期的每一天做 1 步前向预测;
- 每 refit_every 天重新估计一次参数, 期间用 `apply()` 以固定参数
  吸收新观测, 兼顾精度与效率。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

logger = logging.getLogger(__name__)


class ARIMAModel:
    """ARIMA(p, d, q) 单资产预测器。"""

    def __init__(self, p: int = 2, d: int = 1, q: int = 2, refit_every: int = 30):
        self.order = (p, d, q)
        self.refit_every = max(1, int(refit_every))
        self.model_ = None
        self.results_ = None

    def fit(self, series: pd.Series) -> "ARIMAModel":
        """在给定序列上拟合 ARIMA。"""
        self.model_ = SM_ARIMA(series.astype(float), order=self.order)
        self.results_ = self.model_.fit(method_kwargs={"maxiter": 200})
        logger.info(
            "ARIMA%s fitted on %d obs | AIC=%.2f",
            self.order,
            len(series),
            self.results_.aic,
        )
        return self

    def forecast_one(self) -> float:
        """基于当前拟合结果预测 1 步。"""
        return float(self.results_.forecast(steps=1).iloc[-1])

    def rolling_forecast(
        self, train: pd.Series, test: pd.Series
    ) -> pd.Series:
        """对 test 区间逐日做 1 步前向预测。

        - 先用 train 拟合;
        - 每预测一天, 把该天真实值并入已知序列, 用 `apply` 固定参数更新;
        - 每 refit_every 天重新完整估计一次。
        """
        known = train.copy()
        preds: list[float] = []
        self.fit(known)

        for i, (ts, _y) in enumerate(test.items()):
            pred = self.forecast_one()
            preds.append(pred)
            known = pd.concat([known, pd.Series([_y], index=[ts])])
            # apply(): 用固定参数吸收新观测, 无需重估
            self.results_ = self.results_.apply(known)
            if (i + 1) % self.refit_every == 0:
                self.fit(known)

        return pd.Series(preds, index=test.index, name="arima_pred")


def arima_rolling_predictions(
    train: pd.Series,
    test: pd.Series,
    p: int = 2,
    d: int = 1,
    q: int = 2,
    refit_every: int = 30,
) -> pd.Series:
    """便捷函数: 训练 ARIMA 并对回测区间输出 1 步滚动预测。"""
    model = ARIMAModel(p=p, d=d, q=q, refit_every=refit_every)
    return model.rolling_forecast(train, test)
