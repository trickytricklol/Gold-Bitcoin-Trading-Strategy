"""LSTM 神经网络模型实现 (基于 TensorFlow / Keras)。

对齐论文 2.2 节:
- 单层 LSTM, 12 个神经元, 激活函数 relu;
- Dropout 层 (rate=0.3) 防止过拟合;
- 优化器 Adam, 损失函数 MAE (式 1)。

实现细节:
- 滑动窗口监督学习: 用过去 lookback 天的价格预测下一天价格;
- 训练前做 Min-Max 归一化 (LSTM 训练的必要工程步骤, 主流
  LSTM 预测仓库的标准做法), 预测后反归一化;
- 回测期逐日预测使用"教师强制": 输入窗口来自真实观测,
  与策略回测的"已知当日价格"设定一致。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _TF_AVAILABLE = True
except Exception as e:  # pragma: no cover - 环境缺 TensorFlow 时
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = e


class LSTMModel:
    """单资产 LSTM 预测器 (结构对齐论文 Fig.2 流程图)。"""

    def __init__(
        self,
        units: int = 12,
        lookback: int = 10,
        dropout: float = 0.3,
        activation: str = "relu",
        epochs: int = 120,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 12,
        refit_every: int = 30,
        random_state: int = 42,
    ):
        if not _TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow 未安装或导入失败, 无法使用 LSTM 模型: "
                f"{_TF_IMPORT_ERROR}"
            )
        self.units = units
        self.lookback = lookback
        self.dropout = dropout
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.refit_every = max(1, int(refit_every))
        self.random_state = random_state
        self.model_ = None
        self.scaler_min_ = 0.0
        self.scaler_max_ = 1.0

    # ---------- 数据构建 ----------
    def _build_sequences(self, values: np.ndarray):
        """把一维价格序列切成 (样本, lookback, 1) 与目标。"""
        X, y = [], []
        for i in range(self.lookback, len(values)):
            X.append(values[i - self.lookback : i])
            y.append(values[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _scale(self, values: np.ndarray) -> np.ndarray:
        self.scaler_min_, self.scaler_max_ = values.min(), values.max()
        if self.scaler_max_ - self.scaler_min_ < 1e-12:
            self.scaler_max_ = self.scaler_min_ + 1.0
        return (values - self.scaler_min_) / (self.scaler_max_ - self.scaler_min_)

    def _inverse_scale(self, values: np.ndarray) -> np.ndarray:
        return values * (self.scaler_max_ - self.scaler_min_) + self.scaler_min_

    # ---------- 模型构建 ----------
    def build(self):
        """构建单层 LSTM + Dropout + Dense 网络 (对齐论文 Fig.2)。"""
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        model = keras.Sequential(
            [
                layers.LSTM(
                    self.units,
                    activation=self.activation,
                    input_shape=(self.lookback, 1),
                ),
                layers.Dropout(self.dropout),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model_ = model
        return model

    @property
    def optimizer(self):
        return "adam"

    @property
    def loss(self):
        return "mae"

    # ---------- 训练 ----------
    def fit(self, series: pd.Series) -> "LSTMModel":
        values = series.astype(float).to_numpy()
        scaled = self._scale(values)
        X, y = self._build_sequences(scaled)
        if len(X) < self.lookback + 2:
            raise ValueError("训练样本过少, 无法构建 LSTM 序列。")

        if self.model_ is None:
            self.build()

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stop],
            verbose=0,
        )
        logger.info(
            "LSTM(units=%d, lookback=%d) fitted on %d samples",
            self.units,
            self.lookback,
            len(X),
        )
        return self

    # ---------- 预测 ----------
    def predict_next(self, recent_values: np.ndarray) -> float:
        """给定最近 lookback 个真实价格, 预测下一个价格。"""
        window = np.asarray(recent_values[-self.lookback :], dtype=np.float32)
        scaled = self._scale(window)
        pred_scaled = float(self.model_.predict(scaled[None, :, None], verbose=0)[0, 0])
        return float(self._inverse_scale(np.array([pred_scaled]))[0])

    def rolling_forecast(self, train: pd.Series, test: pd.Series) -> pd.Series:
        """对回测区间逐日 1 步预测 (窗口用真实观测, 每 refit_every 天重训)。"""
        self.fit(train)
        known = list(train.astype(float).to_numpy())
        preds: list[float] = []

        for i, ts in enumerate(test.index):
            pred = self.predict_next(np.array(known))
            preds.append(pred)
            known.append(float(test.loc[ts]))
            if (i + 1) % self.refit_every == 0:
                # 用截至当前的全部观测重训
                recent = pd.Series(known, index=train.index.append(test.index[: i + 1]))
                self.fit(recent)

        return pd.Series(preds, index=test.index, name="lstm_pred")


def lstm_rolling_predictions(
    train: pd.Series,
    test: pd.Series,
    units: int = 12,
    lookback: int = 10,
    dropout: float = 0.3,
    epochs: int = 120,
    batch_size: int = 32,
    patience: int = 12,
    refit_every: int = 30,
) -> pd.Series:
    """便捷函数: 训练 LSTM 并对回测区间输出 1 步滚动预测。"""
    model = LSTMModel(
        units=units,
        lookback=lookback,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        refit_every=refit_every,
    )
    return model.rolling_forecast(train, test)
