"""CRITIC 客观赋权法 —— ARIMA 与 LSTM 预测的融合 (论文 2.3 节)。

CRITIC (Criteria Importance Through Intercriteria Correlation) 法基于
指标对比强度 (变异性) 与指标间冲突性确定客观权重, 完全由数据本身驱动。

流程 (对齐论文 Fig.3 与公式 2~7):
1. 构建数据矩阵 X: 样本 = 回测期每个交易日, 指标 = [ARIMA 预测, LSTM 预测], p=2;
2. 无量纲化: 正向化 min-max 归一化 (式 3);
3. 变异性: 标准差 S_j (式 4);
4. 冲突性: R_j = Σ_i (1 - r_ij), r 为指标间相关系数 (式 5);
5. 信息量: C_j = S_j * R_j (式 6);
6. 客观权重: W_j = C_j / Σ C_j (式 7);
7. 融合预测 = Σ W_j * 各模型预测。

实现参考:
- AzureLandin/MathModeling (模块四练习/code/problem_critic_weight.py):
  https://github.com/AzureLandin/MathModeling —— CRITIC 权重核心计算
  (在此基础上补充论文公式 3 要求的无量纲化步骤)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CriticResult:
    """CRITIC 计算过程与结果。"""

    weights: np.ndarray          # 各指标客观权重 W_j
    sigma: np.ndarray            # 标准差 S_j
    conflict: np.ndarray         # 冲突性 R_j
    info: np.ndarray             # 信息量 C_j
    corr: np.ndarray             # 指标相关系数矩阵


def critic_weight(X: np.ndarray) -> CriticResult:
    """计算 CRITIC 客观权重。

    Args:
        X: 形状 (n_samples, n_indicators) 的原始指标矩阵。

    Returns:
        CriticResult 包含权重与中间量。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X 必须是二维矩阵 (样本, 指标)")

    # (2) 正向化无量纲处理 (式 3)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    span = x_max - x_min
    span[span < 1e-12] = 1.0  # 常数指标保护
    Xn = (X - x_min) / span

    # (3) 变异性: 标准差 (式 4), ddof=1 样本标准差
    sigma = np.std(Xn, axis=0, ddof=1)

    # (4) 冲突性: 相关系数 (式 5)
    corr = np.corrcoef(Xn, rowvar=False)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    conflict = np.sum(1.0 - corr, axis=0)

    # (5) 信息量 (式 6)
    info = sigma * conflict

    # (6) 客观权重 (式 7)
    total = info.sum()
    weights = info / total if total > 0 else np.full_like(info, 1.0 / X.shape[1])

    return CriticResult(weights=weights, sigma=sigma, conflict=conflict, info=info, corr=corr)


def ensemble_forecast(
    arima_pred: pd.Series, lstm_pred: pd.Series, weights: np.ndarray | None = None
) -> pd.Series:
    """按 CRITIC 权重融合两条预测序列 (式 7 之后的加权输出)。

    权重缺省时自动调用 critic_weight 计算: 以两条预测序列为指标,
    交易日为样本 (论文 3.1 节 (1): n = 预测期天数, p = 2)。
    """
    df = pd.DataFrame({"arima": arima_pred, "lstm": lstm_pred}).dropna()
    if len(df) < 2:
        raise ValueError("预测序列过短或全为空, 无法融合。")

    if weights is None:
        result = critic_weight(df.to_numpy())
        weights = result.weights
    else:
        result = None

    fused = df["arima"].to_numpy() * weights[0] + df["lstm"].to_numpy() * weights[1]
    return pd.Series(fused, index=df.index, name="ensemble_pred"), result


def weights_to_string(result: CriticResult, labels=("ARIMA", "LSTM")) -> str:
    """把 CRITIC 结果格式化为可读文本。"""
    lines = [
        f"CRITIC 权重: {dict(zip(labels, np.round(result.weights, 4)))}",
        f"标准差 S_j   : {np.round(result.sigma, 6)}",
        f"冲突性 R_j   : {np.round(result.conflict, 4)}",
        f"信息量 C_j   : {np.round(result.info, 4)}",
    ]
    return "\n".join(lines)
