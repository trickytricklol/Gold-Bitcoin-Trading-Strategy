"""预测精度评价指标 (对齐论文式 1: MAE, 另附 RMSE / MAPE)。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 MAE = Σ|y_i - x_i| / n (论文式 1)。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对百分比误差。"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = np.abs(y_true) > 1e-12
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def report_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    """一次性输出 MAE / RMSE / MAPE。"""
    df = pd.DataFrame({"y": actual, "p": pred}).dropna()
    return {
        "MAE": mae(df["y"], df["p"]),
        "RMSE": rmse(df["y"], df["p"]),
        "MAPE(%)": mape(df["y"], df["p"]),
    }
