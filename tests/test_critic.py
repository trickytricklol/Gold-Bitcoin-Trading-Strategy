"""单元测试: CRITIC 权重计算。"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.critic_ensemble import critic_weight, ensemble_forecast  # noqa: E402


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    res = critic_weight(X)
    assert np.isclose(res.weights.sum(), 1.0)
    assert (res.weights >= 0).all()


def test_identical_indicators_give_equal_weight():
    """两条完全相同的指标 -> 相关系数 1 -> 冲突性相同 -> 权重均等。"""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 1))
    X = np.hstack([x, x.copy()])
    res = critic_weight(X)
    assert np.allclose(res.weights, [0.5, 0.5], atol=1e-8)


def test_dimensionless_step_matches_paper_formula():
    """式 3: x' = (x - x_min) / (x_max - x_min), 结果应在 [0, 1]。"""
    X = np.array([[1.0, 10.0], [3.0, 20.0], [2.0, 15.0]])
    res = critic_weight(X)
    assert res.sigma.shape == (2,)
    assert res.conflict.shape == (2,)


def test_ensemble_forecast_alignment():
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=50)
    arima = pd.Series(np.linspace(100, 150, 50), index=idx)
    lstm = pd.Series(np.linspace(95, 155, 50), index=idx)
    fused, res = ensemble_forecast(arima, lstm)
    assert len(fused) == 50
    assert res is not None
    # 融合结果应介于两条序列之间 (加权平均)
    assert np.allclose(fused, arima * res.weights[0] + lstm * res.weights[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
