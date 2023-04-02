"""单元测试: 期望收益计算。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy.expected_return import (  # noqa: E402
    _turning_point_expected_return,
    expected_return_series,
)


def test_uptrend_expected_return():
    """预测持续上涨: 期望收益 = 最高预测价 - 当日价。"""
    today = 100.0
    future = np.array([101.0, 103.0, 105.0, 104.0])  # 在 105 处转折
    assert np.isclose(_turning_point_expected_return(today, future), 5.0)


def test_downtrend_expected_return():
    """预测持续下跌: 期望收益为负 = 最低预测价 - 当日价。"""
    today = 100.0
    future = np.array([99.0, 97.0, 98.0])
    assert np.isclose(_turning_point_expected_return(today, future), -3.0)


def test_no_move_returns_zero():
    today = 100.0
    future = np.array([100.0, 100.5])
    assert _turning_point_expected_return(today, future) == 0.0


def test_series_output_shape():
    idx = pd.date_range("2020-01-01", periods=20)
    actual = pd.Series(np.linspace(100, 120, 20), index=idx)
    pred = actual + 1.0
    er = expected_return_series(actual, pred)
    assert len(er) == 20
    assert er.index.equals(idx)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
