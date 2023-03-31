"""p / k 网格寻优 (论文 3.2.4: 最终收益是 p 与 k 的二元函数)。

直接读取 run_pipeline.py 保存的预测结果 (results/predictions.csv),
复用期望收益计算, 在 (p, k) 网格上跑策略模拟, 输出:
- 最优 (p*, k*) 与最大期末资产;
- 收益矩阵 CSV;
- 论文 Fig.4 / Fig.5 对应的 payoff 曲线与热力图。

用法:
    python experiments/tune_pk.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.strategy.expected_return import expected_return_series  # noqa: E402
from src.strategy.simulator import StrategyConfig, grid_search_pk  # noqa: E402
from src.utils.plots import plot_heatmap, plot_payoff_k  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="p/k 网格寻优")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "config.yaml"))
    parser.add_argument("--predictions", default=str(REPO_ROOT / "results" / "predictions.csv"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("tune_pk")

    preds = pd.read_csv(args.predictions, parse_dates=[0], index_col=0)
    s_cfg = cfg["strategy"]
    t_cfg = cfg["tuning"]
    fig_dir = REPO_ROOT / cfg["output"]["figure_dir"]
    res_dir = REPO_ROOT / cfg["output"]["results_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # 期望收益 (基于 CRITIC 融合预测)
    er_gold = expected_return_series(preds["gold_actual"], preds["gold_ensemble"])
    er_btc = expected_return_series(preds["bitcoin_actual"], preds["bitcoin_ensemble"])
    dates = preds.index

    base_cfg = StrategyConfig(
        initial_capital=s_cfg["initial_capital"],
        commission_gold=s_cfg["commission_gold"],
        commission_bitcoin=s_cfg["commission_bitcoin"],
    )
    p_values = np.arange(t_cfg["p_start"], t_cfg["p_end"] + 1e-9, t_cfg["p_step"])
    k_values = np.arange(t_cfg["k_start"], t_cfg["k_end"] + 1e-9, t_cfg["k_step"])

    log.info("网格规模: p × k = %d × %d = %d 次模拟", len(p_values), len(k_values), len(p_values) * len(k_values))
    wealth_grid, p_best, k_best = grid_search_pk(
        dates=dates,
        gold_price=preds["gold_actual"],
        bitcoin_price=preds["bitcoin_actual"],
        gold_expected_return=er_gold,
        bitcoin_expected_return=er_btc,
        p_values=p_values,
        k_values=k_values,
        base_cfg=base_cfg,
    )
    best_wealth = float(wealth_grid[np.argmin(np.abs(p_values - p_best)), np.argmin(np.abs(k_values - k_best))])
    log.info(
        "最优参数: p* = %.0f%%, k* = %.0f%%, 期末总资产 = $%.0f",
        p_best * 100, k_best * 100, best_wealth,
    )

    # 保存收益矩阵
    grid_df = pd.DataFrame(wealth_grid, index=np.round(p_values, 4), columns=np.round(k_values, 4))
    grid_df.index.name = "p"
    grid_df.columns.name = "k"
    grid_df.to_csv(res_dir / "pk_grid_wealth.csv", encoding="utf-8-sig")

    # 图表
    plot_payoff_k(p_values, k_values, wealth_grid, p_best, k_best, fig_dir)
    plot_heatmap(p_values, k_values, wealth_grid, p_best, k_best, fig_dir / "pk_heatmap.png")

    log.info("图表与收益矩阵已保存到 %s", res_dir)


if __name__ == "__main__":
    main()
