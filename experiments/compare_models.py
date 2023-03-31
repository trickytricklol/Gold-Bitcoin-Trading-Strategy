"""单模型 (ARIMA / LSTM) 与 CRITIC 集成模型的预测精度对比。

验证论文结论: 集成模型综合 ARIMA 的线性拟合能力与 LSTM 的非线性
映射能力, 通常优于单一模型。

用法:
    python experiments/compare_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.metrics import report_metrics  # noqa: E402
from src.utils.plots import plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="模型精度对比")
    parser.add_argument("--predictions", default=str(REPO_ROOT / "results" / "predictions.csv"))
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions, parse_dates=[0], index_col=0)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for asset in ["gold", "bitcoin"]:
        actual = preds[f"{asset}_actual"]
        for model in ["arima", "lstm", "ensemble"]:
            m = report_metrics(actual, preds[f"{asset}_{model}"])
            rows.append({"asset": asset, "model": model.upper(), **m})

    table = pd.DataFrame(rows)
    print("\n================ 模型精度对比 ================")
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # 柱状图: 各模型 MAE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, asset in zip(axes, ["gold", "bitcoin"]):
        sub = table[table["asset"] == asset]
        ax.bar(sub["model"], sub["MAE"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.set_title(f"{asset.title()} MAE")
        ax.set_ylabel("MAE")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\n柱状图已保存: {fig_dir / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
