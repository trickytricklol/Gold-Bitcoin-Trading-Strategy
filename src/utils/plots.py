"""可视化工具 —— 产出与论文 Fig.1 / Fig.4 / Fig.5 对应的图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头环境可用

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_price_trend(gold: pd.Series, bitcoin: pd.Series, save_path: Path | str):
    """论文 Fig.1: 黄金与比特币收盘价趋势。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(gold.index, gold, color="#1f77b4", linewidth=1.0)
    axes[0].set_title("Gold Price (USD/troy oz)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("USD (PM)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(bitcoin.index, bitcoin, color="#1f77b4", linewidth=1.0)
    axes[1].set_title("Bitcoin Price (USD)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("USD")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_prediction_vs_actual(
    actual: pd.Series,
    preds: dict[str, pd.Series],
    asset: str,
    save_path: Path | str,
):
    """预测 vs 实际对比图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual.index, actual, label="Actual", color="black", linewidth=1.4)
    for name, p in preds.items():
        ax.plot(p.index, p, label=name, linewidth=1.0, alpha=0.9)
    ax.set_title(f"{asset} — Model Predictions vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_wealth_curve(
    result_curve: pd.Series,
    initial_capital: float,
    save_path: Path | str,
    title: str = "Total Wealth Curve",
):
    """策略总资产曲线。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(result_curve.index, result_curve, color="#d62728", linewidth=1.4)
    ax.axhline(initial_capital, color="gray", linestyle="--", label=f"Initial ${initial_capital:,.0f}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Wealth (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_payoff_k(p_values: np.ndarray, k_values: np.ndarray, wealth_grid: np.ndarray,
                  p_best: float, k_best: float, save_dir: Path | str):
    """论文 Fig.4 / Fig.5: 固定一侧最优值, 收益随另一参数的变化。"""
    save_dir = Path(save_dir)
    p_idx = int(np.argmin(np.abs(p_values - p_best)))
    k_idx = int(np.argmin(np.abs(k_values - k_best)))

    # Fig.4: p = p_best 时收益随 k 变化
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, wealth_grid[p_idx, :], color="#2ca02c", linewidth=1.6)
    ax.scatter([k_best], [wealth_grid[p_idx, k_idx]], color="red", zorder=5,
               label=f"k*={k_best:.2f} (wealth=${wealth_grid[p_idx, k_idx]:,.0f})")
    ax.set_title(f"Payoff vs k at p = {p_best:.0%}")
    ax.set_xlabel("k (risk coefficient)")
    ax.set_ylabel("Final Wealth (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "payoff_vs_k.png", dpi=150)
    plt.close(fig)

    # Fig.5: k = k_best 时收益随 p 变化
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_values, wealth_grid[:, k_idx], color="#1f77b4", linewidth=1.6)
    ax.scatter([p_best], [wealth_grid[p_idx, k_idx]], color="red", zorder=5,
               label=f"p*={p_best:.0%} (wealth=${wealth_grid[p_idx, k_idx]:,.0f})")
    ax.set_title(f"Payoff vs p at k = {k_best:.0%}")
    ax.set_xlabel("p (investment ratio in bitcoin)")
    ax.set_ylabel("Final Wealth (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "payoff_vs_p.png", dpi=150)
    plt.close(fig)


def plot_heatmap(p_values: np.ndarray, k_values: np.ndarray, wealth_grid: np.ndarray,
                 p_best: float, k_best: float, save_path: Path | str):
    """p × k 收益热力图。"""
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(wealth_grid.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[p_values[0], p_values[-1], k_values[0], k_values[-1]])
    ax.scatter([p_best], [k_best], color="red", marker="*", s=200,
               label=f"Optimal (p={p_best:.2f}, k={k_best:.2f})")
    ax.set_xlabel("p (investment ratio)")
    ax.set_ylabel("k (risk coefficient)")
    ax.set_title("Final Wealth Heatmap over (p, k)")
    ax.legend()
    fig.colorbar(im, ax=ax, label="Final Wealth (USD)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
