"""端到端主流程: 数据 → ARIMA/LSTM 预测 → CRITIC 融合 → 策略回测。

复现论文全链路:
  1. 加载 2016-09-11 ~ 2021-09-10 黄金/比特币数据并可视化 (Fig.1);
  2. 训练 ARIMA(2,1,2) 与 LSTM(12) 并滚动预测价格;
  3. 用 CRITIC 客观赋权融合两条预测序列;
  4. 计算期望收益, 按论文 3.2.3 规则以 p=54%, k=19% 回测;
  5. 输出资产曲线与各类中间结果。

两种回溯模式 (config.yaml -> data.backtest_mode):
  - full: 全程回溯。用前 warmup_days 天作初始训练窗口, 逐日 walk-forward
          预测 2016-2021 全区间, 策略从 2016-09-11 的 $1,000 起跑 (论文设定);
  - tail: 快速模式。前 70% 训练, 后 30% 回测。

用法:
    python experiments/run_pipeline.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_market_data, train_test_split  # noqa: E402
from src.models.arima_model import ARIMAModel  # noqa: E402
from src.models.critic_ensemble import ensemble_forecast, weights_to_string  # noqa: E402
from src.models.lstm_model import LSTMModel  # noqa: E402
from src.strategy.expected_return import expected_return_series  # noqa: E402
from src.strategy.simulator import StrategyConfig, simulate_strategy  # noqa: E402
from src.utils.plots import (  # noqa: E402
    plot_prediction_vs_actual,
    plot_price_trend,
    plot_wealth_curve,
)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ARIMA+LSTM+CRITIC 交易策略端到端复现")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "config.yaml"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_cfg = cfg["output"]
    fig_dir = REPO_ROOT / out_cfg["figure_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(REPO_ROOT / out_cfg["log_file"])
    log = logging.getLogger("pipeline")
    log.info("=== 论文复现流水线启动 (mode=%s) ===", cfg["data"].get("backtest_mode", "tail"))

    t0 = time.time()

    # ---------- 1. 数据 ----------
    data_cfg = cfg["data"]
    market = load_market_data(
        REPO_ROOT / data_cfg["raw_dir"],
        gold_file=data_cfg["gold_file"],
        bitcoin_file=data_cfg["bitcoin_file"],
        start=data_cfg["start_date"],
        end=data_cfg["end_date"],
    )
    log.info(
        "数据区间 %s ~ %s, 共 %d 天",
        market.dates[0].date(), market.dates[-1].date(), len(market.dates),
    )
    plot_price_trend(market.gold, market.bitcoin, fig_dir / "fig1_price_trend.png")

    mode = data_cfg.get("backtest_mode", "tail")
    if mode == "full":
        # 全程回溯: 每个资产按自身日历取前 warmup 天为初始训练窗口
        warmup = int(data_cfg["warmup_days"])
        asset_series = {
            "gold": market.gold.dropna(),
            "bitcoin": market.bitcoin,
        }
        splits = {}
        for asset, s in asset_series.items():
            splits[asset] = (s.iloc[:warmup], s.iloc[warmup:])
            log.info(
                "资产 %s: 初始训练 %d 天, 预测区间 %s ~ %s (%d 天)",
                asset, len(splits[asset][0]),
                splits[asset][1].index[0].date(), splits[asset][1].index[-1].date(),
                len(splits[asset][1]),
            )
        strat_dates, strat_gold_px, strat_btc_px = market.dates, market.gold, market.bitcoin
    else:
        train, test = train_test_split(market, train_ratio=data_cfg["train_ratio"])
        log.info("训练 %d 天, 回测 %d 天", len(train.dates), len(test.dates))
        splits = {
            "gold": (train.gold.dropna(), test.gold.dropna()),
            "bitcoin": (train.bitcoin, test.bitcoin),
        }
        strat_dates, strat_gold_px, strat_btc_px = test.dates, test.gold, test.bitcoin

    # ---------- 2. ARIMA / LSTM 滚动预测 ----------
    a_cfg = cfg["arima"]
    l_cfg = cfg["lstm"]
    predictions = {}

    for asset in ["gold", "bitcoin"]:
        tr, te = splits[asset]
        log.info(">>> 预测资产: %s", asset)
        # ARIMA
        arima = ARIMAModel(
            p=a_cfg[asset]["p"], d=a_cfg[asset]["d"], q=a_cfg[asset]["q"],
            refit_every=a_cfg["refit_every"],
        )
        predictions[f"{asset}_arima"] = arima.rolling_forecast(tr, te)

        # LSTM
        lstm = LSTMModel(
            units=l_cfg["units"], lookback=l_cfg["lookback"],
            dropout=l_cfg["dropout"], epochs=l_cfg["epochs"],
            batch_size=l_cfg["batch_size"], patience=l_cfg["patience"],
            refit_every=l_cfg["refit_every"],
        )
        predictions[f"{asset}_lstm"] = lstm.rolling_forecast(tr, te)

    # ---------- 3. CRITIC 融合 ----------
    ensemble = {}
    critic_meta = {}
    for asset in ["gold", "bitcoin"]:
        fused, result = ensemble_forecast(
            predictions[f"{asset}_arima"], predictions[f"{asset}_lstm"]
        )
        ensemble[f"{asset}_ensemble"] = fused
        critic_meta[asset] = {
            "weights": result.weights.tolist(),
            "sigma": result.sigma.tolist(),
            "conflict": result.conflict.tolist(),
            "info": result.info.tolist(),
        }
        log.info("资产 %s: %s", asset, weights_to_string(result).replace("\n", " | "))

    # ---------- 4. 期望收益 ----------
    er = {}
    for asset in ["gold", "bitcoin"]:
        _, te = splits[asset]
        er[f"{asset}_expected_return"] = expected_return_series(
            te, ensemble[f"{asset}_ensemble"]
        )

    # ---------- 5. 策略回测 (p=54%, k=19%) ----------
    s_cfg = cfg["strategy"]
    strategy_cfg = StrategyConfig(
        initial_capital=s_cfg["initial_capital"],
        commission_gold=s_cfg["commission_gold"],
        commission_bitcoin=s_cfg["commission_bitcoin"],
        p=s_cfg["p"],
        k=s_cfg["k"],
    )
    result = simulate_strategy(
        dates=strat_dates,
        gold_price=strat_gold_px,
        bitcoin_price=strat_btc_px,
        gold_expected_return=er["gold_expected_return"],
        bitcoin_expected_return=er["bitcoin_expected_return"],
        cfg=strategy_cfg,
    )
    log.info(
        "策略回测完成: 初始资金 $%.0f -> 期末总资产 $%.0f (p=%.0f%%, k=%.0f%%)",
        strategy_cfg.initial_capital, result.final_wealth, s_cfg["p"] * 100, s_cfg["k"] * 100,
    )
    log.info("交易次数统计: %s", result.n_buys)

    plot_wealth_curve(
        result.wealth_curve, strategy_cfg.initial_capital,
        fig_dir / "wealth_curve.png",
        title=f"Total Wealth Curve (p={s_cfg['p']:.0%}, k={s_cfg['k']:.0%}, mode={mode})",
    )
    for asset in ["gold", "bitcoin"]:
        _, te = splits[asset]
        plot_prediction_vs_actual(
            te,
            {
                "ARIMA": predictions[f"{asset}_arima"],
                "LSTM": predictions[f"{asset}_lstm"],
                "CRITIC Ensemble": ensemble[f"{asset}_ensemble"],
            },
            asset.title(),
            fig_dir / f"{asset}_predictions.png",
        )

    # ---------- 6. 保存中间结果 ----------
    res_dir = REPO_ROOT / out_cfg["results_dir"]
    res_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(
        {
            "gold_actual": splits["gold"][1],
            "gold_arima": predictions["gold_arima"],
            "gold_lstm": predictions["gold_lstm"],
            "gold_ensemble": ensemble["gold_ensemble"],
            "bitcoin_actual": splits["bitcoin"][1],
            "bitcoin_arima": predictions["bitcoin_arima"],
            "bitcoin_lstm": predictions["bitcoin_lstm"],
            "bitcoin_ensemble": ensemble["bitcoin_ensemble"],
        }
    )
    df_out.to_csv(res_dir / "predictions.csv", encoding="utf-8-sig")

    with open(res_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(critic_meta, f, ensure_ascii=False, indent=2)

    summary = {
        "mode": mode,
        "final_wealth": result.final_wealth,
        "initial_capital": strategy_cfg.initial_capital,
        "p": s_cfg["p"],
        "k": s_cfg["k"],
        "trades": result.n_buys,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info("结果已保存到 %s", res_dir)
    log.info("=== 流水线完成, 耗时 %.1fs ===", time.time() - t0)


if __name__ == "__main__":
    main()
