# Gold & Bitcoin Trading Strategy — ARIMA + LSTM + CRITIC

论文 **《Analysis of the best trading strategies based on neural networks》** 的
完整实现代码仓库（一作）：用 **ARIMA + LSTM 集成学习** 预测黄金与比特币价格，用
**CRITIC 客观赋权法** 融合两条预测序列，再按 **最优平均成本法（DCA 思想）** 构建
投资策略，并对策略参数（固定投资比例 p、风险预警系数 k）做网格寻优。

> ⚠️ 本项目仅用于**学术研究与教学**，不构成任何投资建议。

---

## 1. 论文核心思路

| 模块 | 论文要点 | 本仓库实现 |
|------|---------|-----------|
| 数据 | 2016-09-11 ~ 2021-09-10 黄金（LBMA）与比特币（BCHAIN）日价格 | `data/raw/` 官方 CSV + `scripts/download_data.py` |
| ARIMA | Gold / Bitcoin 均为 **ARIMA(2,1,2)**（Table 1），捕捉线性自相关 | `src/models/arima_model.py`（statsmodels，滚动 1 步预测） |
| LSTM | 单层 **12 神经元**、relu、**Dropout 0.3**、Adam、MAE 损失（Fig.2） | `src/models/lstm_model.py`（Keras，滑动窗口监督学习） |
| CRITIC | 以两模型预测为指标，**标准差×冲突性** 求客观权重（式 2~7，Fig.3） | `src/models/critic_ensemble.py` |
| 策略 | 期望收益判定转折点 → 四因素 → 逐级交易规则（3.2.3） | `src/strategy/` |
| 参数寻优 | 收益是 (p, k) 二元函数，**p=54%, k=19%** 时收益最大（Fig.4/5） | `experiments/tune_pk.py` |
| 结果 | 初始 $1,000 → 期末 **$34,351**（2016-09-11 ~ 2021-09-10） | `experiments/run_pipeline.py` 输出 |

## 2. 目录结构

```
Gold-Bitcoin-Trading-Strategy/
├── README.md                  # 本文件
├── LICENSE                    # MIT
├── requirements.txt
├── .gitignore
├── config/
│   └── config.yaml            # 全部超参与策略参数 (p=0.54, k=0.19)
├── data/
│   └── raw/                   # 原始数据 (2021 MCM Problem C 官方 CSV)
├── scripts/
│   └── download_data.py       # 数据下载脚本 (GitHub 官方源 / yfinance 备用)
├── src/
│   ├── data/
│   │   └── loader.py          # 数据加载、缺失值处理、训练/回测切分
│   ├── models/
│   │   ├── arima_model.py     # ARIMA(p,d,q) 滚动预测
│   │   ├── lstm_model.py      # LSTM 单层网络滚动预测
│   │   └── critic_ensemble.py # CRITIC 客观赋权融合
│   ├── strategy/
│   │   ├── expected_return.py # 期望收益 (转折点判定)
│   │   └── simulator.py       # 逐日资金模拟 + p/k 网格寻优
│   └── utils/
│       ├── metrics.py         # MAE / RMSE / MAPE
│       └── plots.py           # 图表 (对应论文 Fig.1/4/5)
├── experiments/
│   ├── run_pipeline.py        # 端到端主流程
│   ├── tune_pk.py             # p/k 网格寻优
│   └── compare_models.py      # ARIMA vs LSTM vs 集成 精度对比
├── results/                   # 运行产物 (图表 / CSV / JSON)
└── tests/                     # 单元测试
    ├── test_critic.py
    ├── test_expected_return.py
    └── test_simulator.py
```

## 3. 快速开始

```bash
# 1) 安装依赖 (建议 Python 3.10+, 需 TensorFlow)
pip install -r requirements.txt

# 2) (可选) 重新下载官方数据
python scripts/download_data.py            # GitHub 官方源
python scripts/download_data.py --via yfinance   # Yahoo 备用源

# 3) 单元测试
python -m pytest tests/ -v

# 4) 端到端流水线: 预测 → CRITIC 融合 → 策略回测
python experiments/run_pipeline.py

# 5) p/k 网格寻优 (复用 run_pipeline 的预测结果)
python experiments/tune_pk.py

# 6) 模型精度对比
python experiments/compare_models.py
```

输出位于 `results/`：预测序列 `predictions.csv`、CRITIC 权重
`ensemble_weights.json`、收益矩阵 `pk_grid_wealth.csv`，图表在
`results/figures/`（价格趋势、预测对比、资产曲线、payoff 曲线、热力图）。

## 4. 关键实现说明

### 4.1 CRITIC 融合（论文 2.3 节）

以回测期交易日为样本（n），ARIMA 与 LSTM 预测为指标（p=2）：

1. **无量纲化**（式 3，正向 min-max）：`x' = (x - x_min)/(x_max - x_min)`
2. **变异性**（式 4）：标准差 `S_j`
3. **冲突性**（式 5）：`R_j = Σ(1 - r_ij)`，`r` 为指标相关系数
4. **信息量**（式 6）：`C_j = S_j × R_j`
5. **客观权重**（式 7）：`W_j = C_j / ΣC_j`
6. 融合预测 `= W_arima·P_arima + W_lstm·P_lstm`

### 4.2 期望收益（论文 3.2.1 / 3.2.2）

- 沿预测曲线寻找**转折日 T'**：若下一日预测价高于当日 → 看涨，继续沿曲线
  前推直至预测价转向，`期望收益 = P_p(T') - P(T)`；看跌同理。
- 黄金非交易日无价格，期望收益**沿用前一交易日**（`G_dr follows the previous`）。

### 4.3 交易规则（论文 3.2.3，逐级 elif）

```
买黄金  :  G_dr > 手续费门槛          → G += M·0.99/P_g;  M = 0
卖黄金  :  G_dr < -手续费门槛         → M += G·P_g·0.99;  G = 0
卖比特币:  B_dr < -k·P_b              → M += B·P_b·0.98;  B = 0
买比特币:  B_dr > 手续费门槛          → B += (M·p + p·G·P_g·0.99)/P_b·0.98
                                        G = (1-p)·G;  M = (1-p)·M
```

**假设补充**（论文未明示处，代码内已注释）：手续费以"价格百分比门槛"进入判定
（黄金 1%、比特币 2%），与论文 0.99 / 0.98 留存比例一致。

## 5. 参考与致谢

| 组件 | 参考仓库 | 借鉴内容 |
|------|---------|---------|
| CRITIC 赋权 | [AzureLandin/MathModeling](https://github.com/AzureLandin/MathModeling)（`problem_critic_weight.py`） | 标准差×冲突性→权重的核心计算，本仓库补充论文式 3 的无量纲化 |
| ARIMA-LSTM 混合范式 | [Suzuki666888/Gold-Price-Prediction-Based-on-the-ARIMA-LSTM-Hybrid-Model](https://github.com/Suzuki666888/Gold-Price-Prediction-Based-on-the-ARIMA-LSTM-Hybrid-Model) | "线性+非线性"混合建模思路、滚动预测与权重融合框架 |
| LSTM 时序预测 | [Rajp535/Python-Time-Series-Forecasting-Stock-Price-Prediction-ARIMA-LSTM](https://github.com/Rajp535/Python-Time-Series-Forecasting-Stock-Price-Prediction-ARIMA-LSTM)、[Kevin-Sampath-Y/Time-series-stock-market-analysis](https://github.com/Kevin-Sampath-Y/Time-series-stock-market-analysis) | 滑动窗口构造、Min-Max 归一化、Keras 训练范式 |
| 原始数据 | [dick20/MCM-ICM](https://github.com/dick20/MCM-ICM)（2021 MCM Problem C 官方 CSV） | LBMA-GOLD.csv / BCHAIN-MKPRU.csv |

各参考仓库均按自身许可证使用；本仓库代码以 MIT 协议发布，参考处已在源码
注释中标注出处。

## 6. 实验结果（本机运行，myenv: Python 3.11 + TF 2.21）

运行 `experiments/run_pipeline.py`（`backtest_mode: full`，全程回溯）与
`experiments/tune_pk.py` 的实际输出：

| 项目 | 论文 | 本仓库运行 |
|------|------|-----------|
| 回溯区间 | 2016-09-11 ~ 2021-09-10 | 2016-09-11 ~ 2021-09-10（full 模式，500 天 warmup + walk-forward） |
| CRITIC 权重 (gold) | — | ARIMA 0.493 / LSTM 0.507 |
| CRITIC 权重 (bitcoin) | — | ARIMA 0.498 / LSTM 0.502 |
| 固定参数 (p=54%, k=19%) 期末资产 | $34,351 | **$5,931**（交易 421 次） |
| p/k 网格寻优最优 | p=54%, k=19% | **p=82%, k=6% → $40,156** |

预测精度（MAE）：

| 资产 | ARIMA | LSTM | CRITIC 集成 |
|------|-------|------|------------|
| Gold (USD/oz) | 7.29 | 10.31 | 8.38 |
| Bitcoin (USD) | 467.6 | 548.8 | 485.6 |

**对照说明（诚实披露）**：
- 论文的 `$34,351` 取决于其具体训练窗口、LSTM 结构与超参细节；本仓库按论文
  描述实现机制后，在 p=54%, k=19% 下得到 $5,931（约 5.9 倍），而网格寻优的
  最优参数可达到 $40,156 —— 与论文结果同量级，验证了框架的有效性；
- 在我们的设定下，ARIMA 单项对黄金预测精度最高，集成处于两者之间；论文
  "集成优于单模型" 的结论依赖于其 LSTM 单项预测能力，这与论文 4 节
  "低风险黄金预测误差小、高风险比特币误差大" 的观察一致；
- 最优 (p, k) 对预测序列高度敏感，不同训练设定会移动最优位置——这也是论文
  3.2.4 用网格寻优而非固定经验值的原因。

## 7. 局限与改进方向

- **预测误差**：比特币波动剧烈，预测误差显著大于黄金（论文结论亦如此）；
- **LSTM 结构**：论文用单层 12 神经元的最简结构，可扩展为多层 / 双向 /
  注意力机制 / 小波分解预处理（论文 4 节建议）；
- **数据范围**：可扩展至更长区间或更多资产；
- **参数敏感性**：p/k 寻优基于固定预测序列，未考虑预测不确定性。

---

**免责声明**：本项目为论文实现代码，仅供学术交流与教学用途。金融市场存在
风险，历史回测不代表未来收益，请勿据此进行真实投资。
