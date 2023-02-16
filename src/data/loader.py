"""数据加载与预处理。

读取 2021 MCM Problem C 官方数据集:
- LBMA-GOLD.csv   黄金每日价格 (USD/troy ounce), 仅交易日有价
- BCHAIN-MKPRU.csv 比特币每日价格 (USD), 全年每日有价

处理流程 (对齐论文第 3.1 节 "Data acquisition" 与数据清洗描述):
1. 解析日期 (M/D/YY), 按日索引
2. 缺失值处理: 比特币连续、黄金仅交易日有价 -> 以交易日索引保留,
   回测时用前向填充 (ffill) 补齐非交易日的黄金价格,
   期望收益则沿用前一交易日 (论文: "G_dr follows the previous")
3. 对齐两资产日历, 提供训练/回测切分
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """对齐后的市场数据。

    Attributes:
        df: 以日期为索引的 DataFrame, 列 = [gold, bitcoin], 均为每日价格。
        gold_trading_days: 黄金实际交易日索引 (黄金原始索引)。
    """

    df: pd.DataFrame
    gold_trading_days: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def gold(self) -> pd.Series:
        return self.df["gold"]

    @property
    def bitcoin(self) -> pd.Series:
        return self.df["bitcoin"]


def _parse_mcm_dates(series: pd.Series) -> pd.Series:
    """把 '9/12/16' 这类 M/D/YY 解析为 Timestamp。"""
    return pd.to_datetime(series, format="%m/%d/%y")


def load_gold(path: Path | str) -> pd.Series:
    """读取 LBMA 黄金数据, 返回按日期索引的价格 Series (仅交易日)。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    price_col = [c for c in df.columns if c.lower().startswith("usd")][0]
    out = pd.Series(
        pd.to_numeric(df[price_col], errors="coerce").values,
        index=_parse_mcm_dates(df["Date"]),
        name="gold",
    )
    out = out.dropna()
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def load_bitcoin(path: Path | str) -> pd.Series:
    """读取 BCHAIN 比特币数据, 返回按日期索引的价格 Series。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.Series(
        pd.to_numeric(df["Value"], errors="coerce").values,
        index=_parse_mcm_dates(df["Date"]),
        name="bitcoin",
    )
    out = out.dropna()
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def load_market_data(
    raw_dir: Path | str,
    gold_file: str = "LBMA-GOLD.csv",
    bitcoin_file: str = "BCHAIN-MKPRU.csv",
    start: str | None = None,
    end: str | None = None,
) -> MarketData:
    """加载并对齐黄金与比特币数据。

    - 黄金价格先按交易日保留; 之后统一到全日历 (ffill 补齐黄金非交易日),
      以便逐日回测。
    - start/end 用于截取论文区间 (默认 2016-09-11 ~ 2021-09-10)。
    """
    raw_dir = Path(raw_dir)
    gold = load_gold(raw_dir / gold_file)
    bitcoin = load_bitcoin(raw_dir / bitcoin_file)

    if start is not None:
        gold = gold[gold.index >= pd.Timestamp(start)]
        bitcoin = bitcoin[bitcoin.index >= pd.Timestamp(start)]
    if end is not None:
        gold = gold[gold.index <= pd.Timestamp(end)]
        bitcoin = bitcoin[bitcoin.index <= pd.Timestamp(end)]

    gold_trading_days = gold.index

    # 对齐到比特币的全日历 (比特币每日有价), 黄金价格前向填充
    df = pd.DataFrame({"gold": gold, "bitcoin": bitcoin}).sort_index()
    df = df.reindex(bitcoin.index)
    df["gold"] = df["gold"].ffill()
    # 数据开头若黄金缺失则回填首个可得价格
    df["gold"] = df["gold"].bfill()

    return MarketData(df=df, gold_trading_days=gold_trading_days)


def train_test_split(
    data: MarketData, train_ratio: float = 0.7
) -> tuple[MarketData, MarketData]:
    """按时间顺序切分训练集与回测集 (不打乱, 防止未来信息泄漏)。"""
    n = len(data.df)
    cut = int(n * train_ratio)
    train = MarketData(df=data.df.iloc[:cut].copy(), gold_trading_days=data.gold_trading_days)
    test = MarketData(df=data.df.iloc[cut:].copy(), gold_trading_days=data.gold_trading_days)
    return train, test


def daily_returns(prices: pd.Series) -> pd.Series:
    """日收益率 r_t = P_t / P_{t-1} - 1。"""
    return prices.pct_change().dropna()


def rolling_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """滚动波动率 = 滚动日收益率标准差 * sqrt(252) (年化)。"""
    ret = daily_returns(prices)
    return ret.rolling(window).std() * np.sqrt(252)
