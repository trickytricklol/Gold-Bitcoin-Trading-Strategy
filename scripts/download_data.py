"""数据下载脚本。

默认从 GitHub 拉取 2021 MCM Problem C 官方数据集 (与论文使用数据一致):
- LBMA-GOLD.csv    黄金每日价格 (USD/troy ounce, 交易日)
- BCHAIN-MKPRU.csv 比特币每日价格 (USD, 每日)

可选: --via yfinance 从 Yahoo Finance 拉取 GC=F 与 BTC-USD 作为替代数据源。

用法:
    python scripts/download_data.py [--out data/raw] [--via github|yfinance]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GOLD_URL = (
    "https://raw.githubusercontent.com/dick20/MCM-ICM/master/"
    "2022%E7%BE%8E%E8%B5%9B%E7%89%B9%E7%AD%89%E5%A5%96/problems/LBMA-GOLD.csv"
)
BITCOIN_URL = (
    "https://raw.githubusercontent.com/dick20/MCM-ICM/master/"
    "2022%E7%BE%8E%E8%B5%9B%E7%89%B9%E7%AD%89%E5%A5%96/problems/BCHAIN-MKPRU.csv"
)


def download_github(out: Path) -> None:
    import urllib.request

    out.mkdir(parents=True, exist_ok=True)
    for name, url in [("LBMA-GOLD.csv", GOLD_URL), ("BCHAIN-MKPRU.csv", BITCOIN_URL)]:
        target = out / name
        print(f"下载 {name} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  已保存 -> {target} ({target.stat().st_size} bytes)")


def download_yfinance(out: Path) -> None:
    import yfinance as yf

    out.mkdir(parents=True, exist_ok=True)
    gold = yf.download("GC=F", start="2016-09-11", end="2021-09-11", progress=False)
    btc = yf.download("BTC-USD", start="2016-09-11", end="2021-09-11", progress=False)

    g = gold["Close"].dropna()
    g.index = g.index.tz_localize(None)
    g.name = "USD (PM)"
    g.to_csv(out / "LBMA-GOLD.csv", header=True)

    b = btc["Close"].dropna()
    b.index = b.index.tz_localize(None)
    b.name = "Value"
    b.to_csv(out / "BCHAIN-MKPRU.csv", header=True)
    print(f"yfinance 数据已保存 -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载黄金/比特币数据")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--via", default="github", choices=["github", "yfinance"])
    args = parser.parse_args()

    out = Path(args.out)
    if args.via == "github":
        download_github(out)
    else:
        download_yfinance(out)


if __name__ == "__main__":
    main()
