"""
data_seed.py —— 内置可复现样本数据集（离线可用）

说明：
  沙盒/离线环境下外部行情接口可能不可用。为保证你下载后无需联网即可
  跑通 数据层→因子层→策略层→回测层 全流程，本模块用固定随机种子
  生成一批"形似真实A股"的日线数据，具备：
    - 真实的收益率肥尾 / 波动率聚集特征
    - 与真实A股量级一致的 price / volume / amount
    - 可复现：SEED 固定，任何人运行结果一致
  数据写入 data/raw/daily/*.csv，结构与 fetch_history 落盘完全一致。

  【重要】这是教学/联调用的合成数据，不可用于实盘决策。
  当你在有真实数据源的机器上运行时，用 fetch_history + update_universe
  抓取的实盘数据会自动覆盖并优先被加载（见 data.py 的 load_daily）。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from . import config
from .logger import get_logger

log = get_logger("data_seed")

SEED = 20261026
N_DAYS = 1200          # 约 5 年交易日
N_ASSETS = 12
START = pd.Timestamp("2021-01-04")

# 12 只合成标的，market+code 与真实A股编码一致
UNIVERSE = [
    ("sh600000", "浦发银行",  9.50),
    ("sh600036", "招商银行", 38.00),
    ("sh601318", "中国平安", 55.00),
    ("sh600519", "贵州茅台",1680.00),
    ("sz000001", "平安银行", 13.00),
    ("sz000002", "万科A",   18.00),
    ("sh601988", "中国银行", 3.40),
    ("sh601398", "工商银行", 5.20),
    ("sh600030", "中信证券", 25.00),
    ("sz300750", "宁德时代", 210.00),
    ("sh601899", "紫金矿业", 12.00),
    ("sz000333", "美的集团", 70.00),
]

# 特征参数：模拟真实A股的年化收益/波动差异（日度参数，mu=年化/242, vol=年化/sqrt(242)）
PARAMS = {
    "浦发银行":  (0.00002, 0.014),
    "招商银行":  (0.00005, 0.016),
    "中国平安":  (0.00003, 0.019),
    "贵州茅台":  (0.00008, 0.018),
    "平安银行":  (0.00002, 0.019),
    "万科A":     (-0.00004, 0.022),
    "中国银行":  (0.00002, 0.012),
    "工商银行":  (0.00003, 0.011),
    "中信证券":  (0.00004, 0.022),
    "宁德时代":  (0.00010, 0.029),
    "紫金矿业":  (0.00006, 0.024),
    "美的集团":  (0.00005, 0.017),
}


def _make_calendar(start: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """生成近似A股交易日历（剔除周末，含极少量模拟节假日，可复现）。"""
    all_days = pd.bdate_range(start, periods=int(n * 1.55))
    rng = np.random.default_rng(SEED)
    # 用确定性规则模拟节假日：每 21 个工作日固定剔除 1 天，避免依赖全局随机状态
    drop_idx = list(range(20, len(all_days), 21))
    cal = all_days[~np.isin(np.arange(len(all_days)), drop_idx)]
    return cal[:n]


def _gbm_path(mu: float, vol: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """带波动率聚集的几何布朗运动，模拟真实收益率序列。"""
    dt = 1.0 / 242
    # 用 GARCH(1,1) 风味调制波动率
    base = vol * np.sqrt(dt)
    vol_path = np.empty(n)
    vol_path[0] = base
    rets = np.empty(n)
    for t in range(n):
        z = rng.standard_normal()
        rets[t] = mu * dt + vol_path[t] * z
        # 波动率均值回归 + 聚集
        vol_path[t + 1] if t < n - 1 else None
        nv = 0.94 * vol_path[t] + 0.04 * base + 0.02 * abs(rets[t]) * 20
        if t < n - 1:
            vol_path[t + 1] = nv
    return rets


def build_seed(force: bool = False) -> int:
    """生成合成日线并落盘。返回生成的文件数。返回 0 表示已有数据且非强制。"""
    out_dir = config.RAW_DIR / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.glob("*.csv")):
        log.info("已存在落盘数据，跳过生成（用 force=True 强制重建）")
        return 0

    rng = np.random.default_rng(SEED)
    cal = _make_calendar(START, N_DAYS)
    dates = cal
    n = len(dates)
    np.seterr(over="warn")
    written = 0

    for code, name, price0 in UNIVERSE:
        mu, vol = PARAMS[name]
        rets = np.clip(_gbm_path(mu, vol, n, rng), -0.15, 0.15)
        close = price0 * np.exp(np.clip(np.cumsum(rets), -40, 40))
        # 日内高低开
        intraday = np.abs(rng.standard_normal(n)) * vol * np.sqrt(1 / 242) * 1.6
        open_ = close / np.exp(rets) * (1 + rng.standard_normal(n) * 0.002)
        high = np.maximum(open_, close) * (1 + intraday * 0.5)
        low = np.minimum(open_, close) * (1 - intraday * 0.5)
        # 成交量：与价格负相关（金额量级趋同）+ 聚集
        base_vol = 5e6 / (price0 ** 0.3)
        vol_shock = np.exp(np.abs(rng.standard_normal(n)) * 0.8)
        volume = (base_vol * vol_shock * (0.6 + 0.4 * np.abs(rets) * 30)).astype(np.float64)
        amount = volume * close
        turnover = np.clip(rng.beta(1.2, 8, n) * 8, 0.1, 25)

        df = pd.DataFrame({
            "date": dates,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "volume": np.round(volume, 0),
            "amount": np.round(amount, 2),
            "turnover": np.round(turnover, 2),
            "code": code,
        })
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(out_dir / f"{code}.csv", index=False, encoding="utf-8-sig")
        written += 1

    # 落盘交易日历
    pd.DataFrame({"date": dates}).to_csv(
        config.RAW_DIR / "trade_calendar.csv", index=False, encoding="utf-8-sig")
    log.info(f"生成 {written} 只合成标的，{n} 个交易日 → {out_dir}")
    return written


if __name__ == "__main__":
    print("生成中...")
    n = build_seed(force=True)
    print(f"完成，写入 {n} 个文件")
