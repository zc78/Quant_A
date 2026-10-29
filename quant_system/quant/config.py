"""
config.py —— 全局配置
所有可调参数集中在这里，策略层/回测层只读不改。
"""
import os
from pathlib import Path

# ============ 路径 ============
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # 原始落盘（不复权/首次抓取）
FEAT_DIR = DATA_DIR / "features"    # 因子特征
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for _d in (RAW_DIR, FEAT_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============ 数据源 ============
# "sina_tx" 走 akshare 的腾讯接口（前复权，最稳定）；"eastmoney" 为备选
PRICE_SOURCE = "sina_tx"

# ============ 回测区间 ============
START_DATE = "20180101"
END_DATE = "20261026"

# ============ 股票池规则 ============
# 这些规则在数据层应用，保证"当时能买"的语义
MIN_LIST_DAYS = 60          # 上市不满 60 个交易日剔除（次新股）
DROP_ST = True              # 剔除 ST / *ST
DROP_SUSPEND = True         # 剔除停牌
MAX_PRICE = 2000.0          # 超高价股通常流动性异常

# ============ 成本模型（回测用，单位 %）============
COST = {
    "commission": 0.025,    # 双边佣金，万 2.5
    "commission_min": 5.0,  # 最低 5 元（按笔，此处以金额占比近似）
    "stamp_tax": 0.05,      # 卖出印花税（2023-08-28 起减半）
    "transfer_fee": 0.001,  # 过户费，双边
    "slippage": 0.15,       # 滑点，单边，按成交价百分比
}

# ============ 组合约束 ============
PORTFOLIO = {
    "universe": "ZZ500",    # 选股池：HS300 / ZZ500 / ZZ1000 / ALL_A
    "rebalance": "M",       # 调仓频率：M 月频 / W 周频
    "top_n": 20,            # 选前 N 名
    "max_weight": 0.05,     # 单票上限
    "max_turnover": 0.5,    # 单次调仓换手上限
}
