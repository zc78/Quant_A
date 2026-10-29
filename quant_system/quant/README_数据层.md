# 数据层使用说明

## 目录结构

```
quant/
├── config.py                  # 路径与参数集中管理
├── README_数据层.md
├── data/
│   ├── raw/                   # 原始行情（未复权，网络拉取落盘）
│   ├── processed/             # 清洗后（复权价、停牌补全、去极值）
│   │   ├── all_daily.parquet  # 全市场面板表（回测直接读这个）
│   │   └── stock_list.parquet # 股票清单
│   └── factor_store/          # 因子值（下一阶段）
├── scripts/
│   ├── data_source.py         # 数据源：akshare + tushare 双源冗余
│   ├── clean.py               # 清洗：复权/补全/去极值/剔除
│   ├── build_dataset.py       # 主流程：批量下载→清洗→合并
│   └── _seed_mock.py          # 沙盒内模拟数据（本地可忽略）
├── strategies/                # 策略层（下一阶段）
├── backtest/                  # 回测层（下一阶段）
└── logs/
```

## 快速开始（本地有网环境）

### 1. 安装依赖

```bash
pip install pandas numpy pyarrow akshare tushare
```

### 2. 批量下载全 A 行情

```bash
cd quant
# 单线程（稳定）
python scripts/build_dataset.py --start 20180101 --end 20261026

# 多线程加速（akshare 有频率限制，建议 ≤4）
python scripts/build_dataset.py --start 20180101 --end 20261026 --workers 4
```

支持**断点续跑**：已拉取的股票自动跳过，中断后重跑即可。

### 3. 只合并已有数据（跳过下载）

```bash
python scripts/build_dataset.py --merge-only
```

### 4. 验证清洗效果

```bash
python scripts/clean.py
```

## 回测时怎么读数据

```python
import pandas as pd
df = pd.read_parquet("data/processed/all_daily.parquet")

# 取某只股票
pingan = df[df["symbol"] == "000001"].sort_values("trade_date")

# 取某个交易日全市场截面
cross = df[df["trade_date"] == "2025-06-01"]

# 按日期对齐的宽表（行=交易日，列=股票）
wide = df.pivot(index="trade_date", columns="symbol", values="close")
```

## 下一步：因子层

有了 `all_daily.parquet`，就可以写 `scripts/factor.py`：

- 价值因子：EP = 1/PE、BP = 1/PB（需合并财务数据）
- 动量因子：20日收益率 `close.pct_change(20)`
- 波动率因子：20日收益率标准差
- 截面标准化 → 行业/市值中性化 → IC 检验
