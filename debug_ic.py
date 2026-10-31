"""
调试：检查因子面板与未来收益面板的对齐情况
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from factors import build_factor_panel
from data_source import get_daily

codes = ["600519.SH","000858.SZ","601318.SH","600036.SH",
         "000333.SZ","300750.SZ","002415.SZ","600276.SH"]
start, end = "20200101", "20261029"

# 构建因子面板
factors = build_factor_panel(codes, start, end)

# 构建收盘价面板
close_data = {}
for code in codes:
    df = get_daily(code, start, end)
    df['date'] = pd.to_datetime(df['date'])
    close_data[code] = df.set_index('date')['close']
close_df = pd.DataFrame(close_data)

# 计算未来20日收益
fwd = close_df.shift(-20) / close_df - 1

# 选择一个因子，例如 momentum_20
factor_name = 'momentum_20'
f = factors[factor_name]
print(f"因子 {factor_name} 的形状: {f.shape}")
print(f"因子非空比例: {f.notna().mean().mean():.1%}")
print(f"未来收益形状: {fwd.shape}")
print(f"未来收益非空比例: {fwd.notna().mean().mean():.1%}")

# 取一天为例，检查对齐
sample_date = f.dropna(how='all').index[0]  # 第一个有数据的日期
print(f"\n样本日期: {sample_date}")
print(f"该日因子值: {f.loc[sample_date].dropna().to_dict()}")
print(f"该日未来收益: {fwd.loc[sample_date].dropna().to_dict()}")

# 计算该日的 IC
f_day = f.loc[sample_date].dropna()
r_day = fwd.loc[sample_date].reindex(f_day.index).dropna()
common = f_day.index.intersection(r_day.index)
print(f"共同股票数: {len(common)}")
if len(common) >= 3:
    ic, pval = spearmanr(f_day[common], r_day[common])
    print(f"该日 IC: {ic:.4f}, p-value: {pval:.4f}")
else:
    print("共同股票不足3只，无法计算IC")

# 遍历所有日期，统计有多少天能满足至少5只股票的共同交集
count_enough = 0
total_days = 0
for dt in f.index:
    f_day = f.loc[dt].dropna()
    r_day = fwd.loc[dt].reindex(f_day.index).dropna()
    common = f_day.index.intersection(r_day.index)
    if len(common) >= 5:
        count_enough += 1
    total_days += 1
print(f"\n总天数: {total_days}, 满足>=5只共同股票的天数: {count_enough}")