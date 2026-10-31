import pandas as pd
import numpy as np
from data_source import get_daily, get_daily_basic, get_trade_calendar
from factors import build_factor_panel

codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
         "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
start, end = "20200101", "20261029"

# 1. 交易日历
cal = get_trade_calendar(start, end)
cal = get_trade_calendar(start, end)
td = cal[cal['is_open'] == 1]['cal_date'].astype(str)
td = td[(td >= start) & (td <= end)]
print(f"[1] trade_dates 数量: {len(td)}, 类型: {td.dtype}")
print(f"    首: {td.iloc[0]}, 尾: {td.iloc[-1]}")

# 2. 单只股票日线
df = get_daily(codes[0], start, end)
df['date'] = pd.to_datetime(df['date'])
print(f"\n[2] {codes[0]} 日线形状: {df.shape}, date类型: {df['date'].dtype}")
print(f"    日期范围: {df['date'].min()} ~ {df['date'].max()}")

# 3. reindex 到交易日历
s = df.set_index('date')['close'].reindex(td)
print(f"\n[3] reindex 后非空点数: {s.notna().sum()} / {len(td)}")

# 4. daily_basic
basic = get_daily_basic(codes[0], start, end)
print(f"\n[4] {codes[0]} daily_basic 形状: {basic.shape}")
print(f"    列: {basic.columns.tolist()}")
if not basic.empty:
    print(basic.head(2).to_string())