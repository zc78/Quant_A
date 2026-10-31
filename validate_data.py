"""
数据校验脚本：检查接口连通性、数据完整性、字段名正确性
"""
import pandas as pd
from data_source import (
    get_daily, get_daily_basic, get_fina_indicator,
    get_trade_calendar, clear_cache, pro
)

TS_CODES = [
    "600519.SH", "000858.SZ", "601318.SH", "600036.SH",
    "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"
]
START = "20200101"
END = "20261029"

pass_count = 0
warn_count = 0
fail_count = 0

def check(name, condition, detail=""):
    global pass_count, warn_count, fail_count
    if condition:
        print(f"[PASS] {name}  -> {detail}")
        pass_count += 1
    else:
        print(f"[FAIL] {name}  -> {detail}")
        fail_count += 1

def warn(name, detail=""):
    global warn_count
    print(f"[WARN] {name}  -> {detail}")
    warn_count += 1

print("\n========== 1. 接口连通性 ==========")
try:
    df_stock = pro.stock_basic(exchange='SSE', list_status='L',
                               fields='ts_code,symbol,name,area,industry,list_date')
    check("tushare 连通 + token 有效", len(df_stock) > 100,
          f"返回 {len(df_stock)} 只股票基础信息")
except Exception as e:
    check("tushare 连通 + token 有效", False, str(e))

print("\n========== 2. 交易日历 ==========")
cal = get_trade_calendar(START, END)
check("日历非空", not cal.empty, f"共 {len(cal)} 条")
check("日历字段完整", {'cal_date', 'is_open', 'exchange'}.issubset(cal.columns),
      f"字段: {list(cal.columns)}")
# 检查交易日数量（2020-2026 约 1700 个交易日）
cal_open = cal[cal['is_open'] == 1]
n_trade = len(cal_open)
check("交易日数量合理", 1500 <= n_trade <= 2000,
      f"共 {n_trade} 个交易日")
check("日历无重复日期", cal['cal_date'].duplicated().sum() == 0,
      f"重复 {cal['cal_date'].duplicated().sum()} 条")

print("\n========== 3. 日线行情 ==========")
required_daily = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
for code in TS_CODES:
    df = get_daily(code, START, END)
    if df.empty:
        check(f"{code} 日线", False, "空表")
        continue
    missing = [c for c in required_daily if c not in df.columns]
    if missing:
        check(f"{code} 日线", False, f"缺字段 {missing}")
    else:
        check(f"{code} 日线", True, f"{len(df)} 行")

print("\n========== 4. 每日基本面 daily_basic ==========")
for code in TS_CODES:
    df = get_daily_basic(code, START, END)
    if df.empty:
        warn(f"{code} daily_basic", "空表（可能无权限或数据未覆盖）")
        continue
    check(f"{code} daily_basic", len(df) > 400, f"{len(df)} 行")

print("\n========== 5. 财务数据（未来函数防护）==========")
for code in TS_CODES:
    df = get_fina_indicator(code, START, END)
    if df.empty:
        warn(f"{code} fina_indicator", "空表")
        continue
    dup = df['ann_date'].duplicated().sum()
    if dup > 0:
        warn(f"{code} fina_indicator 重复公告日", f"{dup} 条，需按 end_date 取最新")
    else:
        check(f"{code} fina_indicator 重复公告日", True, "无重复")
    check(f"{code} fina_indicator", len(df) >= 5,
          f"{len(df)} 行, 公告日 {df['ann_date'].min()}~{df['ann_date'].max()}")

print("\n========== 6. 因子面板拼接检查 ==========")
try:
    from factors import build_factor_panel
    factors = build_factor_panel(TS_CODES, START, END)
    check("因子面板构建", len(factors) >= 8, f"生成 {len(factors)} 个因子")
except Exception as e:
    check("因子面板构建", False, str(e))

print("\n========== 汇总 ==========")
total = pass_count + warn_count + fail_count
print(f"PASS: {pass_count}  WARN: {warn_count}  FAIL: {fail_count}")

if fail_count > 0:
    print("\n❌ 存在 FAIL 项，先修复再跑因子测试")
else:
    print("\n✅ 全部通过，可以继续因子测试")