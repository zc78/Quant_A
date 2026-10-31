# -*- coding: utf-8 -*-
"""
数据有效性自检脚本
验证范围：
1. token / 接口连通性
2. 交易日历完整性
3. 日线行情：空表、字段、重复、缺失率、价格合理性、复权跳变
4. daily_basic：估值/市值字段完整性
5. 财务数据：公告日不早于报告期（防未来函数）、重复公告日
6. 因子面板拼接前的对齐检查
运行：python validate_data.py
"""
import numpy as np
import pandas as pd

from data_source import (
    pro, get_daily, get_daily_basic,
    get_fina_indicator, get_trade_calendar
)

CODES = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
         "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
START, END = "20200101", "20261029"

# 结果收集
REPORT = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    REPORT.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  -> {detail}" if detail else ""))


def warn(name, detail):
    REPORT.append(("WARN", name, detail))
    print(f"[WARN] {name}  -> {detail}")


# ───────────────────────── 1. 接口连通性 ─────────────────────────
print("\n========== 1. 接口连通性 ==========")
try:
    probe = pro.stock_basic(exchange='SSE', list_status='L', fields='ts_code,name')
    check("tushare 连通 + token 有效", not probe.empty and "ts_code" in probe.columns,
          f"返回 {len(probe)} 只股票基础信息")
except Exception as e:
    check("tushare 连通 + token 有效", False, str(e))
    raise SystemExit("接口不通，先修 token")


# ───────────────────────── 2. 交易日历 ─────────────────────────
print("\n========== 2. 交易日历 ==========")
cal = get_trade_calendar(START, END)
check("日历非空", not cal.empty)
check("日历字段完整", "cal_date" in cal.columns and "is_open" in cal.columns)
open_days = cal[cal["is_open"] == 1]
check("交易日数量合理", len(open_days) > 1000, f"共 {len(open_days)} 个交易日")
# 日历日期不应有重复
check("日历无重复日期", open_days["cal_date"].is_unique,
      f"重复 {open_days['cal_date'].duplicated().sum()} 条")


# ───────────────────────── 3. 日线行情 ─────────────────────────
print("\n========== 3. 日线行情 ==========")
REQUIRED_OHLC = ["date", "open", "high", "low", "close", "volume"]

for code in CODES:
    df = get_daily(code, START, END)
    tag = f"{code} 日线"

    if df.empty:
        check(tag, False, "空表")
        continue

    # 字段完整性
    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        check(tag, False, f"缺字段 {missing}")
        continue

    # 日期处理
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 重复日期
    dup = df["date"].duplicated().sum()
    if dup:
        warn(f"{tag} 重复日期", f"{dup} 条")

    # 缺失率
    na_ratio = df[["open", "high", "low", "close", "volume"]].isna().mean().mean()
    if na_ratio > 0.01:
        warn(f"{tag} 缺失率", f"{na_ratio:.1%}")

    # 价格合理性：close>0，high>=low，high>=close>=low
    bad_price = ((df["close"] <= 0) |
                 (df["high"] < df["low"]) |
                 (df["high"] < df["close"]) |
                 (df["low"] > df["close"])).sum()
    if bad_price:
        warn(f"{tag} 价格异常行", f"{bad_price} 行")

    # 涨跌停粗筛：日涨跌幅绝对值 > 11%（主板10%+容差）视为疑似异常
    ret = df["close"].pct_change()
    extreme = (ret.abs() > 0.11).sum()
    if extreme:
        warn(f"{tag} 极端日收益(>11%)", f"{extreme} 天，需核对是否除权未复权")

    # 复权跳变：相邻收盘价比变化 > 30% 且无公告（粗筛）
    big_jump = (ret.abs() > 0.3).sum()
    if big_jump:
        warn(f"{tag} 疑似除权跳变", f"{big_jump} 天未做复权处理")

    check(tag, True, f"{len(df)} 行, {df['date'].min().date()}~{df['date'].max().date()}")


# ───────────────────────── 4. daily_basic �==========
print("\n========== 4. 每日基本面 daily_basic ==========")
REQUIRED_BASIC = ["pe_ttm", "pb", "turnover_rate", "circ_mv"]

for code in CODES:
    df = get_daily_basic(code, START, END)
    tag = f"{code} daily_basic"

    if df.empty:
        warn(tag, "空表（可能无权限或数据未覆盖）")
        continue

    missing = [c for c in REQUIRED_BASIC if c not in df.columns]
    if missing:
        warn(f"{tag} 缺字段", f"{missing}")
    else:
        # 估值应为正
        neg_pe = (df["pe_ttm"] <= 0).sum()
        if neg_pe:
            warn(f"{tag} PE_TTM<=0", f"{neg_pe} 行（亏损股正常，但因子需处理）")
        check(tag, True, f"{len(df)} 行")
        # 缺失率
        na_ratio = df[REQUIRED_BASIC].isna().mean().mean()
        if na_ratio > 0.05:
            warn(f"{tag} 缺失率", f"{na_ratio:.1%}")


# ───────────────────────── 5. 财务数据 PIT 检查 ─────────────────────────
print("\n========== 5. 财务数据（未来函数防护）==========")
FIN_FIELDS = ["roe", "grossprofit_margin", "netprofit_yoy", "or_yoy"]

for code in CODES:
    try:
        df = get_fina_indicator(code, START, END)
    except Exception as e:
        warn(f"{code} 财务", f"接口异常: {e}")
        continue

    tag = f"{code} fina_indicator"
    if df.empty:
        warn(tag, "空表")
        continue

    # 公告日必须存在且可解析
    if "ann_date" not in df.columns:
        check(tag, False, "缺 ann_date 字段（无法做 PIT 对齐）")
        continue

    df["ann_date"] = pd.to_datetime(df["ann_date"].astype(str), errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"].astype(str), errors="coerce")

    # 未来函数检查：公告日不能早于报告期（逻辑上不可能）
    future_leak = (df["ann_date"] < df["end_date"]).sum()
    if future_leak:
        check(tag, False, f"{future_leak} 行公告日早于报告期（数据异常）")
        continue

    # 重复公告日（同一 ann_date 多行，需去重）
    dup_ann = df["ann_date"].duplicated().sum()
    if dup_ann:
        warn(f"{tag} 重复公告日", f"{dup_ann} 条，需按 end_date 取最新")

    # 负值合理性：roe/grossprofit_margin 是百分比，应在合理区间
    for col in FIN_FIELDS:
        if col in df.columns:
            v = df[col].dropna()
            if len(v) and (v.abs() > 1000).any():
                warn(f"{tag} {col} 异常值", "存在极端值，检查单位（应为百分比）")

    check(tag, True, f"{len(df)} 行, 公告日 {df['ann_date'].min().date()}~{df['ann_date'].max().date()}")


# ───────────────────────── 6. 因子面板对齐检查 ─────────────────────────
print("\n========== 6. 因子面板拼接检查 ==========")
try:
    from factors import build_factor_panel
    factors = build_factor_panel(CODES, START, END)

    check("因子面板非空", len(factors) > 0, f"共 {len(factors)} 个因子")

    for fname, fpanel in factors.items():
        if not isinstance(fpanel, pd.DataFrame):
            warn(f"因子 {fname}", "不是 DataFrame")
            continue
        # 索引应为 datetime
        if not isinstance(fpanel.index, pd.DatetimeIndex):
            warn(f"因子 {fname}", "索引不是 DatetimeIndex")
        # 列应覆盖股票池
        covered = [c for c in CODES if c in fpanel.columns]
        if len(covered) < len(CODES):
            warn(f"因子 {fname}", f"仅覆盖 {len(covered)}/{len(CODES)} 只股票")
        # 非空比例
        fill_ratio = fpanel.notna().mean().mean()
        if fill_ratio < 0.3:
            warn(f"因子 {fname}", f"非空比例仅 {fill_ratio:.1%}")
        else:
            check(f"因子 {fname}", True, f"形状{fpanel.shape}, 非空{fill_ratio:.1%}")
except Exception as e:
    check("因子面板构建", False, str(e))


# ───────────────────────── 汇总 ─────────────────────────
print("\n========== 汇总 ==========")
n_pass = sum(1 for r in REPORT if r[0] == "PASS")
n_warn = sum(1 for r in REPORT if r[0] == "WARN")
n_fail = sum(1 for r in REPORT if r[0] == "FAIL")
print(f"PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}")

if n_fail == 0:
    print("\n✅ 数据链路全部通过，可放心继续因子研究")
else:
    print("\n❌ 存在 FAIL 项，先修复再跑因子测试")
    for status, name, detail in REPORT:
        if status == "FAIL":
            print(f"   - {name}: {detail}")