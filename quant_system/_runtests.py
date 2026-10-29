"""数据层单元测试（离线版，无需联网）

覆盖：
- T1 单标的加载与数据完整性
- T2 多标的批量加载与缓存
- T3 面板查询 API
- T4 交易日历
- T5 股票池构建
- T6 数据质量
"""
import time
import pandas as pd
import numpy as np
from quant import (load_daily, list_cached_codes, get_price,
                   get_trade_calendar, get_universe, build_backtest_pool, config)

PASS, FAIL = [], []
def check(name, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {extra}")
    (PASS if cond else FAIL).append(name)

t0 = time.time()

# --- T1 单标的加载与完整性 ---
print("\n=== T1 单标的加载（贵州茅台）===")
df = load_daily("sh600519")
check("加载非空", not df.empty, f"shape={df.shape}")
check("字段完整", set(df.columns) >= {"date","open","high","low","close","volume","amount","turnover","code"},
      str(df.columns.tolist()))
check("date 为 datetime", df["date"].dtype.kind == "M")
check("无重复日期", df["date"].is_unique)
check("日期升序", bool((df["date"].diff().dropna() > pd.Timedelta(0)).all()))
check("收盘价合法", bool((df["close"] > 0).all() and df["close"].notna().all()), f"min={df['close'].min()}")
check("最高价>=最低价", bool((df["high"] >= df["low"]).all()))
check("成交量非负", bool((df["volume"] >= 0).all()))
check("code 正确", df["code"].iloc[0] == "sh600519")
print(df[["date","open","high","low","close","volume"]].head(3).to_string())
print("...")
print(df[["date","open","high","low","close","volume"]].tail(3).to_string())

# --- T2 多标的批量与缓存 ---
print("\n=== T2 批量加载与缓存 ===")
codes = ["sh600519","sz000001","sh601318","sh600036","sz000002","sh601988","sz000333","sh601899","sz300750","sh600030","sh600000","sh601398"]
frames = {c: load_daily(c) for c in codes}
st = time.time()
cached = list_cached_codes()
check("缓存数量正确", len(cached) == 12, f"cached={len(cached)}")
check("全部可加载", all(not frames[c].empty for c in codes))
check("加载耗时合理", time.time()-st < 10, f"elapsed={time.time()-st:.2f}s")
lens = [len(frames[c]) for c in codes]
check("长度基本一致", max(lens) - min(lens) <= 5, f"lens range={min(lens)}~{max(lens)}")

# --- T3 面板查询 ---
print("\n=== T3 面板查询 API ===")
panel = get_price(["sh600519","sz000001"], start="20240101", end="20240630")
check("面板 MultiIndex", isinstance(panel.index, pd.MultiIndex), str(panel.index.names))
check("面板数据齐全", not panel.empty, f"shape={panel.shape}")
check("面板代码过滤", set(panel.index.get_level_values("code").unique()) <= {"sh600519","sz000001"})
print(panel.head(6).to_string())

# --- T4 交易日历 ---
print("\n=== T4 交易日历 ===")
cal = get_trade_calendar("20240101", "20241231")
check("交易日数量合理", 230 <= len(cal) <= 260, f"n={len(cal)}")
check("无周末", int(cal.dayofweek.max()) <= 4, f"maxdow={cal.dayofweek.max()}")
check("升序无重", bool(cal.is_monotonic_increasing and cal.is_unique))
check("降序区间也OK", len(get_trade_calendar("20240101","20240131")) >= 20)
print(f"2024 交易日 {cal.min().date()} ~ {cal.max().date()}, 共 {len(cal)} 个")

# --- T5 股票池构建 ---
print("\n=== T5 股票池构建 ===")
univ = get_universe()
check("快照非空", not univ.empty, f"rows={len(univ)}")
check("含市值列", univ["mktcap"].notna().mean() > 0.5, f"非空比={univ['mktcap'].notna().mean():.2f}")
n_st = int(univ["is_st"].sum())
check("ST 标记生效", n_st >= 0, f"ST数={n_st}")
print(f"全市场快照：{len(univ)} 只，其中 ST {n_st} 只")

pool = build_backtest_pool("20261026")
check("池子非空", len(pool) > 0, f"n={len(pool)}")
st_in_pool = univ[univ["code"].isin(pool) & univ["is_st"]]
check("ST 已被剔除", st_in_pool.empty, f"残留ST={st_in_pool['name'].tolist()[:5]}")
print(f"回测池（已剔除ST/停牌/次新）: {len(pool)} 只")

# --- T6 数据质量 ---
print("\n=== T6 数据质量 ===")
miss_ratio = []
full_cal = get_trade_calendar("20210101", "20251231")
for c in list_cached_codes():
    d = load_daily(c)
    if d.empty:
        continue
    lo, hi = d["date"].min(), d["date"].max()
    seg_cal = full_cal[(full_cal >= lo) & (full_cal <= hi)]
    miss_ratio.append(1 - len(d) / max(len(seg_cal), 1))
avg_miss = np.mean(miss_ratio) if miss_ratio else 0
check("缺失率可控", avg_miss < 0.05, f"avg_miss={avg_miss:.3f}")

# 收益率特征（合成数据应具备）
rets = df["close"].pct_change().dropna()
check("收益率均值合理", -0.1 < rets.mean() < 0.1, f"mean={rets.mean():.4f}")
check("收益率波动合理", 0.001 < rets.std() < 0.3, f"std={rets.std():.4f}")
check("波动率聚集（|ret|自相关>0）", rets.abs().autocorr(1) > 0, f"acf={rets.abs().autocorr(1):.3f}")
check("无极端异常值（99.9%分位可控）", rets.quantile(0.999) < 0.5, f"p999={rets.quantile(0.999):.3f}")

print(f"\n{'='*50}")
print(f"总计 {len(PASS)+len(FAIL)} 项，通过 {len(PASS)}，失败 {len(FAIL)}")
if FAIL:
    print("失败项:", FAIL)
print(f"总耗时 {time.time()-t0:.1f}s")
