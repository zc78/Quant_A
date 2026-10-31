"""
简化版因子测试：跳过分组和 IC 细节，直接输出因子与未来收益的相关性概览
"""
import numpy as np
import pandas as pd
from factors import build_factor_panel
from data_source import get_daily

codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
         "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
start, end = "20200101", "20261029"

# 1. 构建因子面板
print("构建因子面板...")
factors = build_factor_panel(codes, start, end)
print(f"因子数量: {len(factors)}")

# 2. 构建收盘价面板（用于计算未来收益）
print("构建收盘价面板...")
close_panel = {}
for code in codes:
    df = get_daily(code, start, end)
    df['date'] = pd.to_datetime(df['date'])
    close_panel[code] = df.set_index('date')['close']

close_df = pd.DataFrame(close_panel)

# 3. 计算未来20日收益
fwd_ret = close_df.shift(-20) / close_df - 1

# 4. 对每个因子计算简单统计
print("\n========== 因子概览 ==========")
for name, f in factors.items():
    print(f"\n--- {name} ---")
    print(f"  形状: {f.shape}")
    print(f"  非空比例: {f.notna().mean().mean():.1%}")
    
    # 取最后一个完整截面（最近一个交易日）
    last_valid = f.dropna(how='all')
    if last_valid.empty:
        print("  无有效数据，跳过")
        continue
    
    # 计算因子与未来收益的截面 Spearman 相关
    latest_date = last_valid.index[-1]
    factor_today = f.loc[latest_date].dropna()
    fwd_today = fwd_ret.loc[latest_date].reindex(factor_today.index).dropna()
    common = factor_today.index.intersection(fwd_today.index)
    if len(common) < 3:
        print(f"  共同股票太少 ({len(common)}), 跳过")
        continue
    
    from scipy.stats import spearmanr
    corr, pval = spearmanr(factor_today[common], fwd_today[common])
    print(f"  最新截面 ({latest_date.date()}): 股票数={len(common)}, Spearman R={corr:.4f}, p-value={pval:.4f}")
    
    # 计算全历史平均 IC（逐日截面相关）
    ics = []
    for dt in f.index:
        f_day = f.loc[dt].dropna()
        r_day = fwd_ret.loc[dt].reindex(f_day.index).dropna()
        common_day = f_day.index.intersection(r_day.index)
        if len(common_day) >= 3:
            ic, _ = spearmanr(f_day[common_day], r_day[common_day])
            ics.append(ic)
    if ics:
        print(f"  历史平均 IC: {np.mean(ics):.4f} (标准差 {np.std(ics):.4f})")
        print(f"  IC>0 占比: {np.mean(np.array(ics) > 0):.1%}")
    else:
        print("  无法计算历史 IC")