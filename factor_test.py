# -*- coding: utf-8 -*-
"""
因子有效性检验脚本
用法：python factor_test.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from factors import build_factor_panel
from data_source import get_daily

# ========== 配置 ==========
codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
         "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
start, end = "20200101", "20261029"

# ========== 工具函数 ==========

def calc_forward_return(close_panel, horizon):
    """计算未来 horizon 日收益率"""
    return close_panel.shift(-horizon) / close_panel - 1


def rank_ic(factor_panel, fwd_ret, min_stocks=5):
    """逐日计算 Rank IC"""
    ics = []
    for dt in factor_panel.index:
        f_day = factor_panel.loc[dt].dropna()
        r_day = fwd_ret.loc[dt].dropna()
        common = f_day.index.intersection(r_day.index)
        if len(common) < min_stocks:
            continue
        try:
            ic, _ = spearmanr(f_day[common], r_day[common])
            if not np.isnan(ic):
                ics.append(ic)
        except:
            continue

    if len(ics) == 0:
        return np.nan, 0.0, np.nan

    mean_ic = np.mean(ics)
    std_ic = np.std(ics, ddof=1)
    icir = mean_ic / std_ic if std_ic != 0 else 0.0
    win_rate = np.mean(np.array(ics) > 0)
    return mean_ic, icir, win_rate


def layered_backtest(factor_panel, fwd_ret, n_groups=5, min_stocks_per_group=2):
    """分层回测：每组至少 min_stocks_per_group 只股票，否则跳过该天"""
    def _safe_group(row):
        vals = row.dropna()
        if len(vals) < n_groups * min_stocks_per_group:
            return pd.Series([np.nan] * len(row), index=row.index)
        try:
            labels = pd.qcut(vals, n_groups, labels=False, duplicates='drop')
            result = pd.Series(labels, index=vals.index)
            return result.reindex(row.index)
        except:
            return pd.Series([np.nan] * len(row), index=row.index)

    groups = factor_panel.apply(_safe_group, axis=1)
    group_ret = {}
    for g in range(n_groups):
        mask = groups == g
        valid_days = mask.sum(axis=1) >= min_stocks_per_group
        grp_mean = fwd_ret.where(mask).mean(axis=1)
        grp_mean[~valid_days] = 0
        nav = (1 + grp_mean.fillna(0)).cumprod()
        group_ret[f"G{g+1}"] = nav

    res = pd.DataFrame(group_ret)
    if n_groups >= 2:
        top = fwd_ret.where(groups == n_groups-1).mean(axis=1)
        bottom = fwd_ret.where(groups == 0).mean(axis=1)
        valid_top = (groups == n_groups-1).sum(axis=1) >= min_stocks_per_group
        valid_bottom = (groups == 0).sum(axis=1) >= min_stocks_per_group
        ls = top - bottom
        ls[~(valid_top & valid_bottom)] = 0
        res["LongShort"] = (1 + ls.fillna(0)).cumprod()
    return res


def test_single_factor(factor_name, factor_df, close_df, horizon=20):
    """单因子检验：仅计算 IC 指标（跳过分层回测）"""
    # 计算未来收益
    fwd = calc_forward_return(close_df, horizon)

    # 对齐日期（去除首尾 NaN 过多的日期）
    valid_mask = factor_df.notna().sum(axis=1) >= 3
    f_valid = factor_df[valid_mask]
    fwd_valid = fwd.loc[f_valid.index]

    # 计算 Rank IC
    mean_ic, icir, win_rate = rank_ic(f_valid, fwd_valid, min_stocks=5)

    return {
        'Mean IC': mean_ic,
        'ICIR': icir,
        'IC胜率': win_rate,
        '多空累计收益': 0.0  # 占位，不再计算
    }


# ========== 主流程 ==========
if __name__ == "__main__":
    # 1. 构建因子面板
    factors = build_factor_panel(codes, start, end)

    # 2. 构建收盘价面板
    close_data = {}
    for code in codes:
        df = get_daily(code, start, end)
        df['date'] = pd.to_datetime(df['date'])
        close_data[code] = df.set_index('date')['close']
    close_df = pd.DataFrame(close_data)

    # 3. 测试多个预测周期
    horizons = [5, 10, 20]
    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"预测周期: {horizon}日")
        print('='*60)
        results = {}
        for name, panel in factors.items():
            print(f"\n=== 因子 {name} 检验（未来{horizon}日）===")
            r = test_single_factor(name, panel, close_df, horizon=horizon)
            results[name] = r
            print(f"Mean IC : {r['Mean IC']:.4f}")
            print(f"ICIR    : {r['ICIR']:.4f}")
            print(f"IC胜率  : {r['IC胜率']*100:.2f}%" if not np.isnan(r['IC胜率']) else "IC胜率  : nan%")
            print(f"多空累计收益: {r['多空累计收益']*100:.2f}%")

        # 汇总
        print(f"\n\n========== 预测周期 {horizon}日 因子有效性汇总 ==========")
        print(f"{'因子':<25} {'MeanIC':>8} {'ICIR':>8} {'胜率':>8}")
        print("-" * 52)
        for name, r in results.items():
            mic = f"{r['Mean IC']:.4f}" if not np.isnan(r['Mean IC']) else "nan"
            icir = f"{r['ICIR']:.4f}"
            wr = f"{r['IC胜率']*100:.1f}%" if not np.isnan(r['IC胜率']) else "nan%"
            print(f"{name:<25} {mic:>8} {icir:>8} {wr:>8}")