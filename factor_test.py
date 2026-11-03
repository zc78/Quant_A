"""
因子有效性测试脚本（增强版）
功能：
- 单因子IC检验
- 行业中性化
- IC加权复合因子
- 多空收益回测
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
from factors import build_factor_panel
from data_source import get_daily, get_pro, safe_call, clear_cache
import warnings
warnings.filterwarnings('ignore')

# ---------- 行业映射 ----------
INDUSTRY_MAP_CACHE = None

def get_industry_map():
    global INDUSTRY_MAP_CACHE
    if INDUSTRY_MAP_CACHE is not None:
        return INDUSTRY_MAP_CACHE
    pro = get_pro()
    df = safe_call(pro.stock_basic, "stock_basic", fields="ts_code,industry", limit=5000)
    if df is None or df.empty:
        INDUSTRY_MAP_CACHE = {}
    else:
        INDUSTRY_MAP_CACHE = dict(zip(df['ts_code'], df['industry']))
    return INDUSTRY_MAP_CACHE

def industry_neutralize(factor_df, industry_map):
    result = pd.DataFrame(index=factor_df.index, columns=factor_df.columns)
    for dt in factor_df.index:
        row = factor_df.loc[dt].dropna()
        ind = row.index.map(lambda x: industry_map.get(x, 'Unknown'))
        for ind_name in set(ind):
            mask = ind == ind_name
            subset = row[mask]
            if len(subset) >= 3:
                mean = subset.mean()
                std = subset.std()
                if std > 0:
                    result.loc[dt, subset.index] = (subset - mean) / std
    return result.fillna(0)

def compute_ic(factor_df, forward_returns, method='spearman'):
    dates = factor_df.index.intersection(forward_returns.index)
    ic_list = []
    for dt in dates:
        f = factor_df.loc[dt].dropna()
        r = forward_returns.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        f_vals = f[common]
        r_vals = r[common]
        ic = f_vals.corr(r_vals, method=method)
        ic_list.append((dt, ic))
    if not ic_list:
        return pd.Series(dtype=float)
    return pd.DataFrame(ic_list, columns=['date', 'ic']).set_index('date')['ic']

def test_factor(factor_name, factor_df, forward_returns):
    ic_series = compute_ic(factor_df, forward_returns)
    if len(ic_series) == 0:
        print(f"\n=== 因子 {factor_name} 检验 ===")
        print("无有效IC数据")
        return {'name': factor_name, 'mean_ic': np.nan, 'icir': np.nan, 'win_rate': np.nan}
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic != 0 else 0.0
    win_rate = (ic_series > 0).mean()
    print(f"\n=== 因子 {factor_name} 检验（未来{forward_period}日）===")
    print(f"Mean IC : {mean_ic:.4f}")
    print(f"ICIR    : {icir:.4f}")
    print(f"IC胜率  : {win_rate*100:.2f}%")
    return {'name': factor_name, 'mean_ic': mean_ic, 'icir': icir, 'win_rate': win_rate}

def long_short_backtest(factor_df, forward_returns, top_pct=0.2):
    portfolio_ret = []
    dates = factor_df.index.sort_values()
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i+1]
        f = factor_df.loc[dt].dropna().sort_values(ascending=False)
        n = len(f)
        top_n = max(int(n * top_pct), 1)
        bottom_n = top_n
        long_stocks = f.index[:top_n]
        short_stocks = f.index[-bottom_n:]
        ret_long = forward_returns.loc[next_dt, long_stocks].dropna().mean()
        ret_short = forward_returns.loc[next_dt, short_stocks].dropna().mean()
        # 如果任一为空，跳过该期
        if pd.isna(ret_long) or pd.isna(ret_short):
            continue
        portfolio_ret.append(ret_long - ret_short)
    if not portfolio_ret:
        return pd.Series(dtype=float)
    return pd.Series(portfolio_ret, index=dates[:len(portfolio_ret)]).cumsum()

if __name__ == "__main__":
    # 股票池（沪深300前50只）
    codes = [
        "600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000333.SZ",
        "300750.SZ", "002415.SZ", "600276.SH", "000568.SZ", "002714.SZ",
        "600900.SH", "601166.SH", "600887.SH", "000725.SZ", "601398.SH",
        "601288.SH", "600809.SH", "002304.SZ", "601328.SH", "600031.SH",
        "000651.SZ", "002352.SZ", "600585.SH", "601888.SH", "002230.SZ",
        "600309.SH", "300124.SZ", "000002.SZ", "601668.SH", "600690.SH",
        "000001.SZ", "601939.SH", "601857.SH", "600028.SH", "601988.SH",
        "600104.SH", "000538.SZ", "002142.SZ", "600196.SH", "601211.SH",
        "600588.SH", "000063.SZ", "002049.SZ", "300760.SZ", "002371.SZ",
        "600570.SH", "601688.SH", "600436.SH", "000596.SZ", "002410.SZ"
    ]
    start = "20200101"
    end = "20261029"
    periods = [5, 10, 20]

    # 构建因子面板
    print("正在构建因子面板（50只股票）...")
    factors = build_factor_panel(codes, start, end)
    print(f"因子面板构建完成，共 {len(factors)} 个因子")

    # 获取收盘价
    close_panel = {}
    for code in codes:
        df = get_daily(code, start, end)
        if df.empty:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        close_panel[code] = df['close']
    close_df = pd.DataFrame(close_panel)

    # 获取行业映射
    industry_map = get_industry_map()
    print(f"行业映射加载完成，共 {len(industry_map)} 只股票")

    # 对每个预测周期
    for forward_period in periods:
        print(f"\n{'='*70}")
        print(f"预测周期: {forward_period}日")
        print(f"{'='*70}")

        future_ret = close_df.pct_change(forward_period).shift(-forward_period)

        # 1. 单因子检验（原始）
        print("\n--- 原始因子 ---")
        raw_results = []
        for name, panel in factors.items():
            res = test_factor(name, panel, future_ret)
            raw_results.append(res)

        # 2. 行业中性化后的因子
        print("\n--- 行业中性化后 ---")
        neutral_results = []
        for name, panel in factors.items():
            neutral = industry_neutralize(panel, industry_map)
            res = test_factor(f"{name}_neutral", neutral, future_ret)
            neutral_results.append(res)

        # 3. 复合因子（IC加权）
        print("\n--- 复合因子（IC加权）---")
        # 收集各因子的历史IC
        ic_history = {}
        for name, panel in factors.items():
            ic_series = compute_ic(panel, future_ret)
            if len(ic_series) > 20:
                ic_history[name] = ic_series
        if len(ic_history) >= 2:
            weights = []
            names = list(ic_history.keys())
            for name in names:
                ic = ic_history[name].iloc[-252:]  # 最近一年
                w = ic.mean() / ic.std() if ic.std() > 0 else 0
                weights.append(w)
            weights = np.array(weights)
            weights = weights / np.sum(np.abs(weights))
            composite_raw = sum(w * factors[n] for w, n in zip(weights, names))
            composite_neutral = industry_neutralize(composite_raw, industry_map)
            res = test_factor("composite_ic_weighted", composite_neutral, future_ret)
            neutral_results.append(res)

        # 4. 多空收益回测（对最佳因子）
        print("\n--- 多空收益回测（Top20% vs Bottom20%）---")
        best_factor_name = max(neutral_results, key=lambda x: x['icir'] if not np.isnan(x['icir']) else -999)['name']
        print(f"最佳因子: {best_factor_name}")
        # 获取对应的因子面板
        if best_factor_name.startswith('composite'):
            best_panel = composite_neutral
        else:
            orig_name = best_factor_name.replace('_neutral', '')
            best_panel = industry_neutralize(factors[orig_name], industry_map)
        cum_ret = long_short_backtest(best_panel, future_ret, top_pct=0.2)
        print(f"多空累计收益: {cum_ret.iloc[-1]:.2%}" if len(cum_ret) > 0 else "无数据")
        if len(cum_ret) > 0:
            sharpe = cum_ret.pct_change().mean() / cum_ret.pct_change().std() * np.sqrt(252)
            print(f"夏普比率: {sharpe:.2f}")

        # 汇总表
        print(f"\n{'='*70}")
        print(f"预测周期 {forward_period}日 因子有效性汇总")
        print(f"{'='*70}")
        header = f"{'因子':30s} {'MeanIC':>8s} {'ICIR':>8s} {'胜率':>6s}"
        print(header)
        print('-' * 65)
        for r in raw_results + neutral_results:
            name = r['name'][:30]
            mic = f"{r['mean_ic']:.4f}" if not np.isnan(r['mean_ic']) else '  nan'
            icir = f"{r['icir']:.4f}" if not np.isnan(r['icir']) else '  nan'
            wr = f"{r['win_rate']*100:.1f}%" if not np.isnan(r['win_rate']) else ' nan%'
            print(f"{name:30s} {mic:>8s} {icir:>8s} {wr:>6s}")