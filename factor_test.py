"""
因子有效性测试脚本
测试内容包括：
- 单因子 IC、ICIR、胜率
- 复合因子 IC、ICIR、胜率
- 多空收益（当前为占位符，后续实现）
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
from factors import build_factor_panel, factor_direction
from data_source import get_daily, get_trade_calendar


def compute_ic(factor_df, forward_returns, method='spearman'):
    """
    计算截面IC
    factor_df: DataFrame, index=date, columns=stock, values=factor
    forward_returns: DataFrame, index=date, columns=stock, values=future return
    method: 'spearman' (秩相关) 或 'pearson'
    返回 Series, index=date, values=IC
    """
    dates = factor_df.index.intersection(forward_returns.index)
    ic_list = []
    for dt in dates:
        f = factor_df.loc[dt].dropna()
        r = forward_returns.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 5:
            continue
        f_vals = f[common]
        r_vals = r[common]
        if method == 'spearman':
            ic = f_vals.corr(r_vals, method='spearman')
        else:
            ic = f_vals.corr(r_vals, method='pearson')
        ic_list.append((dt, ic))
    if not ic_list:
        return pd.Series(dtype=float)
    return pd.DataFrame(ic_list, columns=['date', 'ic']).set_index('date')['ic']


def test_factor(factor_name, factor_df, forward_returns):
    """
    测试单个因子的IC指标
    """
    ic_series = compute_ic(factor_df, forward_returns)
    if len(ic_series) == 0:
        print(f"\n=== 因子 {factor_name} 检验 ===")
        print("无有效IC数据")
        return {
            'name': factor_name,
            'mean_ic': np.nan,
            'icir': np.nan,
            'win_rate': np.nan,
            'long_short_return': 0.0
        }
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = mean_ic / std_ic if std_ic != 0 else 0.0
    win_rate = (ic_series > 0).mean()
    # 多空收益暂为占位符
    long_short_return = 0.0

    print(f"\n=== 因子 {factor_name} 检验（未来{forward_period}日）===")
    print(f"Mean IC : {mean_ic:.4f}")
    print(f"ICIR    : {icir:.4f}")
    print(f"IC胜率  : {win_rate*100:.2f}%")
    print(f"多空累计收益: {long_short_return:.2f}%")

    return {
        'name': factor_name,
        'mean_ic': mean_ic,
        'icir': icir,
        'win_rate': win_rate,
        'long_short_return': long_short_return
    }


if __name__ == "__main__":
    # 参数设置
    codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
             "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
    start = "20200101"
    end = "20261029"
    periods = [5, 10, 20]  # 预测周期（交易日）

    # 构建因子面板
    print("正在构建因子面板...")
    factors = build_factor_panel(codes, start, end)
    print(f"因子面板构建完成，共 {len(factors)} 个因子")

    # 获取收盘价用于计算未来收益
    close_panel = {}
    for code in codes:
        df = get_daily(code, start, end)
        if df.empty:
            continue
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        close_panel[code] = df['close']
    close_df = pd.DataFrame(close_panel)

    # 对每个预测周期进行测试
    for forward_period in periods:
        print(f"\n{'='*60}")
        print(f"预测周期: {forward_period}日")
        print(f"{'='*60}")

        # 计算未来收益（shift(-forward_period) 避免未来函数）
        future_ret = close_df.pct_change(forward_period).shift(-forward_period)

        results = []

        # 1. 测试所有单因子
        for name, panel in factors.items():
            res = test_factor(name, panel, future_ret)
            results.append(res)

        # 2. 构造复合因子并测试
        # 确保所需的子因子都存在
        required = ['volatility_20', 'roe', 'roe_waa', 'or_yoy']
        if all(r in factors for r in required):
            # 标准化后合成（注意：apply 返回 DataFrame，但 zscore 返回 numpy 数组，
            # 需要用 pd.DataFrame 包装并保留索引和列）
            def safe_zscore(df_row):
                """对一行（横截面）做z-score，返回Series"""
                arr = zscore(df_row.values, nan_policy='omit')
                return pd.Series(arr, index=df_row.index)

            vol_z = factors['volatility_20'].apply(safe_zscore, axis=1)
            roe_z = factors['roe'].apply(safe_zscore, axis=1)
            roe_waa_z = factors['roe_waa'].apply(safe_zscore, axis=1)
            or_yoy_z = factors['or_yoy'].apply(safe_zscore, axis=1)

            # 复合因子 = -低波 + ROE + ROE_WAA + 营收增长
            composite = -vol_z + roe_z + roe_waa_z + or_yoy_z
            composite.name = 'composite_lowvol_quality'

            res = test_factor('composite_lowvol_quality', composite, future_ret)
            results.append(res)

            # 也测试你原先提议的版本：vol_z - roe_z - roe_waa_z - or_yoy_z
            composite_alt = vol_z - roe_z - roe_waa_z - or_yoy_z
            composite_alt.name = 'composite_alt'
            res_alt = test_factor('composite_alt', composite_alt, future_ret)
            results.append(res_alt)

        # 打印汇总表格
        print(f"\n\n{'='*60}")
        print(f"预测周期 {forward_period}日 因子有效性汇总")
        print(f"{'='*60}")
        header = f"{'因子':25s} {'MeanIC':>8s} {'ICIR':>8s} {'胜率':>6s}"
        print(header)
        print('-' * 50)
        for r in results:
            name = r['name'][:25]
            mic = f"{r['mean_ic']:.4f}" if not np.isnan(r['mean_ic']) else '  nan'
            icir = f"{r['icir']:.4f}" if not np.isnan(r['icir']) else '  nan'
            wr = f"{r['win_rate']*100:.1f}%" if not np.isnan(r['win_rate']) else ' nan%'
            print(f"{name:25s} {mic:>8s} {icir:>8s} {wr:>6s}")