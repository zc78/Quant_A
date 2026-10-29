"""
单因子检验：Rank IC + 分层回测
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calc_forward_return(close_panel, horizon=20):
    """计算未来 N 日收益率（标签）"""
    fwd = close_panel.shift(-horizon) / close_panel - 1
    return fwd


def rank_ic(factor, fwd_ret):
    """每日截面 Rank IC"""
    dates = factor.index
    ics = []
    for d in dates:
        f = factor.loc[d].dropna()
        r = fwd_ret.loc[d].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            ics.append(np.nan)
            continue
        ic = spearmanr(f[common].rank(), r[common].rank()).correlation
        ics.append(ic)
    return pd.Series(ics, index=dates, name="IC")


def ic_stats(ic_series):
    """IC 统计：均值、IR、胜率"""
    ic = ic_series.dropna()
    mean_ic = ic.mean()
    ic_ir = mean_ic / ic.std() if ic.std() > 0 else 0
    win_rate = (ic > 0).mean()
    return {"mean_ic": mean_ic, "ic_ir": ic_ir, "win_rate": win_rate,
            "ic_std": ic.std()}


def layered_backtest(factor, fwd_ret, n_groups=5):
    """
    分层回测：按因子值分 n 组，计算各组未来收益
    返回每组累计净值 + 多空（G_top - G_bottom）
    """
    # 截面排名分组
    rank = factor.rank(axis=1, pct=True)
    group_labels = pd.cut(rank, bins=np.linspace(0, 1, n_groups + 1),
                          labels=range(1, n_groups + 1), include_lowest=True)

    group_ret = {}
    for g in range(1, n_groups + 1):
        # 每期该组股票的平均未来收益
        mask = group_labels == g
        grp_mean = (fwd_ret.where(mask)).mean(axis=1)
        nav = (1 + grp_mean.fillna(0)).cumprod()
        group_ret[f"G{g}"] = nav

    res = pd.DataFrame(group_ret)
    # 多空组合：最高组 - 最低组
    if n_groups >= 2:
        ls = (fwd_ret.where(group_labels == n_groups).mean(axis=1) -
              fwd_ret.where(group_labels == 1).mean(axis=1))
        res["LongShort"] = (1 + ls.fillna(0)).cumprod()
    return res


def test_single_factor(name, factor, close_panel, horizon=20):
    """单因子完整测试，打印结果"""
    fwd = calc_forward_return(close_panel, horizon)
    ic = rank_ic(factor, fwd)
    stats = ic_stats(ic)
    layers = layered_backtest(factor, fwd, 5)

    print(f"\n=== 因子 {name} 检验（未来{horizon}日）===")
    print(f"Mean IC : {stats['mean_ic']:.4f}")
    print(f"ICIR    : {stats['ic_ir']:.4f}")
    print(f"IC胜率  : {stats['win_rate']:.1%}")
    ls_ret = layers['LongShort'].iloc[-1] - 1
    print(f"多空累计收益: {ls_ret:.2%}")
    return {"name": name, "stats": stats, "layers": layers, "ic": ic}


if __name__ == "__main__":
    from factors import build_factor_panel, factor_direction
    from data_source import get_daily

    codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
             "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
    start, end = "20200101", "20261029"

    factors = build_factor_panel(codes, start, end)
    direction = factor_direction()

    # 构造收盘价面板（供算未来收益）
    close_panel = factors['momentum_20'].copy()
    for i, code in enumerate(codes):
        df = get_daily(code, start, end)
        df['date'] = pd.to_datetime(df['date'])
        close_panel[code] = df.set_index('date')['close'].reindex(close_panel.index)

    # 逐个因子测试
    results = []
    for name, f in factors.items():
        # 方向统一：按预设方向翻转
        d = direction.get(name, 0)
        if d == -1:
            f = -f
        # 清洗
        from factor_clean import clean_factor
        f_clean = clean_factor(f)
        r = test_single_factor(name, f_clean, close_panel, horizon=20)
        results.append(r)

    # 汇总
    print("\n\n========== 因子有效性汇总 ==========")
    print(f"{'因子':<20}{'MeanIC':>10}{'ICIR':>10}{'胜率':>10}{'多空收益':>12}")
    print("-" * 62)
    for r in results:
        s = r['stats']
        ls = r['layers']['LongShort'].iloc[-1] - 1
        print(f"{r['name']:<20}{s['mean_ic']:>10.4f}{s['ic_ir']:>10.4f}"
              f"{s['win_rate']:>10.1%}{ls:>12.2%}")