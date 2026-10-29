"""
大盘择时策略 + 组合回测模块
=====================================
核心逻辑：
  1. 用上证指数 MA250 斜率 + ADX 判断市场状态（牛/震荡/熊）
  2. 牛市：长均线金叉满仓；震荡市：短均线金叉半仓；熊市：空仓
  3. 支持多标的组合回测（等权）

定位：熊市保护策略 —— 减少下跌损失，非稳定 alpha 来源。
回测结论：沪深300样本下超额为正比例55%，平均超额3.66%。
适用场景：持有与大盘同步、高波动标的，判断市场走弱时使用。

数据接口：tushare（带本地CSV缓存，离线优先）
=====================================
"""

import pandas as pd
import numpy as np
from data_source import get_daily, get_index_daily


# ============================================================
# 指标计算
# ============================================================

def calc_adx(df, period=14):
    """
    计算 ADX 指标（平均趋向指数），用于衡量趋势强度。
    返回 Series。
    """
    df = df.copy()
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)),
                   abs(df['low'] - df['close'].shift(1)))
    )
    df['DM_plus'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['DM_minus'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    df['TR_smooth'] = df['TR'].ewm(alpha=1 / period, adjust=False).mean()
    df['DM_plus_smooth'] = df['DM_plus'].ewm(alpha=1 / period, adjust=False).mean()
    df['DM_minus_smooth'] = df['DM_minus'].ewm(alpha=1 / period, adjust=False).mean()
    df['DI_plus'] = 100 * df['DM_plus_smooth'] / df['TR_smooth']
    df['DI_minus'] = 100 * df['DM_minus_smooth'] / df['TR_smooth']
    df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
    df['ADX'] = df['DX'].ewm(alpha=1 / period, adjust=False).mean()
    return df['ADX']


def add_market_state(df_index, window=250):
    """
    给指数日线打市场状态标签。
    条件（按优先级）：
      ADX < 20          → 震荡市
      ADX >= 20 且 MA250斜率 <= 0 → 熊市
      ADX >= 20 且 MA250斜率 > 0  → 牛市
    返回加了 'market_state' 列的 DataFrame。
    """
    df = df_index.copy()
    df['MA250'] = df['close'].rolling(window).mean()
    df['MA250_slope'] = df['MA250'].diff(20)
    df['ADX'] = calc_adx(df, 14)
    conds = [
        df['ADX'] < 20,
        (df['ADX'] >= 20) & (df['MA250_slope'] <= 0),
        (df['ADX'] >= 20) & (df['MA250_slope'] > 0),
    ]
    choices = ['震荡市', '熊市', '牛市']
    df['market_state'] = np.select(conds, choices, default='震荡市')
    return df


# ============================================================
# 单标的回测
# ============================================================

def backtest_single(ts_code, start_date, end_date, osc_weight=0.5):
    """
    对单个标的运行大盘择时策略，返回每日收益序列。

    参数
    ----
    ts_code : str
        标的代码，如 '600519.SH'
    start_date : str
        起始日期 'YYYYMMDD'
    end_date : str
        结束日期 'YYYYMMDD'
    osc_weight : float
        震荡市信号为多头时的仓位比例（默认0.5 = 半仓）

    返回
    ----
    DataFrame，列：date, strategy_return, daily_return
    """
    df_stock = get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_index = get_index_daily(ts_code="000001.SH", start_date=start_date, end_date=end_date)

    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_index['date'] = pd.to_datetime(df_index['date'])

    # 大盘逐日状态
    df_index = add_market_state(df_index)

    # 个股均线
    df_stock['MA5']   = df_stock['close'].rolling(5).mean()
    df_stock['MA10']  = df_stock['close'].rolling(10).mean()
    df_stock['MA60']  = df_stock['close'].rolling(60).mean()
    df_stock['MA120'] = df_stock['close'].rolling(120).mean()

    # 对齐大盘状态
    m = df_index[['date', 'market_state']].rename(columns={'market_state': 'idx_state'})
    df_stock = df_stock.merge(m, on='date', how='left')
    df_stock['idx_state'] = df_stock['idx_state'].ffill().fillna('震荡市')

    # 基础0/1信号
    bull = (df_stock['MA60'] > df_stock['MA120']).astype(int)
    osc  = (df_stock['MA5']  > df_stock['MA10']).astype(int)
    base_signal = np.select(
        [df_stock['idx_state'] == '牛市',
         df_stock['idx_state'] == '震荡市',
         df_stock['idx_state'] == '熊市'],
        [bull, osc, 0],
        default=0
    )

    # 仓位：牛市满仓；震荡市按 osc_weight；熊市空仓
    weight = base_signal.astype(float)
    half_mask = (df_stock['idx_state'] == '震荡市') & (base_signal == 1)
    weight[half_mask] = osc_weight
    df_stock['weight'] = weight

    # 收益计算（单边成本 0.1%）
    cost = 0.001
    df_stock['daily_return'] = df_stock['close'].pct_change()
    weight_change = df_stock['weight'].diff().abs()
    df_stock['strategy_return'] = (
        df_stock['weight'].shift(1) * df_stock['daily_return']
        - weight_change * cost
    )

    return df_stock[['date', 'strategy_return', 'daily_return']].copy()


def print_single(ts_code, start_date, end_date, osc_weight=0.5):
    """单标的回测并打印绩效。"""
    result = backtest_single(ts_code, start_date, end_date, osc_weight)
    if result is None or result.empty:
        print(f"回测失败：{ts_code}")
        return

    cum_strat = (1 + result['strategy_return'].fillna(0)).cumprod()
    cum_bench = (1 + result['daily_return'].fillna(0)).cumprod()

    total  = cum_strat.iloc[-1] - 1
    market = cum_bench.iloc[-1] - 1
    std    = result['strategy_return'].std()
    sharpe = (np.sqrt(252) * result['strategy_return'].mean() / std) if std and std > 1e-12 else 0.0
    mdd    = (cum_strat / cum_strat.cummax() - 1).min()

    print(f"\n=== 择时+半仓震荡 {start_date[:4]}-{end_date[:4]} | {ts_code} ===")
    print(f"震荡市仓位: {osc_weight:.0%}")
    print(f"策略: {total:.2%} | 基准: {market:.2%} | 超额: {total - market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%}")
    print("-" * 70)


# ============================================================
# 组合回测
# ============================================================

def backtest_portfolio(ts_codes, start_date, end_date, osc_weight=0.5,
                       filter_vol=None, vol_window=252):
    """
    多标的等权组合回测，可选波动率筛选。

    参数
    ----
    ts_codes : list
        标的代码列表
    start_date, end_date : str
        回测区间
    osc_weight : float
        震荡市仓位比例
    filter_vol : float or None
        若非 None，只保留年化波动率 > filter_vol 的标的
    vol_window : int
        计算波动率的窗口（默认252交易日）

    返回
    ----
    None（打印结果）
    """
    # ---- 可选：波动率筛选 ----
    selected_codes = ts_codes
    if filter_vol is not None:
        print(f"=== 波动率筛选（阈值 {filter_vol:.0%}）===")
        vols = {}
        for code in ts_codes:
            try:
                df = get_daily(ts_code=code, start_date=start_date, end_date=end_date)
                ret = df['close'].pct_change().dropna()
                vols[code] = ret.std() * np.sqrt(252)
                print(f"  {code}: 年化波动率 = {vols[code]:.2%}")
            except Exception as e:
                print(f"  {code}: 失败 - {e}")
        selected_codes = [c for c in ts_codes if vols.get(c, 0) > filter_vol]
        rejected = [c for c in ts_codes if c not in selected_codes]
        print(f"入选({len(selected_codes)}只): {selected_codes}")
        print(f"剔除({len(rejected)}只): {rejected}")
        if not selected_codes:
            print("没有标的通过筛选")
            return

    # ---- 逐标的回测 ----
    all_returns = []
    bench_list = []
    single_rows = []

    for code in selected_codes:
        try:
            result = backtest_single(code, start_date, end_date, osc_weight)
            if result is not None and len(result) > 0:
                cum_strat = (1 + result['strategy_return'].fillna(0)).cumprod()
                cum_bench = (1 + result['daily_return'].fillna(0)).cumprod()
                total_single = cum_strat.iloc[-1] - 1
                bench_single = cum_bench.iloc[-1] - 1
                single_rows.append({
                    '代码': code,
                    '收益': total_single,
                    '基准': bench_single,
                    '超额': total_single - bench_single,
                    '回撤': (cum_strat / cum_strat.cummax() - 1).min()
                })
                all_returns.append(result.set_index('date')['strategy_return'])
                bench_list.append(result.set_index('date')['daily_return'])
        except Exception as e:
            print(f"  {code} 处理失败: {e}")

    if not all_returns:
        print("没有成功回测的标的")
        return

    # 各标的绩效表
    df_rows = pd.DataFrame(single_rows)
    print(f"\n{'代码':<12}{'收益':>9}{'基准':>9}{'超额':>9}{'回撤':>9}")
    print("-" * 50)
    for _, r in df_rows.iterrows():
        print(f"{r['代码']:<12}{r['收益']:>8.2%}{r['基准']:>8.2%}{r['超额']:>8.2%}{r['回撤']:>8.2%}")

    # ---- 等权组合 ----
    port_ret = pd.concat(all_returns, axis=1).mean(axis=1)
    bench_ret = pd.concat(bench_list, axis=1).mean(axis=1)
    cum_port = (1 + port_ret.fillna(0)).cumprod()
    cum_bench = (1 + bench_ret.fillna(0)).cumprod()

    total = cum_port.iloc[-1] - 1
    market = cum_bench.iloc[-1] - 1
    std = port_ret.std()
    sharpe = (np.sqrt(252) * port_ret.mean() / std) if std and std > 1e-12 else 0.0
    mdd = (cum_port / cum_port.cummax() - 1).min()

    print(f"\n=== 组合回测 {start_date[:4]}-{end_date[:4]} ===")
    print(f"标的数: {len(selected_codes)}")
    print(f"组合收益: {total:.2%} | 等权持有: {market:.2%} | 超额: {total - market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%}")
    print("=" * 70)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    codes = [
        "600519.SH",  # 茅台
        "000858.SZ",  # 五粮液
        "601318.SH",  # 中国平安
        "600036.SH",  # 招商银行
        "000333.SZ",  # 美的集团
        "300750.SZ",  # 宁德时代
        "002415.SZ",  # 海康威视
        "600276.SH",  # 恒瑞医药
    ]

    # 单标的示例
    # print_single("600519.SH", "20200101", "20261029")

    # 组合回测（不带筛选）
    backtest_portfolio(codes, "20200101", "20261029", osc_weight=0.5)

    # 组合回测（带波动率筛选）
    # backtest_portfolio(codes, "20200101", "20261029", osc_weight=0.5, filter_vol=0.30)