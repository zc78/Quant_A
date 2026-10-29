import pandas as pd
import numpy as np
from data_source import get_daily, get_index_daily

def calculate_adx(df, period=14):
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
    df['TR_smooth'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    df['DM_plus_smooth'] = df['DM_plus'].ewm(alpha=1/period, adjust=False).mean()
    df['DM_minus_smooth'] = df['DM_minus'].ewm(alpha=1/period, adjust=False).mean()
    df['DI_plus'] = 100 * df['DM_plus_smooth'] / df['TR_smooth']
    df['DI_minus'] = 100 * df['DM_minus_smooth'] / df['TR_smooth']
    df['DX'] = 100 * abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    return df['ADX']

def add_market_state(df, window=250):
    df = df.copy()
    df['MA250'] = df['close'].rolling(window).mean()
    df['MA250_slope'] = df['MA250'].diff(20)
    df['ADX'] = calculate_adx(df, 14)
    conds = [
        df['ADX'] < 20,
        (df['ADX'] >= 20) & (df['MA250_slope'] <= 0),
        (df['ADX'] >= 20) & (df['MA250_slope'] > 0)
    ]
    choices = ['震荡市', '熊市', '牛市']
    df['market_state'] = np.select(conds, choices, default='震荡市')
    return df

def _single_stock_backtest(ts_code, start_date, end_date):
    """
    对单个标的运行择时策略，返回每日收益率 DataFrame。
    返回列：date, strategy_return, daily_return
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

    # 基础 0/1 信号
    bull  = (df_stock['MA60'] > df_stock['MA120']).astype(int)
    osc   = (df_stock['MA5']  > df_stock['MA10']).astype(int)
    base_signal = np.select(
        [df_stock['idx_state'] == '牛市',
         df_stock['idx_state'] == '震荡市',
         df_stock['idx_state'] == '熊市'],
        [bull, osc, 0],
        default=0
    )

    # 震荡市半仓
    df_stock['weight'] = base_signal.astype(float)
    half_mask = (df_stock['idx_state'] == '震荡市') & (base_signal == 1)
    df_stock.loc[half_mask, 'weight'] = 0.5

    # 收益计算
    cost = 0.001
    df_stock['daily_return'] = df_stock['close'].pct_change()
    weight_change = df_stock['weight'].diff().abs()
    df_stock['strategy_return'] = df_stock['weight'].shift(1) * df_stock['daily_return'] - weight_change * cost

    # 返回所需列
    result = df_stock[['date', 'strategy_return', 'daily_return']].copy()
    return result

def backtest_with_filter(ts_code, start_date, end_date):
    df_stock = get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_index = get_index_daily(ts_code="000001.SH", start_date=start_date, end_date=end_date)

    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_index['date'] = pd.to_datetime(df_index['date'])

    df_index = add_market_state(df_index)

    df_stock['MA5']   = df_stock['close'].rolling(5).mean()
    df_stock['MA10']  = df_stock['close'].rolling(10).mean()
    df_stock['MA60']  = df_stock['close'].rolling(60).mean()
    df_stock['MA120'] = df_stock['close'].rolling(120).mean()

    m = df_index[['date', 'market_state']].rename(columns={'market_state': 'idx_state'})
    df_stock = df_stock.merge(m, on='date', how='left')
    df_stock['idx_state'] = df_stock['idx_state'].ffill().fillna('震荡市')

    # 基础 0/1 信号
    bull  = (df_stock['MA60'] > df_stock['MA120']).astype(int)
    osc   = (df_stock['MA5']  > df_stock['MA10']).astype(int)
    base_signal = np.select(
        [df_stock['idx_state'] == '牛市',
         df_stock['idx_state'] == '震荡市',
         df_stock['idx_state'] == '熊市'],
        [bull, osc, 0],
        default=0
    )

    # 震荡市半仓
    df_stock['weight'] = base_signal.astype(float)
    half_mask = (df_stock['idx_state'] == '震荡市') & (base_signal == 1)
    df_stock.loc[half_mask, 'weight'] = 0.5

    # 收益计算
    cost = 0.001
    df_stock['daily_return'] = df_stock['close'].pct_change()
    weight_change = df_stock['weight'].diff().abs()
    df_stock['strategy_return'] = df_stock['weight'].shift(1) * df_stock['daily_return'] - weight_change * cost

    df_stock['cum_strategy'] = (1 + df_stock['strategy_return'].fillna(0)).cumprod()
    df_stock['cum_market']   = (1 + df_stock['daily_return'].fillna(0)).cumprod()

    total  = df_stock['cum_strategy'].iloc[-1] - 1
    market = df_stock['cum_market'].iloc[-1] - 1
    std    = df_stock['strategy_return'].std()
    sharpe = (np.sqrt(252) * df_stock['strategy_return'].mean() / std) if std and std > 1e-12 else 0.0
    mdd    = (df_stock['cum_strategy'] / df_stock['cum_strategy'].cummax() - 1).min()
    trades = int((weight_change > 0.01).sum() // 2)

    dist = df_stock['idx_state'].value_counts(normalize=True)
    avg_weight = df_stock['weight'].mean()

    print(f"大盘状态分布: " + ", ".join(f"{k} {v:.0%}" for k, v in dist.items()))
    print(f"平均仓位: {avg_weight:.0%}")
    print(f"\n=== 择时+半仓震荡 {start_date[:4]}-{end_date[:4]} | {ts_code} ===")
    print(f"策略: {total:.2%} | 基准: {market:.2%} | 超额: {total-market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%} | 调仓: {trades} 次")
    print("-" * 70)

def backtest_portfolio(ts_codes, start_date, end_date):
    """多标的等权组合回测"""
    all_returns = []
    names = []
    
    for code in ts_codes:
        print(f"正在处理 {code} ...")
        try:
            df = _single_stock_backtest(code, start_date, end_date)
            if df is not None and len(df) > 0:
                all_returns.append(df.set_index('date')['strategy_return'])
                names.append(code)
        except Exception as e:
            print(f"  {code} 处理失败: {e}")
    
    if not all_returns:
        print("没有成功回测的标的")
        return
    
    # 合并所有收益率序列（按日期对齐）
    port_returns = pd.concat(all_returns, axis=1, keys=names)
    port_returns = port_returns.mean(axis=1)  # 等权
    
    # 基准：等权持有所有标的
    bench_returns = []
    for code in ts_codes:
        df = get_daily(ts_code=code, start_date=start_date, end_date=end_date)
        df['date'] = pd.to_datetime(df['date'])
        df['ret'] = df['close'].pct_change()
        bench_returns.append(df.set_index('date')['ret'])
    bench_returns = pd.concat(bench_returns, axis=1).mean(axis=1)
    
    # 计算绩效
    cost = 0.001
    # 假设组合调仓次数为各标的调仓次数之和的一半（近似）
    # 简单起见，直接用组合收益率计算，忽略调仓成本
    cum_strategy = (1 + port_returns.fillna(0)).cumprod()
    cum_bench = (1 + bench_returns.fillna(0)).cumprod()
    
    total = cum_strategy.iloc[-1] - 1
    market = cum_bench.iloc[-1] - 1
    std = port_returns.std()
    sharpe = (np.sqrt(252) * port_returns.mean() / std) if std > 1e-12 else 0.0
    mdd = (cum_strategy / cum_strategy.cummax() - 1).min()
    
    print(f"\n=== 多标的组合回测 {start_date[:4]}-{end_date[:4]} ===")
    print(f"标的数: {len(names)}")
    print(f"组合收益: {total:.2%} | 等权持有: {market:.2%} | 超额: {total-market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%}")
    print("-" * 70)

if __name__ == "__main__":
    backtest_with_filter("600519.SH", "20200101", "20261029")