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


if __name__ == "__main__":
    backtest_with_filter("600519.SH", "20200101", "20261029")