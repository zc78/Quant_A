import pandas as pd
import numpy as np
from data_source import get_daily, get_index_daily

# ========== ADX 计算 ==========
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

def calc_atr(high, low, close, period=14):
    """计算 ATR（平均真实波幅）"""
    tr = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    atr = tr.rolling(period).mean()
    return atr
# ========== 大盘市场状态 ==========
def add_market_state(df, window=250):
    df = df.copy()
    df['MA250'] = df['close'].rolling(window).mean()
    df['MA250_slope'] = df['MA250'].diff(20)
    adx = calculate_adx(df, 14)
    df['ADX'] = adx
    conds = [
        df['ADX'] < 20,
        (df['ADX'] >= 20) & (df['MA250_slope'] <= 0),
        (df['ADX'] >= 20) & (df['MA250_slope'] > 0)
    ]
    choices = ['震荡市', '熊市', '牛市']
    df['market_state'] = np.select(conds, choices, default='震荡市')
    return df

# ========== 移动止盈止损 ==========
def apply_trail_stop(signal, close, trail_pct=0.15, stop_loss=0.10):
    """
    在原始持仓信号上叠加移动止盈止损。
    signal: 原始持仓信号(0/1)
    close:  收盘价序列
    trail_pct: 移动止盈回撤阈值（15%）
    stop_loss: 止损阈值（10%）
    返回：最终持仓信号
    """
    position = 0
    entry_price = np.nan
    highest_since_entry = np.nan
    final_signal = signal.copy().to_numpy(copy=True)

    for i in range(len(signal)):
        cur_sig = signal.iloc[i]
        cur_price = close.iloc[i]

        if position == 0:
            # 空仓：收到入场信号就进
            if cur_sig == 1:
                position = 1
                entry_price = cur_price
                highest_since_entry = cur_price
        else:
            # 持仓：更新最高价
            highest_since_entry = max(highest_since_entry, cur_price)
            ret_from_high = cur_price / highest_since_entry - 1
            ret_from_entry = cur_price / entry_price - 1

            # 止损：浮亏超阈值
            if ret_from_entry <= -stop_loss:
                position = 0
                final_signal[i] = 0
            # 移动止盈：从最高点回撤超阈值
            elif ret_from_high <= -trail_pct:
                position = 0
                final_signal[i] = 0
            else:
                final_signal[i] = 1

    return pd.Series(final_signal, index=signal.index)

# ========== 回测 ==========
def backtest_with_filter_old(ts_code, start_date, end_date, trail_pct=0.15, stop_loss=0.10):
    df_stock = get_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_index = get_index_daily(ts_code="000001.SH", start_date=start_date, end_date=end_date)

    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_index['date'] = pd.to_datetime(df_index['date'])

    # 大盘逐日状态
    df_index = add_market_state(df_index)

    # 个股多组均线
    df_stock['MA5']   = df_stock['close'].rolling(5).mean()
    df_stock['MA10']  = df_stock['close'].rolling(10).mean()
    df_stock['MA60']  = df_stock['close'].rolling(60).mean()
    df_stock['MA120'] = df_stock['close'].rolling(120).mean()

    # 对齐大盘状态
    m = df_index[['date', 'market_state']].rename(columns={'market_state': 'idx_state'})
    df_stock = df_stock.merge(m, on='date', how='left')
    df_stock['idx_state'] = df_stock['idx_state'].ffill().fillna('震荡市')

    # 基础信号
    bull  = (df_stock['MA60'] > df_stock['MA120']).astype(int)
    osc   = (df_stock['MA5']  > df_stock['MA10']).astype(int)
    base_signal = np.select(
        [df_stock['idx_state'] == '牛市',
         df_stock['idx_state'] == '震荡市',
         df_stock['idx_state'] == '熊市'],
        [bull, osc, 0],
        default=0
    )

    # 叠加移动止盈止损
    final_signal = apply_trail_stop(
        pd.Series(base_signal, index=df_stock.index),
        df_stock['close'],
        trail_pct=trail_pct,
        stop_loss=stop_loss
    )
    df_stock['signal'] = final_signal.values

    # 收益计算
    cost = 0.001
    df_stock['daily_return'] = df_stock['close'].pct_change()
    pos_change = df_stock['signal'].diff().abs()
    df_stock['strategy_return'] = df_stock['signal'].shift(1) * df_stock['daily_return'] - pos_change * cost

    df_stock['cum_strategy'] = (1 + df_stock['strategy_return'].fillna(0)).cumprod()
    df_stock['cum_market']   = (1 + df_stock['daily_return'].fillna(0)).cumprod()

    total  = df_stock['cum_strategy'].iloc[-1] - 1
    market = df_stock['cum_market'].iloc[-1] - 1
    std    = df_stock['strategy_return'].std()
    sharpe = (np.sqrt(252) * df_stock['strategy_return'].mean() / std) if std and std > 1e-12 else 0.0
    mdd    = (df_stock['cum_strategy'] / df_stock['cum_strategy'].cummax() - 1).min()
    trades = int(pos_change.sum() // 2)

    print(f"=== 择时+止盈止损 {start_date[:4]}-{end_date[:4]} | {ts_code} ===")
    print(f"止盈回撤: {trail_pct:.0%} | 止损: {stop_loss:.0%}")
    print(f"策略: {total:.2%} | 基准: {market:.2%} | 超额: {total-market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%} | 交易: {trades} 次")
    print("-" * 70)

def apply_trail_stop_atr(signal, close, high, low, atr_period=14, 
                         atr_mult_stop=2.0, atr_mult_trail=3.0,
                         fallback_stop=0.10, fallback_trail=0.15):
    """
    基于 ATR 的动态止盈止损。
    - 止损位：入场价 - atr_mult_stop * ATR
    - 移动止盈：最高价回落 atr_mult_trail * ATR
    - 前 atr_period 天用固定百分比 fallback
    """
    atr = calc_atr(high, low, close, period=atr_period)
    
    position = 0
    entry_price = np.nan
    highest_since_entry = np.nan
    final_signal = signal.copy().to_numpy(copy=True)

    for i in range(len(signal)):
        cur_sig = signal.iloc[i]
        cur_close = close.iloc[i]
        cur_atr = atr.iloc[i]

        if pd.isna(cur_atr):
            # 早期数据不足，用固定百分比
            stop_dist = fallback_stop
            trail_dist = fallback_trail
        else:
            stop_dist = atr_mult_stop * cur_atr / cur_close   # 转换为百分比
            trail_dist = atr_mult_trail * cur_atr / cur_close

        if position == 0:
            if cur_sig == 1:
                position = 1
                entry_price = cur_close
                highest_since_entry = cur_close
        else:
            highest_since_entry = max(highest_since_entry, cur_close)
            ret_from_entry = cur_close / entry_price - 1
            ret_from_high = cur_close / highest_since_entry - 1

            # 止损：跌幅超过 stop_dist
            if ret_from_entry <= -stop_dist:
                position = 0
                final_signal[i] = 0
            # 移动止盈：从最高点回落超过 trail_dist
            elif ret_from_high <= -trail_dist:
                position = 0
                final_signal[i] = 0
            else:
                final_signal[i] = 1

    return pd.Series(final_signal, index=signal.index)

def backtest_with_filter(ts_code, start_date, end_date, 
                         atr_period=14, atr_mult_stop=2.0, atr_mult_trail=3.0):
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

    bull  = (df_stock['MA60'] > df_stock['MA120']).astype(int)
    osc   = (df_stock['MA5']  > df_stock['MA10']).astype(int)
    base_signal = np.select(
        [df_stock['idx_state'] == '牛市',
         df_stock['idx_state'] == '震荡市',
         df_stock['idx_state'] == '熊市'],
        [bull, osc, 0],
        default=0
    )

    # 使用 ATR 动态止盈止损
    final_signal = apply_trail_stop_atr(
        pd.Series(base_signal, index=df_stock.index),
        df_stock['close'],
        df_stock['high'],
        df_stock['low'],
        atr_period=atr_period,
        atr_mult_stop=atr_mult_stop,
        atr_mult_trail=atr_mult_trail
    )
    df_stock['signal'] = final_signal.values

    # 收益计算（与之前相同）
    cost = 0.001
    df_stock['daily_return'] = df_stock['close'].pct_change()
    pos_change = df_stock['signal'].diff().abs()
    df_stock['strategy_return'] = df_stock['signal'].shift(1) * df_stock['daily_return'] - pos_change * cost

    df_stock['cum_strategy'] = (1 + df_stock['strategy_return'].fillna(0)).cumprod()
    df_stock['cum_market']   = (1 + df_stock['daily_return'].fillna(0)).cumprod()

    total  = df_stock['cum_strategy'].iloc[-1] - 1
    market = df_stock['cum_market'].iloc[-1] - 1
    std    = df_stock['strategy_return'].std()
    sharpe = (np.sqrt(252) * df_stock['strategy_return'].mean() / std) if std and std > 1e-12 else 0.0
    mdd    = (df_stock['cum_strategy'] / df_stock['cum_strategy'].cummax() - 1).min()
    trades = int(pos_change.sum() // 2)

    print(f"=== 择时+ATR止盈止损 {start_date[:4]}-{end_date[:4]} | {ts_code} ===")
    print(f"ATR周期={atr_period}, 止损倍数={atr_mult_stop}, 止盈倍数={atr_mult_trail}")
    print(f"策略: {total:.2%} | 基准: {market:.2%} | 超额: {total-market:.2%}")
    print(f"夏普: {sharpe:.2f} | 回撤: {mdd:.2%} | 交易: {trades} 次")
    print("-" * 70)

if __name__ == "__main__":
    # 用你之前跑通的 2020-2026 区间（已有缓存）
    backtest_with_filter("600519.SH", "20200101", "20261029")
    # 2015-2019 等限流解除后再跑
    # backtest_with_filter("600519.SH", "20150101", "20191231")