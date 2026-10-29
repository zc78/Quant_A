import pandas as pd
import numpy as np
from data_source import get_daily

# 1. 获取数据
df = get_daily(ts_code="600519.SH", start_date="20200101")

# 2. 计算均线和信号
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()
df['signal'] = (df['MA5'] > df['MA20']).astype(int)

# 3. 计算每日收益率和策略收益率
df['daily_return'] = df['close'].pct_change()
df['strategy_return'] = df['signal'].shift(1) * df['daily_return']  # 次日开盘按信号交易

# 4. 计算累计净值
df['cum_market'] = (1 + df['daily_return']).cumprod()
df['cum_strategy'] = (1 + df['strategy_return']).cumprod()

# 5. 绩效指标
total_return = df['cum_strategy'].iloc[-1] - 1
market_return = df['cum_market'].iloc[-1] - 1
sharpe_ratio = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
max_drawdown = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min()

print(f"=== 双均线策略回测结果 (2020-2026) ===")
print(f"策略总收益率: {total_return:.2%}")
print(f"基准总收益率: {market_return:.2%}")
print(f"超额收益: {total_return - market_return:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"交易次数: {df['signal'].diff().abs().sum() // 2}")  # 每次买卖算一次交易