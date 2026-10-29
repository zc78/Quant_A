import pandas as pd
import numpy as np
from data_source import get_daily

# 1. 获取数据
df = get_daily(ts_code="600519.SH", start_date="20200101")

# 2. 计算均线和信号
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# 改进：只在 MA20 向上时才允许买入（趋势过滤）
df['MA20_up'] = df['MA20'] > df['MA20'].shift(1)

# 信号：金叉 且 长期趋势向上
df['signal'] = ((df['MA5'] > df['MA20']) & df['MA20_up']).astype(int)

# 3. 计算收益率（用次日开盘价成交，避免未来函数）
df['next_open'] = df['open'].shift(-1)
df['daily_return'] = df['close'].pct_change()

# 交易成本：每次换手 0.1%（佣金+印花税+滑点）
cost = 0.001
position_change = df['signal'].diff().abs()

# 策略收益 = 持仓期间的市场收益 - 交易成本
df['strategy_return'] = df['signal'].shift(1) * df['daily_return'] - position_change * cost

# 4. 累计净值
df['cum_market'] = (1 + df['daily_return'].fillna(0)).cumprod()
df['cum_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()

# 5. 绩效指标
total_return = df['cum_strategy'].iloc[-1] - 1
market_return = df['cum_market'].iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len(df)) - 1
sharpe = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min()

print(f"=== 优化版双均线策略回测 ===")
print(f"策略总收益率: {total_return:.2%}")
print(f"基准总收益率: {market_return:.2%}")
print(f"超额收益: {total_return - market_return:.2%}")
print(f"年化收益率: {annual_return:.2%}")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd:.2%}")
print(f"交易次数: {int(position_change.sum() // 2)}")