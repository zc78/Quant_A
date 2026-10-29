import pandas as pd
import numpy as np
from data_source import get_daily

# ========== 参数设置 ==========
TS_CODE = "600519.SH"          # 股票代码
START_DATE = "20150101"        # 样本外起始日期
END_DATE = "20191231"          # 样本外结束日期
SHORT_WINDOW = 25              # 短均线
LONG_WINDOW = 50               # 长均线
COST = 0.001                   # 交易成本（佣金+滑点）

# ========== 获取数据 ==========
df = get_daily(ts_code=TS_CODE, start_date=START_DATE, end_date=END_DATE)
if df.empty:
    raise ValueError("未获取到数据，请检查日期范围或网络连接")

# ========== 计算均线和信号 ==========
df['MA_short'] = df['close'].rolling(window=SHORT_WINDOW).mean()
df['MA_long'] = df['close'].rolling(window=LONG_WINDOW).mean()
df['signal'] = (df['MA_short'] > df['MA_long']).astype(int)

# ========== 计算每日收益率 ==========
df['daily_return'] = df['close'].pct_change()
# 策略收益率：当日信号 * 当日市场收益 - 交易成本（换手时扣除）
position_change = df['signal'].diff().abs()
df['strategy_return'] = df['signal'].shift(1) * df['daily_return'] - position_change * COST

# ========== 累计净值 ==========
df['cum_market'] = (1 + df['daily_return'].fillna(0)).cumprod()
df['cum_strategy'] = (1 + df['strategy_return'].fillna(0)).cumprod()

# ========== 绩效指标 ==========
total_return = df['cum_strategy'].iloc[-1] - 1
market_return = df['cum_market'].iloc[-1] - 1
annual_return = (1 + total_return) ** (252 / len(df)) - 1
sharpe = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
max_dd = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min()
trade_count = int(position_change.sum() // 2)

# ========== 输出结果 ==========
print(f"=== 样本外测试 (2015-2019) ===")
print(f"股票代码: {TS_CODE}")
print(f"均线参数: MA{SHORT_WINDOW} / MA{LONG_WINDOW}")
print(f"交易成本: {COST:.1%}")
print(f"数据长度: {len(df)} 个交易日")
print(f"策略总收益率: {total_return:.2%}")
print(f"基准总收益率: {market_return:.2%}")
print(f"超额收益: {total_return - market_return:.2%}")
print(f"年化收益率: {annual_return:.2%}")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd:.2%}")
print(f"交易次数: {trade_count}")