import pandas as pd
import numpy as np
from data_source import get_daily

df = get_daily(ts_code="600519.SH", start_date="20200101")

best_sharpe = -999
best_params = None
results = []

for short_window in range(5, 31, 5):      # 短均线 5,10,15,20,25,30
    for long_window in range(20, 101, 10): # 长均线 20,30,...,100
        if short_window >= long_window:
            continue
        
        # 计算均线和信号
        df['MA_short'] = df['close'].rolling(window=short_window).mean()
        df['MA_long'] = df['close'].rolling(window=long_window).mean()
        df['signal'] = (df['MA_short'] > df['MA_long']).astype(int)
        
        # 计算收益率（含交易成本）
        cost = 0.001
        position_change = df['signal'].diff().abs()
        df['strategy_return'] = df['signal'].shift(1) * df['close'].pct_change() - position_change * cost
        
        # 绩效指标
        sharpe = np.sqrt(252) * df['strategy_return'].mean() / df['strategy_return'].std()
        cum = (1 + df['strategy_return'].fillna(0)).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        total_ret = cum.iloc[-1] - 1
        
        results.append((short_window, long_window, sharpe, max_dd, total_ret))
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (short_window, long_window, total_ret, max_dd)

# 输出最佳参数
print(f"最佳参数组合：MA{best_params[0]} / MA{best_params[1]}")
print(f"夏普比率：{best_sharpe:.2f}")
print(f"总收益率：{best_params[2]:.2%}")
print(f"最大回撤：{best_params[3]:.2%}")

# 输出排名前5的组合
results.sort(key=lambda x: x[2], reverse=True)
print("\n=== 前5名参数组合 ===")
for i, r in enumerate(results[:5]):
    print(f"{i+1}. MA{r[0]}/MA{r[1]}  夏普:{r[2]:.2f}  回撤:{r[3]:.2%}  收益:{r[4]:.2%}")