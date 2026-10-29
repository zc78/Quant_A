import pandas as pd
import matplotlib.pyplot as plt
from data_source import get_daily  # 假设你的数据模块文件名是 data_source.py

# 获取数据（会优先读缓存）
df = get_daily(ts_code="600519.SH", start_date="20200101")

# 计算均线
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# 生成信号：MA5 > MA20 时持仓（1），否则空仓（0）
df['signal'] = (df['MA5'] > df['MA20']).astype(int)
df['position'] = df['signal'].diff()  # 1 表示金叉买入，-1 表示死叉卖出

# 绘图
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['close'], label='Close Price', alpha=0.5)
plt.plot(df['date'], df['MA5'], label='MA5')
plt.plot(df['date'], df['MA20'], label='MA20')

# 买入点
buy = df[df['position'] == 1]
plt.scatter(buy['date'], buy['close'], marker='^', color='g', s=100, label='Buy')

# 卖出点
sell = df[df['position'] == -1]
plt.scatter(sell['date'], sell['close'], marker='v', color='r', s=100, label='Sell')

plt.title('600519.SH - Double Moving Average Strategy')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()