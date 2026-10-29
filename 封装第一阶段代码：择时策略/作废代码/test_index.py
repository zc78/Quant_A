import tushare as ts
import pandas as pd

ts.set_token("af89c496c7339c11d590ac4de481fc893bad61af13392c820a76f99d")
pro = ts.pro_api()

# 测试上证指数
df = pro.index_daily(ts_code='000001.SH', start_date='20240101', end_date='20241029')
print("返回行数:", len(df))
print(df.head())