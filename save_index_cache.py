import pandas as pd
import tushare as ts
from data_source import TOKEN, CACHE_DIR, os

ts.set_token(TOKEN.strip())
pro = ts.pro_api()

# 分年份拉，避免一次触发限流（每小时拉一次，慢慢攒）
for y in range(2020, 2027):
    sd = f"{y}0101"
    ed = f"{y}1231" if y < 2026 else "20261029"
    print(f"拉取 {sd}~{ed} ...")
    df = pro.index_daily(ts_code="000001.SH", start_date=sd, end_date=ed)
    df = df.rename(columns={"vol": "volume", "trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    out = os.path.join(CACHE_DIR, f"idx_000001.SH_{sd}_{ed}.csv")
    df.to_csv(out, index=False)
    print(f"  保存 {len(df)} 行 -> {out}")
    break  # 第一次先只拉一年，等限流恢复再继续