import time
import os
import pandas as pd
import tushare as ts
from data_source import TOKEN, CACHE_DIR

ts.set_token(TOKEN.strip())
pro = ts.pro_api()

years = [(2020, "0101", "1231"),
         (2021, "0101", "1231"),
         (2022, "0101", "1231"),
         (2023, "0101", "1231"),
         (2024, "0101", "1231"),
         (2025, "0101", "1231"),
         (2026, "0101", "1029")]  # 2026年截止到10月29日

all_parts = []

for year, start_month, end_month in years:
    start = f"{year}{start_month}"
    end = f"{year}{end_month}"
    part_file = os.path.join(CACHE_DIR, f"idx_000001.SH_{start}_{end}.csv")

    if os.path.exists(part_file):
        print(f"[缓存已存在] {start}~{end}，跳过")
        df_part = pd.read_csv(part_file)
        all_parts.append(df_part)
        continue

    print(f"[拉取] {start}~{end} ...")
    try:
        df = pro.index_daily(ts_code="000001.SH", start_date=start, end_date=end)
        if df is not None and not df.empty:
            df = df.rename(columns={"vol": "volume", "trade_date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df.to_csv(part_file, index=False)
            all_parts.append(df)
            print(f"  保存 {len(df)} 行 -> {part_file}")
        else:
            print(f"  警告：{start}~{end} 无数据")
    except Exception as e:
        print(f"  拉取失败：{e}")
        # 如果限流，等待一小时后继续
        if "频率超限" in str(e):
            print("  限流触发，等待 3600 秒...")
            time.sleep(3600)
            continue
        else:
            break

    # 如果不是最后一年，等待 1 小时（避免限流）
    if year != 2026:
        print("  等待 3600 秒（防止限流）...")
        time.sleep(3600)

# 合并所有分段
if all_parts:
    full_df = pd.concat(all_parts, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    full_file = os.path.join(CACHE_DIR, "idx_000001.SH_20200101_20261029.csv")
    full_df.to_csv(full_file, index=False)
    print(f"\n✅ 完整缓存已生成：{full_file}，共 {len(full_df)} 行")
else:
    print("❌ 未能获取任何数据")