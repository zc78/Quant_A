# -*- coding: utf-8 -*-
"""
以 tushare 为数据源的行情数据模块
功能：拉取 A 股日线行情、清洗、存储为 CSV
"""

import os, time
import pandas as pd
import tushare as ts

# ========== 在这里填入你的 tushare token ==========
TOKEN = "af89c496c7339c11d590ac4de481fc893bad61af13392c820a76f99d"
# =================================================

# tushare 初始化
ts.set_token(TOKEN)
pro = ts.pro_api()

# CSV 缓存目录
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_daily(ts_code: str,
              start_date: str = "20180101",
              end_date: str = None,
              use_cache: bool = True) -> pd.DataFrame:
    """
    获取单只股票的日线行情

    参数:
        ts_code   : tushare 代码，如 "000001.SZ"、"600519.SH"
        start_date: 起始日期，格式 "YYYYMMDD"
        end_date  : 结束日期，默认今天
        use_cache : 是否优先读本地缓存

    返回:
        DataFrame，索引为日期(datetime)，按时间升序，包含列：
        open / high / low / close / vol(volume)
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    cache_file = os.path.join(CACHE_DIR, f"{ts_code}_{start_date}_{end_date}.csv")

    # 优先读缓存
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df["trade_date"] = pd.to_datetime(df["date"])
        return df.sort_values("trade_date").reset_index(drop=True)

    # 从 tushare 拉取
    df = pro.daily(ts_code=ts_code,
                   start_date=start_date,
                   end_date=end_date)

    if df is None or df.empty:
        raise ValueError(f"未获取到数据：{ts_code}，请检查代码或 token")

    # 清洗：日期转 datetime、按时间升序、列名统一
    df = df.rename(columns={"vol": "volume", "trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 统一列名：open / high / low / close / volume
    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[keep_cols]

    # 存缓存
    df.to_csv(cache_file, index=False)

    return df

def get_index_daily(ts_code: str = "000001.SH",
                    start_date: str = "20180101",
                    end_date: str = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    获取指数日线：离线优先，远程兜底。
    - 完整区间缓存存在 → 直接读，绝不打接口
    - 缓存不存在 → 按年分片拉取，合并去重后落盘
    - 遇到限流 → 明确报错，不重复请求
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    # 完整区间缓存文件名（按起止日期命名）
    cache_file = os.path.join(CACHE_DIR, f"idx_{ts_code}_{start_date}_{end_date}.csv")

    # 1. 优先读完整缓存
    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # 2. 缓存缺失 → 先试分片缓存合并
    parts = []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.startswith(f"idx_part_{ts_code}_") and f.endswith(".csv"):
            part = pd.read_csv(os.path.join(CACHE_DIR, f))
            part["date"] = pd.to_datetime(part["date"])
            parts.append(part)

    if parts:
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
        if not df.empty:
            df.to_csv(cache_file, index=False)   # 合并后写完整缓存
            return df.reset_index(drop=True)

    # 3. 分片也没有 → 打远程（这里才会碰 tushare）
    df = _fetch_index_from_tushare(ts_code, start_date, end_date)

    # 清洗落盘
    df = df.rename(columns={"vol": "volume", "trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(cache_file, index=False)
    return df


def _fetch_index_from_tushare(ts_code, start_date, end_date):
    """按年分片拉，遇限流直接抛错（不重试狂打）"""
    start_y = int(start_date[:4])
    end_y = int(end_date[:4])
    frames = []

    for y in range(start_y, end_y + 1):
        sd = f"{y}0101" if y != start_y else start_date
        ed = f"{y}1231" if y != end_y else end_date
        part_file = os.path.join(CACHE_DIR, f"idx_part_{ts_code}_{sd}_{ed}.csv")

        # 分片已有缓存就跳过
        if os.path.exists(part_file):
            frames.append(pd.read_csv(part_file))
            continue

        try:
            chunk = pro.index_daily(ts_code=ts_code, start_date=sd, end_date=ed)
        except Exception as e:
            msg = str(e)
            if "频率超限" in msg or "权限" in msg:
                raise RuntimeError(
                    f"⛔ 指数接口限流/无权限：{msg}\n"
                    f"请等待限流恢复后，用 prepare_index_cache.py 预热缓存，\n"
                    f"之后回测会完全走本地 CSV，不再请求网络。"
                )
            raise

        if chunk is not None and not chunk.empty:
            chunk = chunk.rename(columns={"vol": "volume", "trade_date": "date"})
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk = chunk[["date", "open", "high", "low", "close", "volume"]]
            chunk.to_csv(part_file, index=False)
            frames.append(chunk)
            time.sleep(1.2)   # 温和限速，保护配额

    if not frames:
        raise ValueError(f"未获取到指数数据：{ts_code}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def get_fund_daily(ts_code: str = "510300.SH",
                   start_date: str = "20180101",
                   end_date: str = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """获取 ETF/基金 日线（用 pro.fund_daily）"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    cache_file = os.path.join(CACHE_DIR, f"fund_{ts_code}_{start_date}_{end_date}.csv")

    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        raise ValueError(f"未获取到ETF数据：{ts_code}，可能无接口权限")

    df = df.rename(columns={"vol": "volume", "trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_csv(cache_file, index=False)
    return df

# ========== 日频基本面数据（估值/流动性/规模）==========

def get_daily_basic(ts_code: str = None,
                    trade_date: str = None,
                    start_date: str = None,
                    end_date: str = None,
                    use_cache: bool = True) -> pd.DataFrame:
    """
    获取日频基本面数据（pro.daily_basic）
    包含字段：pe_ttm, pb, ps_ttm, dv_ratio, turnover_rate, circ_mv 等

    两种用法：
      1. 按股票：ts_code="600519.SH", start_date, end_date
      2 按日期：trade_date="20261029"（全市场快照）

    返回 DataFrame，带 date 列。
    """
    # 按股票拉取（我们主要用这种）
    if ts_code is not None:
        if start_date is None:
            start_date = "20180101"
        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y%m%d")

        cache_file = os.path.join(CACHE_DIR, f"basic_{ts_code}_{start_date}_{end_date}.csv")
        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)

        # 分片拉取（按年，防限流）
        start_y = int(start_date[:4])
        end_y = int(end_date[:4])
        frames = []
        for y in range(start_y, end_y + 1):
            sd = f"{y}0101" if y != start_y else start_date
            ed = f"{y}1231" if y != end_y else end_date
            try:
                chunk = pro.daily_basic(ts_code=ts_code, start_date=sd, end_date=ed)
            except Exception as e:
                msg = str(e)
                if "频率超限" in msg or "权限" in msg:
                    raise RuntimeError(f"⛔ daily_basic 限流/无权限：{msg}")
                raise
            if chunk is not None and not chunk.empty:
                frames.append(chunk)
                time.sleep(0.3)

        if not frames:
            raise ValueError(f"未获取到 daily_basic：{ts_code}")

        df = pd.concat(frames, ignore_index=True)
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        return df

    # 按日期拉全市场（备用）
    elif trade_date is not None:
        cache_file = os.path.join(CACHE_DIR, f"basic_all_{trade_date}.csv")
        if use_cache and os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        df = pro.daily_basic(trade_date=trade_date)
        if df is None or df.empty:
            raise ValueError(f"未获取到全市场 basic：{trade_date}")
        df = df.rename(columns={"trade_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(cache_file, index=False)
        return df

    else:
        raise ValueError("请至少提供 ts_code 或 trade_date 之一")

# ========== 财务数据接口 ==========

def get_fina_indicator(ts_code: str,
                       start_date: str = "20180101",
                       end_date: str = None,
                       use_cache: bool = True) -> pd.DataFrame:
    """
    获取财务指标（pro.fina_indicator）
    返回关键质量/成长因子：roe, roe_waa, grossprofit_margin, netprofit_yoy, or_yoy 等
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    cache_file = os.path.join(CACHE_DIR, f"fina_{ts_code}_{start_date}_{end_date}.csv")
    if use_cache and os.path.exists(cache_file):
        return pd.read_csv(cache_file)

    frames = []
    start_y = int(start_date[:4])
    end_y = int(end_date[:4])
    for y in range(start_y, end_y + 1):
        sd = f"{y}0101" if y != start_y else start_date
        ed = f"{y}1231" if y != end_y else end_date
        try:
            chunk = pro.fina_indicator(ts_code=ts_code,
                                       start_date=sd, end_date=ed,
                                       fields="ts_code,ann_date,end_date,roe,roe_waa,"
                                              "grossprofit_margin,netprofit_margin,"
                                              "netprofit_yoy,or_yoy,roa,debt_to_assets")
        except Exception as e:
            msg = str(e)
            if "频率超限" in msg or "权限" in msg:
                raise RuntimeError(f"⛔ fina_indicator 限流：{msg}")
            raise
        if chunk is not None and not chunk.empty:
            frames.append(chunk)
            time.sleep(0.3)

    if not frames:
        raise ValueError(f"未获取到财务数据：{ts_code}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["end_date"]).sort_values("end_date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


def get_trade_calendar(start_date="20180101", end_date=None) -> pd.DataFrame:
    """交易日历"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
    cache_file = os.path.join(CACHE_DIR, f"cal_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    df = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
    df.to_csv(cache_file, index=False)
    return df


def get_stock_basic() -> pd.DataFrame:
    """股票基础信息（含行业、上市日期）"""
    cache_file = os.path.join(CACHE_DIR, "stock_basic.csv")
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,symbol,name,area,industry,list_date")
    df.to_csv(cache_file, index=False)
    return df

# ========== 测试 ==========
if __name__ == "__main__":
    # 示例：贵州茅台 600519.SH
    df = get_daily(ts_code="600519.SH",
                   start_date="20200101")
    print(f"获取到 {len(df)} 条数据")
    print(df.head())
    print(df.tail())