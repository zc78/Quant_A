# -*- coding: utf-8 -*-
"""
数据源模块：支持有网时自动更新缓存，无网时从缓存读取。
所有数据获取函数（get_daily, get_daily_basic, get_fina_indicator, get_trade_calendar）
均实现了缓存回退机制。

依赖：
- tushare
- python-dotenv
- pandas
- pyarrow 或 fastparquet（用于读写 parquet 缓存）
"""

import os
import hashlib
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from pathlib import Path

# ========== 初始化 Tushare ==========
load_dotenv()
token = os.getenv('TUSHARE_TOKEN')

if token:
    ts.set_token(token)
else:
    raise ValueError("请设置环境变量 TUSHARE_TOKEN 或在 .env 文件中配置")

pro = ts.pro_api()

# 代理中转用
pro._DataApi__http_url = "https://tl.kaixin8.top/tushare/sdk"

# ========== 缓存配置 ==========
CACHE_ROOT = Path('./data_cache')  # 缓存根目录

# 尝试导入 parquet 引擎
try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE = 'pyarrow'
except ImportError:
    try:
        import fastparquet  # noqa: F401
        PARQUET_ENGINE = 'fastparquet'
    except ImportError:
        PARQUET_ENGINE = None
        print("[WARN] 未安装 pyarrow 或 fastparquet，将无法使用 parquet 缓存。请执行: pip install pyarrow")


def _cache_path(func_name, *args, **kwargs):
    """
    根据函数名和参数生成唯一的缓存文件路径。
    参数会被序列化后取 MD5 哈希的前 12 位作为文件名。
    """
    # 将 args 和 kwargs 转换为可哈希的字符串
    args_str = ','.join(str(a) for a in args)
    kwargs_str = ','.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key = f"{func_name}|{args_str}|{kwargs_str}"
    hash_val = hashlib.md5(key.encode('utf-8')).hexdigest()[:12]
    cache_dir = CACHE_ROOT / func_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hash_val}.parquet"


def _read_cache(cache_file):
    """从缓存文件读取 DataFrame，失败返回 None"""
    if not cache_file.exists():
        return None
    try:
        df = pd.read_parquet(cache_file, engine=PARQUET_ENGINE)
        return df if not df.empty else None
    except Exception as e:
        print(f"[WARN] 读取缓存失败 ({cache_file.name}): {e}")
        return None


def _write_cache(df, cache_file):
    """将 DataFrame 写入缓存文件"""
    if df is None or df.empty:
        return
    try:
        # 确保目录存在
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False, engine=PARQUET_ENGINE)
    except Exception as e:
        print(f"[WARN] 写入缓存失败 ({cache_file.name}): {e}")


# ========== 数据获取函数（带缓存回退） ==========

def get_daily(code, start, end):
    """
    获取日线行情数据
    参数：
        code: str, 股票代码，如 '600519.SH'
        start: str, 开始日期 'YYYYMMDD'
        end: str, 结束日期 'YYYYMMDD'
    返回：
        DataFrame，包含 tushare 返回的原始字段
    """
    cache_file = _cache_path('get_daily', code, start=start, end=end)

    # 1. 尝试读缓存
    df = _read_cache(cache_file)
    if df is not None:
        return df

    # 2. 从网络拉取
    try:
        df = pro.daily(ts_code=code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            _write_cache(df, cache_file)
            return df
    except Exception as e:
        print(f"[WARN] 网络获取失败 (get_daily {code}): {e}")

    # 3. 网络失败后再次尝试缓存（可能之前被其他进程写入）
    df = _read_cache(cache_file)
    if df is not None:
        print(f"[INFO] 使用缓存数据 (get_daily {code})")
        return df

    # 4. 完全无数据
    print(f"[ERROR] 无法获取 {code} 日线数据（无网络且无缓存）")
    return pd.DataFrame()


def get_daily_basic(code, start, end):
    """
    获取每日基本面数据（pe_ttm, pb, turnover_rate, circ_mv 等）
    """
    cache_file = _cache_path('get_daily_basic', code, start=start, end=end)

    df = _read_cache(cache_file)
    if df is not None:
        return df

    try:
        df = pro.daily_basic(ts_code=code, start_date=start, end_date=end)
        if df is not None and not df.empty:
            _write_cache(df, cache_file)
            return df
    except Exception as e:
        print(f"[WARN] 网络获取失败 (get_daily_basic {code}): {e}")

    df = _read_cache(cache_file)
    if df is not None:
        print(f"[INFO] 使用缓存数据 (get_daily_basic {code})")
        return df

    print(f"[ERROR] 无法获取 {code} 每日基本面数据（无网络且无缓存）")
    return pd.DataFrame()


def get_fina_indicator(code, start, end, max_retries=3):
    cache_file = _cache_path('get_fina_indicator', code, start=start, end=end)
    df = _read_cache(cache_file)
    if df is not None:
        return df
    for attempt in range(max_retries):
        try:
            df = pro.fina_indicator(ts_code=code, start_date=start, end_date=end)
            if df is not None and not df.empty:
                _write_cache(df, cache_file)
                return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[WARN] 重试 {code} ({attempt+1}/{max_retries}) 等待 {wait}s: {str(e)[:60]}")
                time.sleep(wait)
            else:
                print(f"[WARN] 网络获取失败 (get_fina_indicator {code}): {e}")
    # 回退到缓存（如果有旧缓存）
    old_cache = _read_cache(cache_file)
    return old_cache

    print(f"[ERROR] 无法获取 {code} 财务指标数据（无网络且无缓存）")
    return pd.DataFrame()


def get_trade_calendar(start, end):
    """
    获取交易日历数据
    """
    cache_file = _cache_path('get_trade_calendar', start=start, end=end)

    df = _read_cache(cache_file)
    if df is not None:
        return df

    try:
        df = pro.trade_cal(start_date=start, end_date=end)
        if df is not None and not df.empty:
            _write_cache(df, cache_file)
            return df
    except Exception as e:
        print(f"[WARN] 网络获取失败 (get_trade_calendar): {e}")

    df = _read_cache(cache_file)
    if df is not None:
        print(f"[INFO] 使用缓存数据 (get_trade_calendar)")
        return df

    print("[ERROR] 无法获取交易日历数据（无网络且无缓存）")
    return pd.DataFrame()


# ========== 可选：清理缓存函数 ==========
def clear_cache(func_name=None):
    """
    清除缓存。如果不传参数，清除所有缓存；如果传函数名，只清除该函数的缓存。
    """
    if func_name:
        target = CACHE_ROOT / func_name
        if target.exists():
            import shutil
            shutil.rmtree(target)
            print(f"[INFO] 已清除 {func_name} 的缓存")
    else:
        import shutil
        if CACHE_ROOT.exists():
            shutil.rmtree(CACHE_ROOT)
            print("[INFO] 已清除所有缓存")