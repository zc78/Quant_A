"""
数据源模块：封装 Tushare API，带缓存、重试和超时控制

运行方法：
# 1. 先设 token（如果脚本里没硬编码）
$env:TS_TOKEN="你的tushare_token"

# 2. 单并发轻量探测（先跑这个）
python probe_mirror.py

# 3. 加大样本
$env:PROBE_N="50"
python probe_mirror.py

# 4. 测低并发下的表现
$env:PROBE_N="30"
$env:PROBE_CONC="3"
python probe_mirror.py
"""
import os
import time
import pickle
import hashlib
import requests
import tushare as ts
import pandas as pd

# ---------- 缓存配置 ----------
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".tushare_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TSHTTP_TIMEOUT = 30          # 单次请求超时（秒）
MAX_RETRIES = 4              # 最大重试次数
BACKOFF_BASE = 1.5          # 退避基数
REQUEST_GAP = 0.35          # 每次请求后强制停顿（秒）

# 镜像站地址（如果您在使用，请取消注释并填写正确地址）
MIRROR_URL = "https://tl.kaixin8.top/tushare/sdk"


def _cache_path(api_name, **kwargs):
    """生成缓存文件路径，参数必须全部为关键字参数"""
    key = api_name + "|" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{api_name}_{h}.pkl")


def _read_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _write_cache(df, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def clear_cache():
    """清除所有本地缓存"""
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass
    print("[INFO] 已清除所有缓存")


# ---------- 全局 pro 对象 ----------
_pro = None
def get_pro():
    global _pro
    if _pro is not None:
        return _pro

    # ----- 设置 Token（优先级：环境变量 > 硬编码）-----
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        # 如果环境变量未设置，可以在这里硬编码你的 token（注意不要提交到 Git）
        # token = "你的token"
        pass  # 如果两者都没有，ts.pro_api() 会尝试从 ~/.tushare/token 读取
    if token:
        ts.set_token(token)

    pro = ts.pro_api()
    # 如果使用镜像站，取消下面两行的注释
    if MIRROR_URL:
         pro._DataApi__http_url = MIRROR_URL
    pro._DataApi__timeout = TSHTTP_TIMEOUT
    _pro = pro
    return _pro


def safe_call(api_func, api_name, max_retries=MAX_RETRIES, cache=True, **kwargs):
    """
    带指数退避的重试包装；先读缓存，失败写缓存
    kwargs 必须全部为关键字参数，用于生成缓存键
    """
    path = _cache_path(api_name, **kwargs)

    if cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    last_err = None
    for attempt in range(max_retries):
        try:
            df = api_func(**kwargs)
            time.sleep(REQUEST_GAP)  # 限速
            if df is not None and not df.empty:
                if cache:
                    _write_cache(df, path)
                return df
            # 返回空不算错，但也不写缓存（下次再试）
            last_err = "empty"
            if attempt < max_retries - 1:
                time.sleep(BACKOFF_BASE * (2 ** attempt))
                continue
            return df
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                ConnectionError) as e:
            last_err = str(e)
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"[RETRY] {api_name} 第{attempt+1}次失败，{wait:.1f}s后重试: {str(e)[:50]}")
            time.sleep(wait)
        except Exception as e:
            err_msg = str(e)
            if 'rate_limited' in err_msg.lower() or '额度' in err_msg:
                wait = 30  # 限流时等待30秒
                print(f"[LIMIT] {api_name} 触发限流，等待{wait}s...")
                time.sleep(wait)
                continue  # 继续重试（不计入重试次数？这里简化，直接继续）
            print(f"[FAIL] {api_name} 业务错误: {err_msg[:80]}")
            return None

    print(f"[GIVEUP] {api_name} 多次重试失败: {last_err}")
    # 最后尝试读旧缓存兜底
    return _read_cache(path)


# ---------- 业务函数 ----------
def get_daily(ts_code, start_date, end_date):
    """获取日线行情，失败时返回空 DataFrame"""
    pro = get_pro()
    df = safe_call(pro.daily, "daily",
                   ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None:
        return pd.DataFrame()   # 返回空 DataFrame 而非 None
    return df


def get_fina_indicator(ts_code, start_date, end_date):
    """获取财务指标，使用 safe_call 包装"""
    pro = get_pro()
    return safe_call(pro.fina_indicator, "fina_indicator",
                     ts_code=ts_code, start_date=start_date, end_date=end_date)


def get_daily_basic(ts_code, start_date, end_date):
    """获取每日指标，使用 safe_call 包装"""
    pro = get_pro()
    return safe_call(pro.daily_basic, "daily_basic",
                     ts_code=ts_code, start_date=start_date, end_date=end_date)


def get_trade_calendar(start_date, end_date):
    """获取交易日历，使用 safe_call 包装"""
    pro = get_pro()
    return safe_call(pro.trade_cal, "trade_cal",
                     exchange="SSE", start_date=start_date,
                     end_date=end_date, is_open="1")