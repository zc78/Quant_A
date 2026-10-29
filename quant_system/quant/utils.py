"""
utils.py —— 小工具
"""
import pandas as pd
from . import config
from .logger import get_logger

log = get_logger("utils")

DATE_FMT = "%Y-%m-%d"


def to_date_str(x) -> str:
    """统一日期输出为 YYYY-MM-DD 字符串"""
    if pd.isna(x):
        return ""
    if isinstance(x, pd.Timestamp):
        return x.strftime(DATE_FMT)
    s = str(x)
    if "-" in s:
        return s[:10]
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def parse_date(x) -> pd.Timestamp | None:
    """容错解析日期"""
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x)
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def yyyymmdd(x) -> str:
    """转 YYYYMMDD 紧凑格式"""
    ts = parse_date(x)
    return "" if ts is None else ts.strftime("%Y%m%d")


def progress(n: int, total: int, prefix: str = "", bar_len: int = 30) -> str:
    if total <= 0:
        return prefix
    pct = n / total
    filled = int(bar_len * pct)
    bar = "█" * filled + "·" * (bar_len - filled)
    return f"\r{prefix}[{bar}] {n}/{total} {pct*100:5.1f}%"
