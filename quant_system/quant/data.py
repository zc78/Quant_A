"""
data.py —— 数据层

职责：拉取、清洗、复权、落盘、加载。
设计原则：
1. 对外只暴露 get_price / get_universe / get_trade_calendar 三个函数，
   上层不知道数据从哪来、存成什么格式。
2. 价格统一为前复权，保证跨期可比；原始价格另行保留。
3. 股票池按"当时能否买"构建：剔除 ST、停牌、次新、上市未足 60 日。
4. 交易日历用 sina 口径，与行情日期严格对齐。

落盘结构：
  data/raw/
    daily/               每个标的一个 csv：{market}_{code}.csv
    meta.csv            全市场最新快照（含 ST 标记）
    trade_calendar.csv
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .logger import get_logger
from .utils import to_date_str, parse_date

log = get_logger("data")

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]

# ============================================================
# 一、数据抓取
# ============================================================

def _normalize_df(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """统一字段名与类型，处理 akshare 返回的脏数据"""
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for c in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[COLUMNS].dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["code"] = code
    return df.reset_index(drop=True)


def _fetch_tx(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """腾讯历史日线接口（沙盒内可用）。adjust: qfq 前复权 / hfq 后复权 / 空 不复权。"""
    import requests

    sym = code
    if not (sym.startswith("sh") or sym.startswith("sz") or sym.startswith("bj")):
        code6 = sym
        sym = "sh" + code6 if code6[0] in "65" else "sz" + code6

    qmap = {"qfq": "qfq", "hfq": "hfq", "": "none", None: "none"}
    q = qmap.get(adjust, "qfq")

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "param": f"{sym},day,{start},{end},640,{q}",
        "_var": "kline_dayqfq",
    }
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            text = r.text
            js = text.split("=", 1)[1] if "=" in text else text
            import json
            data = json.loads(js)
            key = sym.lower()
            payload = data.get("data", {}).get(key, {})
            rows = payload.get("day") or payload.get("qfqday") or payload.get("hfqday") or []
            if not rows:
                raise ValueError(f"空数据: {data}")
            df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low",
                                             "volume"])
            df["date"] = pd.to_datetime(df["date"])
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["amount"] = df["volume"] * 100 * df["close"]
            df["turnover"] = np.nan
            out = df[COLUMNS].dropna(subset=["close"]).sort_values("date")
            out["code"] = code
            return out.reset_index(drop=True)
        except Exception as e:  # noqa
            wait = 1.5 * (attempt + 1)
            log.warning(f"tx 抓取失败 {code} attempt={attempt}: {e}; sleep {wait}s")
            time.sleep(wait)
    return pd.DataFrame(columns=COLUMNS + ["code"])


def _fetch_sina_rt(code: str, days: int = 1023) -> pd.DataFrame:
    """新浪实时接口（仅最近 ~1023 日，无复权，作为降级源）。"""
    import requests
    try:
        r = requests.get("https://hq.sinajs.cn/list=" + code,
                         headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
        f = r.text.split('"')[1].split(",")
        close = float(f[3])
        now = pd.Timestamp.now().normalize()
        df = pd.DataFrame([{"date": now, "open": float(f[1]), "high": float(f[4]),
                            "low": float(f[5]), "close": close,
                            "volume": float(f[8]), "amount": float(f[9]),
                            "turnover": float(f[10]) if len(f) > 10 else np.nan}])
        df["code"] = code
        return df[COLUMNS + ["code"]]
    except Exception as e:
        log.warning(f"sina 实时失败 {code}: {e}")
        return pd.DataFrame(columns=COLUMNS + ["code"])


def fetch_history(code: str, start: str, end: str,
                  adjust: str = "qfq", source: str = None) -> pd.DataFrame:
    """
    抓取单标的日线。adjust: qfq 前复权 / hfq 后复权 / 空 不复权。
    优先走腾讯历史日线接口（沙盒内可用），失败降级新浪实时。
    """
    start_s = to_date_str(start)
    end_s = to_date_str(end)
    out = _fetch_tx(code, start_s, end_s, adjust)
    if out.empty:
        log.error(f"tx 抓取最终失败 {code}，降级 sina 实时")
        out = _fetch_sina_rt(code)
    return out


def _fetch_em(code6: str, sdate: str, edate: str, adjust: str) -> pd.DataFrame:
    """东方财富接口（需 akshare 支持，沙盒未安装时不可用，保留兼容）。"""
    try:
        import akshare as ak
    except ImportError:
        return pd.DataFrame(columns=COLUMNS)
    if adjust == "qfq":
        qfq = "qfq"
    elif adjust == "hfq":
        qfq = "hfq"
    else:
        qfq = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code6, period="daily",
                                    start_date=sdate, end_date=edate,
                                    adjust=qfq)
            return _normalize_df(df, code6)
        except Exception as e:  # noqa
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame(columns=COLUMNS)


# ============================================================
# 二、本地存储
# ============================================================

def _path_of(code: str) -> Path:
    return config.RAW_DIR / "daily" / f"{code}.csv"


def save_daily(df: pd.DataFrame) -> None:
    if df.empty:
        return
    code = df["code"].iloc[0]
    p = _path_of(code)
    df[COLUMNS].to_csv(p, index=False, encoding="utf-8-sig")


def _ensure_seed():
    """若本地无任何落盘数据，自动生成合成样本数据，保证离线可跑。"""
    d = config.RAW_DIR / "daily"
    if d.exists() and any(d.glob("*.csv")):
        return
    try:
        from . import data_seed
        log.warning("本地无行情数据，自动生成合成样本（data_seed），仅用于教学联调")
        data_seed.build_seed()
    except Exception as e:
        log.error(f"合成数据生成失败: {e}")


def load_daily(code: str) -> pd.DataFrame:
    _ensure_seed()
    p = _path_of(code)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def list_cached_codes() -> list[str]:
    _ensure_seed()
    d = config.RAW_DIR / "daily"
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.csv"))


# ============================================================
# 三、批量更新（增量）
# ============================================================

def update_universe(codes: Iterable[str], end: str = None,
                    delay: float = 0.35) -> dict:
    """
    批量增量更新。每个标的只在本地最新日期之后补数据，
    并对本地尾部做"同源重算复权"校验。
    """
    end = end or config.END_DATE
    codes = list(dict.fromkeys(codes))
    stats = {"new": 0, "updated": 0, "failed": 0, "skipped": 0}
    total = len(codes)

    for i, code in enumerate(codes, 1):
        try:
            local = load_daily(code)
            if local.empty:
                new_df = fetch_history(code, config.START_DATE, end)
                if new_df.empty:
                    stats["failed"] += 1
                else:
                    save_daily(new_df)
                    stats["new"] += 1
            else:
                last = local["date"].max()
                if last >= parse_date(end):
                    stats["skipped"] += 1
                else:
                    tail = last.strftime("%Y%m%d")
                    new_df = fetch_history(code, tail, end)
                    if new_df.empty or new_df["date"].max() <= last:
                        stats["skipped"] += 1
                    else:
                        # 同源重算：用新拉的整段覆盖本地，避免复权因子漂移导致接缝错价
                        full = fetch_history(code, config.START_DATE, end)
                        save_daily(full)
                        stats["updated"] += 1
        except Exception as e:  # noqa
            stats["failed"] += 1
            log.error(f"update {code} 异常: {e}")

        if i % 50 == 0:
            log.info(f"进度 {i}/{total} 状态={stats}")
        time.sleep(delay)

    log.info(f"批量更新完成: {stats}")
    return stats


# ============================================================
# 四、对外 API
# ============================================================

def get_price(codes, start=None, end=None) -> pd.DataFrame:
    """
    返回面板数据，MultiIndex(code, date)。
    若本地缺失自动联网抓取（适合临时查询，正式回测请先 update_universe）。
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    if isinstance(codes, str):
        codes = [codes]
    frames = []
    for code in codes:
        df = load_daily(code)
        if df.empty:
            df = fetch_history(code, start, end)
            if df.empty:
                continue
        df = df[(df["date"] >= parse_date(start)) & (df["date"] <= parse_date(end))].copy()
        if not df.empty:
            df["code"] = code
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS + ["code"])
    out = pd.concat(frames, ignore_index=True)
    return out.set_index(["code", "date"]).sort_index()


def get_trade_calendar(start=None, end=None) -> pd.DatetimeIndex:
    """交易日历。优先用 sina 口径（与行情日期一致），失败则用本地数据推断。"""
    import akshare as ak
    start = parse_date(start or config.START_DATE)
    end = parse_date(end or config.END_DATE)
    p = config.RAW_DIR / "trade_calendar.csv"
    try:
        cal = ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        idx = cal["trade_date"].sort_values().reset_index(drop=True)
        idx.to_frame(name="date").to_csv(p, index=False)
    except Exception as e:  # noqa
        log.warning(f"交易日历抓取失败: {e}，用本地数据推断")
        idx = _infer_calendar()
    idx = idx[(idx >= start) & (idx <= end)]
    return pd.DatetimeIndex(idx)


def _infer_calendar() -> pd.DatetimeIndex:
    """从已缓存的日线数据推断交易日集合"""
    files = list((config.RAW_DIR / "daily").glob("*.csv"))
    dates = set()
    for f in files:
        try:
            d = pd.read_csv(f, usecols=["date"])
            dates.update(pd.to_datetime(d["date"]).tolist())
        except Exception:  # noqa
            pass
    return pd.DatetimeIndex(sorted(dates))


# ============================================================
# 五、股票池构建（关键点：剔除 ST / 停牌 / 次新）
# ============================================================

def get_universe(as_of: str = None) -> pd.DataFrame:
    """
    构建"当时可买"的股票池。
    as_of: 基准日，若不传则用最新交易日快照。
    返回列：code, name, market, is_st, is_suspend, list_date, mktcap
    """
    import akshare as ak
    snap = ak.stock_zh_a_spot_tx()

    def norm(c):
        c = str(c)
        if c.startswith(("sh", "sz", "bj")):
            return c
        if c.startswith(("6", "5", "9")):
            return "sh" + c
        if c.startswith("8"):
            return "bj" + c
        return "sz" + c
    snap["code"] = snap["code"].apply(norm)

    name = snap["name"].fillna("")
    out = pd.DataFrame({
        "code": snap["code"],
        "name": name,
        "is_st": name.str.contains("ST", na=False),
        "is_delist": name.str.contains("退", na=False),
        "is_suspend": (snap.get("state", "").eq("停牌")) | (snap["volume"].fillna(0) == 0),
        "mktcap": pd.to_numeric(snap.get("zsz", snap.get("ltsz", np.nan)), errors="coerce") * 1e8,
        "price": pd.to_numeric(snap.get("zxj", np.nan), errors="coerce"),
        "pe_ttm": pd.to_numeric(snap.get("pe_ttm", np.nan), errors="coerce"),
    })

    # 上市日期：腾讯接口不直接给，用 stock_info_a_code_name（东财口径，含上市日）
    try:
        info = ak.stock_info_a_code_name()
        info["code"] = info["code"].apply(norm)
        out = out.merge(info[["code", "list_date"]], on="code", how="left")
    except Exception as e:  # noqa
        log.warning(f"上市日期抓取失败: {e}")
        out["list_date"] = pd.NaT
    return out


def build_backtest_pool(as_of: str, min_list_days: int = None) -> pd.Index:
    """
    回测用股票池。规则：
    1. 基准日仍在交易（未被剔除的缓存列表）
    2. 非 ST、非停牌
    3. 上市满 min_list_days 个交易日
    注意：这里用"截至基准日之前的历史长度"近似上市天数，
    若本地缓存覆盖了完整历史，等价于真实上市天数。
    """
    min_list_days = min_list_days or config.MIN_LIST_DAYS
    snap = get_universe(as_of=as_of)
    cached = set(list_cached_codes())

    codes = []
    cal = get_trade_calendar()
    asof_ts = parse_date(as_of)
    for _, row in snap.iterrows():
        code = row["code"]
        if code not in cached:
            continue
        if config.DROP_ST and row["is_st"]:
            continue
        if config.DROP_SUSPEND and row["is_suspend"]:
            continue
        if not pd.isna(row["price"]) and row["price"] > config.MAX_PRICE:
            continue
        df = load_daily(code)
        if df.empty:
            continue
        hist = df[df["date"] <= asof_ts]
        if len(hist) < min_list_days:
            continue
        last_close = hist["close"].iloc[-1]
        if pd.isna(last_close) or last_close <= 0:
            continue
        codes.append(code)
    return pd.Index(sorted(codes), name="code")
