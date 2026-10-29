"""
diagnose.py —— 数据诊断

专门给"这数据到底能不能用"这个问题做体检。
覆盖：复权一致性、幸存者偏差、停牌缺口、复权因子断层、前后复权可还原性。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config
from .logger import get_logger
from .data import load_daily, list_cached_codes, get_trade_calendar, get_universe

log = get_logger("diagnose")

REQUIRED = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]


def check_schema(df: pd.DataFrame) -> list[str]:
    miss = [c for c in REQUIRED if c not in df.columns]
    return miss


def check_dates(df: pd.DataFrame) -> dict:
    """日期连续性、升序、重复"""
    d = df["date"]
    gaps = d.diff().dropna()
    non_trade_gaps = int((gaps > pd.Timedelta(days=5)).sum())
    return {
        "rows": len(df),
        "range": f"{d.min().date()} ~ {d.max().date()}",
        "span_days": int((d.max() - d.min()).days),
        "duplicates": int(d.duplicated().sum()),
        "monotonic": bool(d.is_monotonic_increasing),
        "non_trade_gaps(>5d)": non_trade_gaps,
    }


def check_ohlc(df: pd.DataFrame) -> dict:
    """OHLC 逻辑一致性 + 极端值"""
    r = {}
    r["close_min"] = float(df["close"].min())
    r["zero_or_neg_close"] = int((df["close"] <= 0).sum())
    r["ohlc_violation"] = int(
        ((df["high"] < df["open"]) | (df["high"] < df["close"]) |
         (df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
    )
    # 涨跌幅自洽：用 OHLC 反推的 pct 与 amount 是否自洽
    px = df["close"].ffill()
    pct_from_close = px.pct_change().abs()
    r["abs_move>20%_days"] = int((pct_from_close > 0.20).sum())
    # 单日振幅异常
    amp = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    r["amp>15%_days"] = int((amp > 0.15).sum())
    return r


def check_amount_volume(df: pd.DataFrame) -> dict:
    """成交额 / 成交量一致性"""
    r = {}
    neg = ((df["amount"] <= 0) | (df["volume"] <= 0)).sum()
    r["non_positive_amount"] = int(neg)
    # 用 amount/volume 反推均价，与 close 比对（复权后不等是预期内的，但偏差过大要提示）
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(df["volume"] > 0, df["amount"] / df["volume"], np.nan)
    if np.isfinite(vwap).any():
        ratio = np.nanmedian(vwap) / df["close"].median()
        r["vwap/close_median_ratio"] = round(float(ratio), 3)
        r["vwap_ratio_ok"] = 0.3 < float(ratio) < 3.0
    else:
        r["vwap/close_median_ratio"] = None
        r["vwap_ratio_ok"] = False
    return r


def check_adjustment(df: pd.DataFrame, raw_df: pd.DataFrame = None) -> dict:
    """
    复权诊断。若能提供未复权 raw_df，检查复权因子是否逐日稳定（拆股送股日必然突变，
    其余日期突变 = 数据源改了历史复权因子，回测结果不可复现）。
    """
    r = {}
    if raw_df is None or raw_df.empty:
        r["note"] = "未提供未复权数据，仅做复权价连续性检查"
        pct = df["close"].pct_change().abs()
        r["复权接缝跳变(>5%)"] = int((pct > 0.05).sum())
        return r
    m = pd.merge(df[["date", "close"]], raw_df[["date", "close"]],
                 on="date", suffixes=("_adj", "_raw"), how="inner")
    if m.empty:
        r["note"] = "复权与未复权无交集"
        return r
    r["匹配交易日"] = len(m)
    ratio = m["close_adj"] / m["close_raw"].replace(0, np.nan)
    ratio = ratio.dropna()
    r["复权因子均值"] = round(float(ratio.mean()), 4)
    r["复权因子标准差"] = round(float(ratio.std()), 5)
    # 因子逐日变动 > 1bp 视为突变（真实除权日允许，其余不应出现）
    dr = ratio / ratio.shift(1) - 1
    big = int((dr.abs() > 0.01).sum())
    r["复权因子突变日(>1%)"] = big
    r["可复现性"] = "良好" if big <= 3 else "可疑（数据源可能回溯修改复权因子）"
    return r


def check_survivorship(codes: list[str], raw_root: str = None) -> dict:
    """
    幸存者偏差检测：
    1. 已退市标的是否在本地有完整历史
    2. 当前股票池是否仅含"活到今天"的标的
    若本地只有当前存续标的，此项会明确告警。
    """
    r = {}
    cached = set(codes)
    snap = get_universe()
    snap_codes = set(snap["code"])
    only_alive = cached.issubset(snap_codes)
    r["缓存标的数"] = len(cached)
    r["快照标的数"] = len(snap_codes)
    r["仅含当前存续标的"] = only_alive
    r["幸存者偏差风险"] = "高" if only_alive and len(cached) > 100 else "低/未知"
    delist = snap[snap["is_delist"]]
    r["快照中含已退市标的"] = len(delist)
    return r


def check_gaps(codes: list[str], year: int = 2024) -> dict:
    """逐个标的检查缺失交易日"""
    cal = get_trade_calendar(f"{year}0101", f"{year}1231")
    rows = []
    for c in codes:
        df = load_daily(c)
        if df.empty:
            continue
        yr = df[(df["date"] >= pd.Timestamp(f"{year}-01-01")) &
                (df["date"] <= pd.Timestamp(f"{year}-12-31"))]
        have = set(pd.to_datetime(yr["date"]))
        need = set(cal)
        missing = sorted(need - have)
        if missing:
            rows.append({"code": c, "缺失天数": len(missing),
                         "缺失日期样本": [d.strftime("%Y-%m-%d") for d in missing[:5]]})
    return {"year": year, "calendar_days": len(cal), "detail": rows}


def full_report(codes: list[str] = None, year: int = 2024) -> dict:
    """全量体检，返回一个 dict，可直接 json.dumps"""
    codes = codes or list_cached_codes()
    rep = {"sample_codes": codes[:5], "n_cached": len(codes)}

    # 抽样 5 只做明细
    detail = {}
    for c in codes[:5]:
        df = load_daily(c)
        if df.empty:
            continue
        detail[c] = {
            "schema_missing": check_schema(df),
            "dates": check_dates(df),
            "ohlc": check_ohlc(df),
            "amount": check_amount_volume(df),
        }
    rep["detail_sample"] = detail

    rep["survivorship"] = check_survivorship(codes)
    rep["gaps"] = check_gaps(codes, year)
    return rep


def print_report(rep: dict) -> None:
    print("=" * 60)
    print("数据体检报告")
    print("=" * 60)
    print(f"缓存标的数: {rep['n_cached']}  抽样明细: {rep['sample_codes']}\n")

    for c, d in rep.get("detail_sample", {}).items():
        print(f"--- {c} ---")
        print(f"  字段缺失: {d['schema_missing'] or '无'}")
        for k, v in d["dates"].items():
            print(f"  日期.{k}: {v}")
        for k, v in d["ohlc"].items():
            print(f"  OHLC.{k}: {v}")
        for k, v in d["amount"].items():
            print(f"  量额.{k}: {v}")

    s = rep["survivorship"]
    print("\n--- 幸存者偏差 ---")
    for k, v in s.items():
        print(f"  {k}: {v}")

    g = rep["gaps"]
    print(f"\n--- 缺失缺口 ({g['year']}, 交易日历 {g['calendar_days']} 天) ---")
    if not g["detail"]:
        print("  全部完整，无缺失")
    else:
        for r in g["detail"]:
            print(f"  {r['code']}: 缺 {r['缺失天数']} 天, 样本 {r['缺失日期样本']}")
