"""
数据源调试脚本 —— 验证 tushare 数据链路是否畅通
运行：python debug_datasource.py
"""
import os
import time
import sys
import traceback
import numpy as np
import pandas as pd

DEBUG = os.getenv("DATA_SOURCE_DEBUG", "1") == "1"

# -------------------------------------------------------------------
# 0. 环境检查
# -------------------------------------------------------------------
def check_env():
    print("\n" + "=" * 60)
    print("【1】环境检查")
    print("=" * 60)
    try:
        import tushare as ts
        print(f"✓ tushare 版本: {ts.__version__}")
    except Exception as e:
        print(f"✗ tushare 未安装: {e}")
        sys.exit(1)

    token = os.getenv("TUSHARE_TOKEN")
    print(f"{'✓' if token else '✗'} TUSHARE_TOKEN 环境变量: {'已设置' if token else '未设置'}")
    return token


# -------------------------------------------------------------------
# 1. 尝试导入项目自有 data_source，失败则回退原生 pro
# -------------------------------------------------------------------
def init_pro():
    print("\n" + "=" * 60)
    print("【2】初始化 Pro 接口")
    print("=" * 60)
    import tushare as ts

    # 优先用项目里配置的 token
    token = os.getenv("TUSHARE_TOKEN")
    if token:
        ts.set_token(token)
    try:

        pro = ts.pro_api()

        # 代理中转用
        pro._DataApi__http_url = "https://tl.kaixin8.top/tushare/sdk"

        # 轻量探活：取一天日线
        t0 = time.time()
        df = pro.daily(trade_date="20240115", limit=1)
        cost = time.time() - t0
        if df is not None and not df.empty:
            print(f"✓ pro_api 初始化成功，探活耗时 {cost:.2f}s，返回 {len(df)} 行")
        else:
            print(f"⚠ pro_api 返回空（可能无该日数据），但接口未报错，耗时 {cost:.2f}s")
        return pro
    except Exception as e:
        print(f"✗ pro_api 初始化/探活失败: {e}")
        traceback.print_exc()
        sys.exit(1)


# -------------------------------------------------------------------
# 2. 单接口测试（带超时与字段校验）
# -------------------------------------------------------------------
def test_interface(name, func, required_cols=None):
    print(f"\n--- 接口测试: {name} ---")
    t0 = time.time()
    try:
        df = func()
        cost = time.time() - t0
        if df is None or df.empty:
            print(f"⚠ 返回空 DataFrame（耗时 {cost:.2f}s）")
            return None
        print(f"✓ 成功，{len(df)} 行 × {len(df.columns)} 列，耗时 {cost:.2f}s")
        if DEBUG:
            print("  字段:", list(df.columns)[:20])
            print("  前1行:\n", df.head(1).to_string())
        # 字段完整性校验
        if required_cols:
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"✗ 缺失关键字段: {missing}")
            else:
                print(f"✓ 关键字段齐全: {required_cols}")
        return df
    except Exception as e:
        cost = time.time() - t0
        err = str(e)
        # 错误分类
        if "timed out" in err or "Read timed out" in err:
            print(f"✗ 超时（{cost:.2f}s）: {err[:120]}")
        elif "2002" in err or "权限" in err or "积分" in err:
            print(f"✗ 权限/积分不足: {err[:120]}")
        elif "freq" in err or "限流" in err or "429" in err:
            print(f"✗ 触发限流: {err[:120]}")
        else:
            print(f"✗ 其他错误（{cost:.2f}s）: {err[:160]}")
        return None


# -------------------------------------------------------------------
# 3. 缓存测试（如果项目有 data_source.clear_cache / _read_cache）
# -------------------------------------------------------------------
def test_cache():
    print("\n" + "=" * 60)
    print("【4】缓存层测试")
    print("=" * 60)
    try:
        import data_source as ds
        have_cache = hasattr(ds, "clear_cache") and hasattr(ds, "_write_cache") and hasattr(ds, "_read_cache")
        if not have_cache:
            print("⚠ data_source 未暴露缓存函数，跳过缓存测试")
            return
        # 构造假数据写入再读出
        dummy = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        key = "debug_test"
        ds._write_cache(dummy, ds._cache_path(key))
        got = ds._read_cache(ds._cache_path(key))
        if got is not None and len(got) == 2:
            print("✓ 缓存写入/读取正常")
        else:
            print("✗ 缓存读写异常")
        ds.clear_cache()
        print("✓ 缓存清理完成")
    except Exception as e:
        print(f"⚠ 缓存测试跳过: {e}")


# -------------------------------------------------------------------
# 4. 批量探活
# -------------------------------------------------------------------
def batch_probe(pro, codes):
    print("\n" + "=" * 60)
    print("【5】批量股票探活（日线 + 财务）")
    print("=" * 60)
    for code in codes:
        # 日线
        try:
            t0 = time.time()
            d = pro.daily(ts_code=code, start_date="20240101", end_date="20240131")
            dcost = time.time() - t0
            dstatus = f"✓{len(d)}行/{dcost:.1f}s" if (d is not None and not d.empty) else f"⚠空/{dcost:.1f}s"
        except Exception as e:
            dstatus = f"✗{str(e)[:40]}"

        # 财务
        try:
            t0 = time.time()
            f = pro.fina_indicator(ts_code=code, start_date="20230101", end_date="20231231")
            fcost = time.time() - t0
            fstatus = f"✓{len(f)}行/{fcost:.1f}s" if (f is not None and not f.empty) else f"⚠空/{fcost:.1f}s"
            # 检查毛利率字段
            if f is not None and not f.empty:
                if "grossprofit_margin" not in f.columns:
                    fstatus += " (无grossprofit_margin字段!)"
        except Exception as e:
            fstatus = f"✗{str(e)[:40]}"

        print(f"{code:12s} 日线[{dstatus:18s}] 财务[{fstatus}]")


# -------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------
if __name__ == "__main__":
    codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
             "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]

    check_env()
    pro = init_pro()

    print("\n" + "=" * 60)
    print("【3】关键接口测试")
    print("=" * 60)

    # 日线行情
    test_interface(
        "pro.daily（日线）",
        lambda: pro.daily(ts_code="600519.SH", start_date="20240101", end_date="20240131"),
        required_cols=["trade_date", "open", "close", "high", "low", "vol"]
    )

    # 财务指标
    test_interface(
        "pro.fina_indicator（财务）",
        lambda: pro.fina_indicator(ts_code="600519.SH", start_date="20200101", end_date="20231231"),
        required_cols=["ts_code", "end_date", "roe"]
    )

    # 每日指标（估值/换手）
    test_interface(
        "pro.daily_basic（每日指标）",
        lambda: pro.daily_basic(ts_code="600519.SH", start_date="20240101", end_date="20240131"),
        required_cols=["ts_code", "trade_date"]
    )

    # 交易日历
    test_interface(
        "pro.trade_cal（交易日历）",
        lambda: pro.trade_cal(exchange="SSE", start_date="20240101", end_date="20240110", is_open="1")
    )

    # 项目自有函数（如果存在）
    try:
        import data_source as ds
        print("\n--- 项目 data_source.get_daily ---")
        t0 = time.time()
        dd = ds.get_daily("600519.SH", "20240101", "20240131")
        print(f"✓ 返回 {len(dd)} 行，耗时 {time.time()-t0:.2f}s" if dd is not None and not dd.empty
              else f"⚠ 返回空，耗时 {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"⚠ 项目 data_source 调用异常（可忽略）: {str(e)[:100]}")

    test_cache()
    batch_probe(pro, codes)

    print("\n" + "=" * 60)
    print("调试完成。请根据上面 ✓/⚠/✗ 标记定位问题。")
    print("=" * 60)
