"""
镜像站稳定性压测脚本
用法：
    python probe_mirror.py
可选环境变量：
    MIRROR_URL   镜像站地址（默认 https://tl.kaixin8.top/tushare/sdk）
    TS_TOKEN     tushare token
"""
import os
import time
import json
import statistics
from collections import Counter

import requests

# ---------- 配置 ----------
MIRROR_URL = os.getenv("MIRROR_URL", "https://tl.kaixin8.top/tushare/sdk")
TOKEN = os.getenv("TS_TOKEN")  # 留空会用 ts 已保存的 token
N = int(os.getenv("PROBE_N", "30"))          # 请求次数
CONCURRENCY = int(os.getenv("PROBE_CONC", "1"))  # 并发数（先测 1，再测 3）
CONNECT_TIMEOUT = 5      # 连接超时
READ_TIMEOUT = 20        # 读取超时
GAP = 0.3                # 每次请求间隔（秒），避免自己把自己限流

# 测试用的轻量请求体（daily 接口，少量数据）
def make_payload():
    return {
        "api_name": "daily",
        "token": TOKEN,
        "params": {
            "ts_code": "600519.SH",
            "trade_date": "20240115"
        },
        "fields": "ts_code,trade_date,open,close,vol"
    }


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(p * len(xs)) - 1))
    return xs[k]


def probe_once(session):
    """单次探测，返回 (ok, cost_ms, category, detail)"""
    payload = make_payload()
    t0 = time.perf_counter()
    try:
        resp = session.post(
            MIRROR_URL,
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        cost = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            return False, cost, "bad_json", f"http={resp.status_code}"

        if body.get("code") == 0 and body.get("data"):
            return True, cost, "ok", f"http={resp.status_code}"
        else:
            msg = str(body.get("msg", "")).lower()
            if "rate" in msg or "额度" in msg or "limit" in msg:
                return False, cost, "rate_limited", msg[:60]
            return False, cost, f"biz_err(http={resp.status_code})", msg[:60]
    except requests.exceptions.ConnectTimeout:
        return False, (time.perf_counter() - t0) * 1000, "connect_timeout", ""
    except requests.exceptions.ReadTimeout:
        return False, (time.perf_counter() - t0) * 1000, "read_timeout", ""
    except requests.exceptions.ConnectionError as e:
        return False, (time.perf_counter() - t0) * 1000, "connection_err", str(e)[:50]
    except Exception as e:
        return False, (time.perf_counter() - t0) * 1000, "other_err", str(e)[:50]


def run_probe():
    print(f"目标: {MIRROR_URL}")
    print(f"次数: {N}  并发: {CONCURRENCY}  超时: ({CONNECT_TIMEOUT}s连接/{READ_TIMEOUT}s读取)")
    print("-" * 60)

    session = requests.Session()
    # 挂载重试适配器（只对网络层瞬断重试，不影响统计）
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=0)  # 不自动重试，真实反映稳定性
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    ok_times = []
    categories = Counter()
    details = []

    for i in range(N):
        ok, cost, cat, detail = probe_once(session)
        categories[cat] += 1
        if ok:
            ok_times.append(cost)
        details.append((i + 1, ok, cost, cat, detail))
        # 打印单行进度
        flag = "✓" if ok else "✗"
        print(f"[{flag}] #{i+1:02d} {cost:7.1f}ms  {cat:16s} {detail}")
        if CONCURRENCY == 1:
            time.sleep(GAP)

    # ---------- 汇总 ----------
    print("=" * 60)
    print("汇总结果")
    print("=" * 60)
    total = N
    success = len(ok_times)
    print(f"总请求数      : {total}")
    print(f"成功数        : {success}  (成功率 {success/total*100:.1f}%)")
    print(f"失败数        : {total - success}")
    print(f"错误分类      : {dict(categories)}")
    if ok_times:
        print(f"延迟(ms) 平均 : {statistics.mean(ok_times):.1f}")
        print(f"延迟(ms) P50  : {percentile(ok_times, 0.50):.1f}")
        print(f"延迟(ms) P95  : {percentile(ok_times, 0.95):.1f}")
        print(f"延迟(ms) Max  : {max(ok_times):.1f}")
    else:
        print("无成功样本，无法统计延迟")

    # ---------- 判定 ----------
    print("-" * 60)
    print("稳定性判定（参考阈值）:")
    success_rate = success / total
    timeout_rate = (categories.get("connect_timeout", 0)
                    + categories.get("read_timeout", 0)) / total
    if success_rate >= 0.95 and timeout_rate <= 0.01:
        print("✅ 优秀：可作为主力数据源")
    elif success_rate >= 0.85:
        print("⚠️  可用但偶发抖动：必须加重试+缓存，别裸跑")
    else:
        print("❌ 不稳定：建议切回官方 api.tushare.pro 或换镜像")

    return {
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "categories": dict(categories),
        "p95_ms": percentile(ok_times, 0.95) if ok_times else None,
    }


if __name__ == "__main__":  
    run_probe()