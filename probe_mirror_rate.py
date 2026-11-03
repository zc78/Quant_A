"""
镜像站安全请求速率探测
目标：找出镜像站在混合接口批量请求下，不触发限流的稳定间隔
"""

import os
import time
import sys
import statistics
import tushare as ts

# ============ 配置区 ============
MIRROR_URL = "https://tl.kaixin8.top/tushare/sdk"
TOKEN = os.environ.get("TUSHARE_TOKEN")  # 镜像站如果不需要 token 可留空，但建议设上
SAMPLE_SIZE = 20          # 每个间隔发多少请求
GAP_START = 10.0           # 起始间隔（秒）
GAP_STEP = 20.0           # 每次增大多少
GAP_MAX = 100.0             # 最大间隔上限
SUCCESS_THRESHOLD = 0.95  # 成功率达标线
ROUNDS_NEEDED = 2         # 连续几轮稳定才采纳
# =================================

# 镜像站初始化
if TOKEN:
    ts.set_token(TOKEN)
pro = ts.pro_api()
pro._DataApi__http_url = MIRROR_URL
pro._DataApi__timeout = 30

# 混合测试用例（覆盖你实际会用到的接口）
CASES = [
    ("daily",          {"ts_code": "600519.SH", "start_date": "20250101", "end_date": "20250301"}),
    ("daily",          {"ts_code": "000001.SZ", "start_date": "20250101", "end_date": "20250301"}),
    ("daily",          {"ts_code": "300750.SZ", "start_date": "20250201", "end_date": "20250401"}),
    ("fina_indicator", {"ts_code": "600519.SH", "start_date": "20240101", "end_date": "20240630"}),
    ("fina_indicator", {"ts_code": "000001.SZ", "start_date": "20240101", "end_date": "20240630"}),
    ("stock_basic",    {}),
]

def is_rate_limit(err: str) -> bool:
    """判断是不是限流/上游耗尽类错误"""
    keys = ["rate_limited", "upstream_pool_exhausted", "too many", "limit", "exhausted"]
    return any(k in err.lower() for k in keys)

def call_once(func_name, params, retry=2):
    """单次调用，带简单重试；返回 (ok:bool, latency:float, reason:str)"""
    for attempt in range(retry + 1):
        t0 = time.time()
        try:
            df = getattr(pro, func_name)(**params)
            lat = (time.time() - t0) * 1000
            if df is not None and not df.empty:
                return True, lat, ""
            return False, lat, "empty"
        except Exception as e:
            lat = (time.time() - t0) * 1000
            msg = str(e)
            if is_rate_limit(msg):
                if attempt < retry:
                    time.sleep(1.5 * (attempt + 1))  # 指数退避
                    continue
                return False, lat, "rate_limit"
            else:
                if attempt < retry:
                    time.sleep(0.5)
                    continue
                return False, lat, msg[:40]
    return False, 0.0, "unknown"

def test_gap(gap):
    ok = 0
    lats = []
    reasons = {}
    for i in range(SAMPLE_SIZE):
        fn, p = CASES[i % len(CASES)]
        success, lat, reason = call_once(fn, p)
        if success:
            ok += 1
            lats.append(lat)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
        time.sleep(gap)
    rate = ok / SAMPLE_SIZE
    p95 = statistics.quantiles(lats, n=20)[-1] if lats else 0
    return rate, p95, reasons

def main():
    print("=" * 64)
    print(f"镜像站限流探测  {MIRROR_URL}")
    print(f"样本/间隔={SAMPLE_SIZE}  起始间隔={GAP_START}s  步长={GAP_STEP}s")
    print("=" * 64)
    sys.stdout.flush()

    gap = GAP_START
    stable_rounds = 0
    best_gap = None

    while gap <= GAP_MAX:
        print(f"\n▶ 间隔 {gap:.3f}s ...", flush=True)
        rate, p95, reasons = test_gap(gap)
        print(f"  成功率 {rate*100:5.1f}%  P95延迟 {p95:7.1f}ms  失败分布 {reasons}", flush=True)

        if rate >= SUCCESS_THRESHOLD:
            stable_rounds += 1
            best_gap = gap
            if stable_rounds >= ROUNDS_NEEDED:
                print(f"\n{'='*64}")
                print(f"✅ 稳定间隔: {best_gap:.3f}s")
                print(f"   频率: {1/best_gap:.2f} 次/秒 ≈ {int(60/best_gap)} 次/分钟")
                print(f"   建议 data_source.py 中 REQUEST_GAP = {best_gap:.3f}")
                print(f"   建议 safe_call 重试等待用 1.5s 指数退避，限流后等 30s")
                print(f"{'='*64}")
                return best_gap
            print(f"  ✔ 稳定 ({stable_rounds}/{ROUNDS_NEEDED})，再确认一轮", flush=True)
        else:
            stable_rounds = 0
            print(f"  ⚠ 不稳定，增大间隔", flush=True)
            gap += GAP_STEP

    print(f"\n❌ 到 {GAP_MAX}s 仍不稳定，建议 REQUEST_GAP 直接设 1.0 保底")
    return 1.0

if __name__ == "__main__":
    main()