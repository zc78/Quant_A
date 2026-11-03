"""
仅用 daily 接口探测镜像站限流边界
从 REQUEST_GAP=0.35s 开始，逐步减小间隔，找到不触发限流的最快频率
"""

import os
import time
import sys
import statistics
import tushare as ts

MIRROR_URL = "https://tl.kaixin8.top/tushare/sdk"
TOKEN = os.environ.get("TUSHARE_TOKEN", "")
SAMPLE_SIZE = 30          # 每个间隔发30次请求
GAP_LIST = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]  # 从当前值往下探测
SUCCESS_THRESHOLD = 0.93  # 成功率≥93%就算稳定（允许偶尔超时）

# 初始化镜像站
if TOKEN:
    ts.set_token(TOKEN)
pro = ts.pro_api()
pro._DataApi__http_url = MIRROR_URL
pro._DataApi__timeout = 30

# 只用 daily 接口，参数固定
STOCK_CODES = ["600519.SH","000001.SZ","300750.SZ","002415.SZ","601318.SH",
               "000858.SZ","600036.SH","002594.SZ","300059.SZ","601166.SH"]

def test_gap(gap):
    ok = 0
    lats = []
    failures = {}
    for i in range(SAMPLE_SIZE):
        code = STOCK_CODES[i % len(STOCK_CODES)]
        t0 = time.time()
        try:
            df = pro.daily(ts_code=code, start_date="20250101", end_date="20250301")
            lat = (time.time() - t0) * 1000
            if df is not None and not df.empty:
                ok += 1
                lats.append(lat)
            else:
                failures["empty"] = failures.get("empty", 0) + 1
        except Exception as e:
            lat = (time.time() - t0) * 1000
            msg = str(e).lower()
            if "rate_limited" in msg or "upstream_pool_exhausted" in msg:
                failures["rate_limit"] = failures.get("rate_limit", 0) + 1
            elif "timeout" in msg:
                failures["timeout"] = failures.get("timeout", 0) + 1
            else:
                failures[f"other:{msg[:30]}"] = failures.get(f"other:{msg[:30]}", 0) + 1
        time.sleep(gap)
    rate = ok / SAMPLE_SIZE
    p95 = statistics.quantiles(lats, n=20)[-1] if lats else 0
    return rate, p95, failures

def main():
    print("=" * 68)
    print(f"镜像站 daily 接口限流探测  {MIRROR_URL}")
    print(f"每个间隔 {SAMPLE_SIZE} 次请求，成功率≥{SUCCESS_THRESHOLD*100:.0f}% 即认为稳定")
    print("=" * 68)
    sys.stdout.flush()

    for gap in GAP_LIST:
        print(f"\n▶ 间隔 {gap:.2f}s ...", end=" ", flush=True)
        rate, p95, failures = test_gap(gap)
        status = "✅ 稳定" if rate >= SUCCESS_THRESHOLD else "⚠️ 不稳定"
        print(f"{status}  成功率 {rate*100:5.1f}%  P95延迟 {p95:7.1f}ms  失败: {failures}", flush=True)

    print("\n建议：取最后一个稳定间隔作为 REQUEST_GAP，若不稳则用 0.35s")

if __name__ == "__main__":
    main()