"""
预热缓存：拉取所有需要的股票数据，之后回测完全离线
"""
from data_source import get_daily, get_daily_basic, get_fina_indicator, get_trade_calendar

codes = [
    "600519.SH", "000858.SZ", "601318.SH", "600036.SH",
    "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"
]
start = "20200101"
end   = "20261029"

print("=== 预热交易日历 ===")
get_trade_calendar(start, end)

for code in codes:
    print(f"=== {code} ===")
    print("  日线...", end=" ")
    get_daily(code, start, end)
    print("OK")
    print("  估值...", end=" ")
    get_daily_basic(code, start, end)
    print("OK")
    print("  财务...", end=" ")
    get_fina_indicator(code, start, end)
    print("OK")

print("缓存预热完成！")