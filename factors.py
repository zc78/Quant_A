"""
因子计算模块
输入：股票池 + 回测区间
输出：横截面因子矩阵（date × ts_code）

先实现 4 个基础因子：
  1. momentum_20    : 20日动量（收益）
  2. momentum_60    : 60日动量
  3. volatility_20  : 20日年化波动率
  4. pe_ttm         : 估值（来自 daily_basic）
"""
import numpy as np
import pandas as pd
from data_source import get_daily, get_daily_basic


def build_factor_panel(ts_codes, start_date, end_date):
    """
    构建因子面板。

    返回
    ----
    factor_dict : dict
        每个因子名 -> DataFrame(date 为索引, 列为 ts_code)
    """
    # ---------- 1. 从日线计算量价因子 ----------
    close_panel = {}
    for code in ts_codes:
        df = get_daily(code, start_date, end_date)
        df['date'] = pd.to_datetime(df['date'])
        s = df.set_index('date')['close']
        close_panel[code] = s

    close_df = pd.DataFrame(close_panel)  # 行=日期, 列=股票

    # 动量
    mom20 = close_df.pct_change(20)
    mom60 = close_df.pct_change(60)

    # 波动率（20日收益率标准差，年化）
    vol20 = close_df.pct_change().rolling(20).std() * np.sqrt(252)

    # ---------- 2. 从 daily_basic 取估值因子 ----------
    pe_panel = {}
    for code in ts_codes:
        try:
            db = get_daily_basic(code, start_date=start_date, end_date=end_date)
            s = db.set_index('date')['pe_ttm']
            pe_panel[code] = s
        except Exception as e:
            print(f"  {code} 估值数据获取失败: {e}")
            pe_panel[code] = np.nan

    pe_df = pd.DataFrame(pe_panel)

    # ---------- 3. 组装因子字典 ----------
    factors = {
        'momentum_20': mom20,
        'momentum_60': mom60,
        'volatility_20': vol20,
        'pe_ttm': pe_df,
    }

    return factors


def neutralize_market_cap():
    """后续做市值中性化用，先留接口"""
    pass


if __name__ == "__main__":
    codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "600276.SH"]
    factors = build_factor_panel(codes, "20200101", "20261029")

    print("因子面板构建完成")
    for name, f in factors.items():
        print(f"\n因子: {name}  形状: {f.shape}")
        print(f.tail(3))