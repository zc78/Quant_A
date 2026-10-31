"""
因子计算模块
输入：股票池 + 回测区间
输出：横截面因子面板（date × ts_code）

因子来源：
  - 量价因子：日线行情本地计算
  - 质量/成长：fina_indicator（按报告期对齐到交易日）
"""
import numpy as np
import pandas as pd
from data_source import get_daily, get_daily_basic, get_fina_indicator, get_trade_calendar


def _align_fina_to_daily(fina_df, trade_dates):
    """
    把季报数据按公告日向前填充到每个交易日。
    避免未来函数：只用已公告的数据。
    处理重复公告日：按 end_date 降序取第一条（最新报告期）。
    """
    fina_df = fina_df.copy()
    fina_df['ann_date'] = pd.to_datetime(fina_df['ann_date'].astype(str), errors='coerce')
    # 按 ann_date 和 end_date 排序，保留每个 ann_date 的最新一条
    fina_df = fina_df.sort_values(['ann_date', 'end_date'], ascending=[True, False])
    fina_df = fina_df.drop_duplicates(subset=['ann_date'], keep='first')
    fina_df = fina_df.set_index('ann_date')
    num_cols = fina_df.select_dtypes(include=[np.number]).columns.tolist()
    filled = fina_df[num_cols].reindex(trade_dates).ffill()
    filled.index = trade_dates
    return filled


def build_factor_panel(ts_codes, start_date, end_date):
    """
    构建因子面板：日期索引为 datetime64[ns]，列为股票代码，值为因子值
    包含量价因子 + 质量/成长因子（不含估值因子）
    """
    from data_source import get_trade_calendar, get_daily, get_fina_indicator

    # 1. 交易日历（统一转为 datetime64[ns]）
    cal = get_trade_calendar(start_date, end_date)
    if cal.empty or 'cal_date' not in cal.columns:
        raise RuntimeError("无法获取交易日历数据，请确保网络畅通或缓存存在。")
    cal['cal_date_dt'] = pd.to_datetime(cal['cal_date'].astype(str))
    trade_dates = cal[cal['is_open'] == 1]['cal_date_dt']
    trade_dates = trade_dates[(trade_dates >= pd.to_datetime(start_date)) &
                              (trade_dates <= pd.to_datetime(end_date))]
    trade_dates = trade_dates.sort_values().reset_index(drop=True)
    print(f"[INFO] 交易日数量: {len(trade_dates)}")

    # 2. 量价因子（动量、波动、换手率）
    close_panel = {}
    volume_panel = {}
    for code in ts_codes:
        df = get_daily(code, start_date, end_date)
        if df.empty:
            print(f"[WARN] 跳过 {code}（无日线数据）")
            continue
        # 修正字段名：trade_date 和 vol
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index('trade_date')
        close_panel[code] = df['close'].reindex(trade_dates)
        volume_panel[code] = df['vol'].reindex(trade_dates)

    close_df = pd.DataFrame(close_panel)
    volume_df = pd.DataFrame(volume_panel)

    # 动量因子（过去20/60日收益率，shift(1)避免未来函数）
    mom20 = close_df.pct_change(20).shift(1)
    mom60 = close_df.pct_change(60).shift(1)

    # 波动因子（过去20日收益率标准差）
    ret = close_df.pct_change()
    vol20 = ret.rolling(20).std().shift(1)

    # 换手率因子（过去5日平均成交量 / 过去20日平均成交量）
    volume_ma5 = volume_df.rolling(5).mean()
    volume_ma20 = volume_df.rolling(20).mean()
    volume_ratio = (volume_ma5 / volume_ma20).shift(1)

    # 存储所有因子
    all_factors = {}
    all_factors['momentum_20'] = mom20
    all_factors['momentum_60'] = mom60
    all_factors['volatility_20'] = vol20
    all_factors['volume_ratio'] = volume_ratio

    # 3. 质量/成长因子（ROE, 毛利率, 净利同比, 营收同比）
    stock_factors = {}  # {code: {factor_name: series}}
    for code in ts_codes:
        try:
            df = get_fina_indicator(code, start_date, end_date)
            if df.empty:
                continue
            # ann_date 是整数格式，转为 datetime
            df['ann_date_dt'] = pd.to_datetime(df['ann_date'].astype(str))
            # 处理重复公告日：按 end_date 降序保留最新一条
            df = df.sort_values(['ann_date_dt', 'end_date'], ascending=[True, False])
            df = df.drop_duplicates(subset=['ann_date_dt'], keep='first')
            df = df.set_index('ann_date_dt')
            if code not in stock_factors:
                stock_factors[code] = {}
            for col in ['roe', 'roe_waa', 'grossprofit_margin', 'netprofit_yoy', 'or_yoy']:
                if col in df.columns:
                    stock_factors[code][col] = df[col].reindex(trade_dates).ffill()
        except Exception as e:
            print(f"  {code} 财务因子失败: {e}")

    # 4. 整理为面板：每个因子是一个 DataFrame（日期x股票）
    result = {}
    # 全局因子
    for name in ['momentum_20', 'momentum_60', 'volatility_20', 'volume_ratio']:
        if name in all_factors:
            result[name] = all_factors[name]

    # 个股因子（仅财务因子）
    for name in ['roe', 'roe_waa', 'grossprofit_margin', 'netprofit_yoy', 'or_yoy']:
        cols = {}
        for code in ts_codes:
            if code in stock_factors and name in stock_factors[code]:
                cols[code] = stock_factors[code][name]
        if cols:
            result[name] = pd.DataFrame(cols, index=trade_dates)

    print(f"[INFO] 共生成 {len(result)} 个因子面板")
    print(f"[INFO] 因子列表: {list(result.keys())}")
    return result


def factor_direction():
    """
    返回每个因子的预期方向：
      1  = 因子值越大越好
     -1  = 因子值越小越好（如 PE、波动率）
      0  = 方向待定（靠 IC 测试决定）
    """
    return {
        'momentum_20': 0,
        'momentum_60': 0,
        'volatility_20': -1,
        'volume_ratio': 0,
        'roe': 1,
        'roe_waa': 1,
        'grossprofit_margin': 1,
        'netprofit_yoy': 1,
        'or_yoy': 1,
    }


if __name__ == "__main__":
    codes = ["600519.SH", "000858.SZ", "601318.SH", "600036.SH",
             "000333.SZ", "300750.SZ", "002415.SZ", "600276.SH"]
    factors = build_factor_panel(codes, "20200101", "20261029")
    print(f"构建了 {len(factors)} 个因子")
    for name, f in factors.items():
        print(f"  {name:20s} 形状={f.shape}  非空比例={f.notna().mean().mean():.1%}")