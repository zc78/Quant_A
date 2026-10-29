"""
因子计算模块
输入：股票池 + 回测区间
输出：横截面因子面板（date × ts_code）

因子来源：
  - 量价因子：日线行情本地计算
  - 估值/流动性：daily_basic
  - 质量/成长：fina_indicator（按报告期对齐到交易日）
"""
import numpy as np
import pandas as pd
from data_source import get_daily, get_daily_basic, get_fina_indicator, get_trade_calendar


def _align_fina_to_daily(fina_df, trade_dates):
    """
    把季报数据按公告日向前填充到每个交易日。
    避免未来函数：只用已公告的数据。
    """
    fina_df = fina_df.copy()
    fina_df['ann_date'] = pd.to_datetime(fina_df['ann_date'].astype(str), errors='coerce')
    fina_df = fina_df.sort_values('ann_date')
    # 以 ann_date 为索引，重采样到交易日序列
    fina_df = fina_df.set_index('ann_date')
    # 取数值列
    num_cols = fina_df.select_dtypes(include=[np.number]).columns.tolist()
    filled = fina_df[num_cols].reindex(trade_dates).ffill()
    filled.index = trade_dates
    return filled


def build_factor_panel(ts_codes, start_date, end_date):
    """
    构建因子面板，返回 dict：
      factor_name -> DataFrame(index=date, columns=ts_code)
    """
    # 交易日序列
    cal = get_trade_calendar(start_date, end_date)
    cal['cal_date'] = pd.to_datetime(cal['cal_date'])
    trade_dates = cal[cal['is_open'] == 1]['cal_date']
    trade_dates = trade_dates[(trade_dates >= pd.to_datetime(start_date)) &
                              (trade_dates <= pd.to_datetime(end_date))]

    factors = {}

    # ---------- 量价因子 ----------
    close_panel = {}
    vol_panel = {}
    for code in ts_codes:
        df = get_daily(code, start_date, end_date)
        df['date'] = pd.to_datetime(df['date'])
        s = df.set_index('date')['close'].reindex(trade_dates)
        close_panel[code] = s
        v = df.set_index('date')['volume'].reindex(trade_dates)
        vol_panel[code] = v

    close_df = pd.DataFrame(close_panel)
    vol_df = pd.DataFrame(vol_panel)

    # 动量 / 反转
    factors['momentum_20'] = close_df.pct_change(20)
    factors['momentum_60'] = close_df.pct_change(60)
    # 波动率
    factors['volatility_20'] = close_df.pct_change().rolling(20).std() * np.sqrt(252)
    # 换手代理（用成交量变化率）
    factors['volume_ratio'] = vol_df / vol_df.rolling(20).mean()

    # ---------- 估值 / 流动性因子 ----------
    for code in ts_codes:
        db = get_daily_basic(code, start_date=start_date, end_date=end_date)
        db['date'] = pd.to_datetime(db['date'])
        db = db.set_index('date').reindex(trade_dates)
        if 'pe_ttm' not in factors:
            factors['pe_ttm'] = pd.DataFrame(index=trade_dates)
            factors['pb'] = pd.DataFrame(index=trade_dates)
            factors['turnover_rate'] = pd.DataFrame(index=trade_dates)
            factors['circ_mv'] = pd.DataFrame(index=trade_dates)
        factors['pe_ttm'][code] = db['pe_ttm']
        factors['pb'][code] = db['pb']
        factors['turnover_rate'][code] = db['turnover_rate']
        factors['circ_mv'][code] = db['circ_mv']

    # ---------- 质量 / 成长因子 ----------
    for code in ts_codes:
        try:
            fina = get_fina_indicator(code, start_date, end_date)
            aligned = _align_fina_to_daily(fina, trade_dates)
            for col in ['roe', 'roe_waa', 'grossprofit_margin', 'netprofit_yoy', 'or_yoy']:
                if col not in factors:
                    factors[col] = pd.DataFrame(index=trade_dates)
                factors[col][code] = aligned[col] if col in aligned else np.nan
        except Exception as e:
            print(f"  {code} 财务因子失败: {e}")

    return factors


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
        'pe_ttm': -1,
        'pb': -1,
        'turnover_rate': 0,
        'circ_mv': 0,
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