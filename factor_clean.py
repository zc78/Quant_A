"""
因子预处理：缺失值、去极值、标准化、中性化
"""
import numpy as np
import pandas as pd
from scipy.stats import mstats


def winsorize(frame, lower=0.01, upper=0.99):
    """截面分位数去极值"""
    def _w(s):
        if s.notna().sum() < 10:
            return s
        return pd.Series(mstats.winsorize(s.dropna(), limits=(lower, upper)),
                         index=s.dropna().index).reindex(s.index)
    return frame.apply(_w, axis=1)


def zscore(frame):
    """截面 Z-score 标准化"""
    mu = frame.mean(axis=1)
    sigma = frame.std(axis=1)
    return (frame.sub(mu, axis=0) / sigma.replace(0, np.nan), axis=0)


def neutralize_market_cap(frame, log_mktcap):
    """
    市值中性化：对每行做 因子 ~ log(市值) 的回归，取残差
    frame: 因子矩阵
    log_mktcap: 对数流通市值矩阵（同形状）
    """
    resid = {}
    for date, row in frame.iterrows():
        y = row
        x = log_mktcap.loc[date]
        mask = y.notna() & x.notna()
        if mask.sum() < 10:
            resid[date] = y
            continue
        X = np.c_[np.ones(mask.sum()), x[mask].values]
        Y = y[mask].values
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ beta
        r = pd.Series(Y - pred, index=y[mask].index)
        full = pd.Series(np.nan, index=y.index)
        full[r.index] = r
        resid[date] = full
    return pd.DataFrame(resid).T


def clean_factor(raw_factor, log_mktcap=None, do_neutralize=True):
    """
    标准预处理流水线：
    1. 缺失值：截面用中位数填充
    2. 去极值
    3. 标准化
    4. （可选）市值中性化
    """
    # 1. 缺失值填充
    f = raw_factor.fillna(raw_factor.median(axis=1), axis=0)
    # 2. 去极值
    f = winsorize(f)
    # 3. 标准化
    f = zscore(f)
    # 4. 中性化
    if do_neutralize and log_mktcap is not None:
        f = neutralize_market_cap(f, log_mktcap)
        f = zscore(f)  # 中性化后再标准化一次
    return f