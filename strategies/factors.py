"""A股技术因子库"""
import numpy as np
import pandas as pd
from typing import Optional


def calculate_chip_concentration(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """筹码集中度指标
    
    基于成交量分布的筹码集中度:
    - 集中度上升：价格上扬 + 缩量（主力控盘）
    - 集中度下降：价格下跌 + 放量（筹码分散）
    """
    vol_ma = volume.rolling(period, min_periods=1).mean()
    vol_ratio = volume / vol_ma
    
    price_change = close.pct_change()
    price_ma = close.rolling(period, min_periods=1).mean()
    price_position = (close - price_ma) / price_ma
    
    concentration = (1 / (vol_ratio + 0.1)) * (1 + price_position)
    concentration = (concentration - concentration.rolling(period, min_periods=1).min())
    concentration = concentration / (concentration.rolling(period, min_periods=1).max() - concentration.rolling(period, min_periods=1).min() + 0.001)
    
    return concentration * 100


def calculate_wr(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """威廉指标"""
    highest_high = high.rolling(period, min_periods=1).max()
    lowest_low = low.rolling(period, min_periods=1).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 0.001)
    return wr


def calculate_cci(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """顺势指标(CCI)"""
    tp = (high + low + close) / 3
    sma = tp.rolling(period, min_periods=1).mean()
    mad = tp.rolling(period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad + 0.001)
    return cci


def calculate_obi(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """OBV能量潮指标"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ma = obv.rolling(period, min_periods=1).mean()
    return obv / (obv_ma + 0.001)


def calculate_mfi(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """资金流量指标"""
    tp = (high + low + close) / 3
    raw_money_flow = tp * volume
    
    money_flow_sign = np.where(tp > tp.shift(1), 1, -1)
    signed_money_flow = raw_money_flow * money_flow_sign
    
    pos_flow = signed_money_flow.where(signed_money_flow > 0, 0).rolling(period, min_periods=1).sum()
    neg_flow = (-signed_money_flow.where(signed_money_flow < 0, 0)).rolling(period, min_periods=1).sum()
    
    mfi = 100 - (100 / (1 + pos_flow / (neg_flow + 0.001)))
    return mfi


def calculate_vpt(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """量价趋势指标"""
    vpt = ((close - close.shift(1)) / close.shift(1) * volume).fillna(0).cumsum()
    vpt_ma = vpt.rolling(period, min_periods=1).mean()
    return vpt / (vpt_ma.abs() + 0.001)


def calculate_ad(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """累积/派发指标"""
    ad = ((close - low) - (high - close)) / (high - low + 0.001) * volume
    ad = ad.fillna(0).cumsum()
    return ad


def calculate_cmf(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin资金流指标"""
    ad = ((close - low) - (high - close)) / (high - low + 0.001) * volume
    ad = ad.fillna(0)
    
    cmf = ad.rolling(period, min_periods=1).sum() / volume.rolling(period, min_periods=1).sum()
    return cmf


def calculate_vwap(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
    """成交量加权平均价"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def calculate_ichimoku(df: pd.DataFrame):
    """一目均衡表指标"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    nine_high = high.rolling(9, min_periods=1).max()
    nine_low = low.rolling(9, min_periods=1).min()
    tenkan_sen = (nine_high + nine_low) / 2
    
    twenty_six_high = high.rolling(26, min_periods=1).max()
    twenty_six_low = low.rolling(26, min_periods=1).min()
    kijun_sen = (twenty_six_high + twenty_six_low) / 2
    
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    fifty_two_high = high.rolling(52, min_periods=1).max()
    fifty_two_low = low.rolling(52, min_periods=1).min()
    senkou_span_b = ((fifty_two_high + fifty_two_low) / 2).shift(26)
    
    chikou_span = close.shift(-26)
    
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术因子"""
    factors = pd.DataFrame(index=df.index)
    
    factors["chip_concentration"] = calculate_chip_concentration(df["close"], df["vol"])
    factors["wr"] = calculate_wr(df["close"], df["high"], df["low"])
    factors["cci"] = calculate_cci(df["close"], df["high"], df["low"])
    factors["obi"] = calculate_obi(df["close"], df["vol"])
    factors["mfi"] = calculate_mfi(df["close"], df["high"], df["low"], df["vol"])
    factors["vpt"] = calculate_vpt(df["close"], df["vol"])
    factors["ad"] = calculate_ad(df["close"], df["high"], df["low"], df["vol"])
    factors["cmf"] = calculate_cmf(df["close"], df["high"], df["low"], df["vol"])
    factors["vwap"] = calculate_vwap(df["close"], df["high"], df["low"], df["vol"])
    
    ichimoku = calculate_ichimoku(df)
    factors["tenkan_sen"] = ichimoku["tenkan_sen"]
    factors["kijun_sen"] = ichimoku["kijun_sen"]
    
    return factors
