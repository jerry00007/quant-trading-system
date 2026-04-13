"""北向资金因子（沪深港通数据）"""
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta


try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class NorthboundFactor:
    """北向资金因子计算器
    
    数据来源: AKShare stock_hsgt_hold_stock_em
    """

    def __init__(self):
        self._cache = {}
        self._last_update = None

    def get_hsgt_data(self, symbol: str, period: str = "daily") -> Optional[pd.DataFrame]:
        """获取沪深港通持股数据
        
        Args:
            symbol: 股票代码 (如: 000001.SZ)
            period: 数据周期 (daily/weekly/monthly)
        """
        if not AKSHARE_AVAILABLE:
            return None

        cache_key = f"{symbol}_{period}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if period == "daily":
                df = ak.stock_hsgt_hold_stock_em(symbol=symbol)
            elif period == "weekly":
                df = ak.stock_hsgt_hold_stock_em(symbol=symbol)
            else:
                df = ak.stock_hsgt_hold_stock_em(symbol=symbol)
            
            if df is not None and not df.empty:
                self._cache[cache_key] = df
                return df
        except Exception:
            pass

        return None

    def get_fund_flow(self, symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
        """获取资金流向数据"""
        if not AKSHARE_AVAILABLE:
            return None

        try:
            df = ak.stock_hsgt_hist_em(symbol=symbol.replace(".SZ", "").replace(".SH", ""))
            if df is not None and not df.empty:
                df = df.tail(days)
                return df
        except Exception:
            pass

        return None

    def calculate_north_flow_factor(self, close: pd.Series, hsgt_data: pd.DataFrame) -> pd.Series:
        """计算北向资金净流入因子
        
        基于北向资金持股比例变化和价格走势的关系:
        - 北向增持 + 价格上涨 -> 强势
        - 北向减持 + 价格下跌 -> 弱势
        """
        if hsgt_data is None or hsgt_data.empty:
            return pd.Series(0, index=close.index)

        try:
            if "持股数量" in hsgt_data.columns:
                hold_ratio = hsgt_data["持股数量"].pct_change()
            elif "持股比例" in hsgt_data.columns:
                hold_ratio = hsgt_data["持股比例"].pct_change()
            else:
                return pd.Series(0, index=close.index)

            price_change = close.pct_change()

            north_factor = hold_ratio * 10 + price_change * 5
            north_factor = north_factor.fillna(0)
            
            return north_factor.reindex(close.index, fill_value=0)
        except Exception:
            return pd.Series(0, index=close.index)

    def calculate_hsgt_ma_ratio(self, close: pd.Series, hsgt_data: pd.DataFrame, period: int = 5) -> pd.Series:
        """计算北向资金MA交叉因子"""
        if hsgt_data is None or hsgt_data.empty:
            return pd.Series(0, index=close.index)

        try:
            if "持股数量" in hsgt_data.columns:
                hold_col = "持股数量"
            elif "持股比例" in hsgt_data.columns:
                hold_col = "持股比例"
            else:
                return pd.Series(0, index=close.index)

            hold_values = hsgt_data[hold_col].values
            if len(hold_values) < period:
                return pd.Series(0, index=close.index)

            hold_ma = pd.Series(hold_values).rolling(period, min_periods=1).mean().values
            hold_current = hold_values[-1]

            ratio = (hold_current - hold_ma) / (hold_ma + 0.001)
            
            result = pd.Series(ratio, index=close.index[-len(ratio):] if len(ratio) <= len(close) else close.index)
            result = result.reindex(close.index, fill_value=0)
            
            return result
        except Exception:
            return pd.Series(0, index=close.index)


def calculate_chip_distribution_index(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """筹码分布指数
    
    衡量筹码在持股者中的分布状态:
    - 高位密集：主力可能出货
    - 低位密集：主力可能吸筹
    """
    ret = close.pct_change()
    
    volume_ma = volume.rolling(period, min_periods=1).mean()
    vol_ratio = volume / (volume_ma + 0.001)
    
    price_std = close.rolling(period, min_periods=1).std()
    price_zscore = (close - close.rolling(period, min_periods=1).mean()) / (price_std + 0.001)
    
    cdi = vol_ratio * (1 + price_zscore * 0.1)
    
    cdi = (cdi - cdi.rolling(period, min_periods=1).min())
    cdi = cdi / (cdi.rolling(period, min_periods=1).max() - cdi.rolling(period, min_periods=1).min() + 0.001)
    
    return cdi * 100


def calculate_turnover_distribution(turnover_rate: pd.Series, period: int = 20) -> pd.Series:
    """换手率分布因子
    
    高换手率区间占比:
    - 高换手占比上升 -> 筹码分散
    - 高换手占比下降 -> 筹码集中
    """
    turnover_ma = turnover_rate.rolling(period, min_periods=1).mean()
    turnover_std = turnover_rate.rolling(period, min_periods=1).std()
    
    turnover_zscore = (turnover_rate - turnover_ma) / (turnover_std + 0.001)
    
    high_turnover_ratio = (turnover_zscore > 1).rolling(period, min_periods=1).sum() / period
    
    return high_turnover_ratio * 100
