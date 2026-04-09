"""AKShare 数据下载封装"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

from .base_source import BaseDataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AKShareDataSource(BaseDataSource):
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @staticmethod
    def _rename_daily_columns(df: pd.DataFrame, original_code: str) -> pd.DataFrame:
        column_mapping = {
            "日期": "trade_date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "vol",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover"
        }

        df = df.rename(columns=column_mapping)
        df["ts_code"] = original_code

        if "pct_chg" not in df.columns and "涨跌幅" in column_mapping.values():
            if "change" in df.columns and "pre_close" in df.columns:
                df["pct_chg"] = (df["change"] / df["pre_close"] * 100).round(2)

        return df

    def get_stock_list(self) -> pd.DataFrame:
        try:
            df = ak.stock_info_a_code_name()
            logger.info(f"获取股票列表成功，共 {len(df)} 只股票")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise

    def get_daily_quotes(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = "qfq"
    ) -> pd.DataFrame:
        original_code = symbol

        if len(symbol) == 6:
            if symbol.startswith('6'):
                symbol = f"sh{symbol}"
            elif symbol.startswith(('0', '3')):
                symbol = f"sz{symbol}"

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            df = self._rename_daily_columns(df, original_code)
            logger.info(f"获取 {symbol} 日线数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 日线数据失败: {e}")
            raise

    def get_minute_quotes(
        self,
        symbol: str,
        period: str = "1",
        adjust: Optional[str] = "qfq"
    ) -> pd.DataFrame:
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                adjust=adjust
            )
            logger.info(f"获取 {symbol} 分钟线数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 分钟线数据失败: {e}")
            raise

    def get_financial(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            df = ak.stock_financial_analysis_indicator(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"获取 {symbol} 财务数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 财务数据失败: {e}")
            raise

    def get_index_list(self) -> pd.DataFrame:
        try:
            df = ak.index_stock_info()
            logger.info(f"获取指数列表成功，共 {len(df)} 个指数")
            return df
        except Exception as e:
            logger.error(f"获取指数列表失败: {e}")
            raise

    def get_index_quotes(
        self,
        symbol: str,
        period: str = "daily"
    ) -> pd.DataFrame:
        try:
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period=period
            )
            logger.info(f"获取 {symbol} 指数数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 指数数据失败: {e}")
            raise

    def get_source_name(self) -> str:
        return "AKShare"


AKShareDownloader = AKShareDataSource
