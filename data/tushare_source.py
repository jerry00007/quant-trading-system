"""Tushare 数据源实现"""
import tushare as ts
import pandas as pd
from typing import Optional
import logging

from .base_source import BaseDataSource, DataSourceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TushareDataSource(BaseDataSource):
    """Tushare 数据源"""

    def __init__(self, token: str):
        ts.set_token(token)
        self.pro = ts.pro_api()
        logger.info("Tushare 数据源初始化成功")

    def get_stock_list(self) -> pd.DataFrame:
        try:
            df = self.pro.stock_basic(
                ts_code="",
                list_status="L"
            )
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
        try:
            df = self.pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                adj=adjust if adjust == "qfq" else ("hfq" if adjust == "hfq" else None)
            )
            logger.info(f"获取 {symbol} 日线数据成功，共 {len(df)} 条")
            return self.normalize_daily_data(df, symbol)
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
            df = ts.pro_bar(
                ts_code=symbol,
                freq=f"{period}min",
                adj=adjust if adjust == "qfq" else ("hfq" if adjust == "hfq" else None),
                api="pro"
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
            df = self.pro.income(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
                indicators="inc"
            )
            logger.info(f"获取 {symbol} 财务数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 财务数据失败: {e}")
            raise

    def get_index_list(self) -> pd.DataFrame:
        try:
            df = self.pro.index_basic()
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
            df = self.pro.index_daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"获取 {symbol} 指数数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 指数数据失败: {e}")
            raise

    def get_source_name(self) -> str:
        return "Tushare"
