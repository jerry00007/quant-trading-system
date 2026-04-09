"""网易163数据源实现"""
import pandas as pd
import logging
from typing import Optional

from .base_source import BaseDataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Netease163DataSource(BaseDataSource):
    """网易163股票历史数据源"""

    BASE_URL = "http://quotes.money.163.com/service/chddata.html"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        logger.warning("网易163暂不支持获取完整股票列表，请使用 Tushare 或 AKShare")
        return pd.DataFrame(columns=["ts_code", "symbol", "name"])

    def get_daily_quotes(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = "qfq"
    ) -> pd.DataFrame:
        """获取日线行情"""
        original_code = symbol

        normalized_code = self._normalize_code(symbol)

        params = {
            "code": normalized_code,
            "start": start_date,
            "end": end_date
        }

        try:
            url = f"{self.BASE_URL}?code={params['code']}&start={params['start']}&end={params['end']}"
            logger.info(f"请求网易163接口: {url}")

            df = pd.read_csv(url, encoding="gb2312")

            if df.empty:
                logger.warning(f"股票 {symbol} 无数据")
                return df

            df = self._normalize_daily_data(df, original_code)
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
        """获取分钟线行情"""
        logger.warning("网易163暂不支持分钟线数据，建议使用 Tushare 或 AKShare")
        return pd.DataFrame()

    def get_financial(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取财务数据"""
        logger.warning("网易163暂不支持财务数据，建议使用 Tushare 或 AKShare")
        return pd.DataFrame()

    def get_index_list(self) -> pd.DataFrame:
        """获取指数列表"""
        logger.warning("网易163暂不支持指数列表，建议使用 Tushare 或 AKShare")
        return pd.DataFrame(columns=["ts_code", "name"])

    def get_index_quotes(
        self,
        symbol: str,
        period: str = "daily"
    ) -> pd.DataFrame:
        """获取指数行情"""
        logger.warning("网易163暂不支持指数行情，建议使用 Tushare 或 AKShare")
        return pd.DataFrame()

    @staticmethod
    def _normalize_code(symbol: str) -> str:
        """标准化股票代码：添加市场前缀"""
        if isinstance(symbol, int):
            symbol = str(symbol).zfill(6, "0")

        if symbol.startswith('6') or symbol.startswith('900'):
            return symbol

        if symbol.startswith('0') or symbol.startswith('3'):
            return f"1{symbol}"

        raise ValueError(f"无法识别的股票代码格式: {symbol}")

    def _normalize_daily_data(self, df: pd.DataFrame, original_code: str) -> pd.DataFrame:
        """标准化日线数据为统一格式"""
        df = df.copy()

        column_mapping = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "vol",
            "成交额": "amount"
        }

        df = df.rename(columns=column_mapping)

        df["ts_code"] = original_code

        if "股票代码" in df.columns:
            df = df.drop(columns=["股票代码"])

        if "涨跌幅" not in df.columns and "涨跌额" in df.columns:
            if "涨跌额" in df.columns and "收盘价" in df.columns and "前收盘" in df.columns:
                df["涨跌幅"] = (df["涨跌额"] / df["前收盘"] * 100).round(2)

        numeric_cols = ["open", "high", "low", "close", "vol", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        return df

    def get_source_name(self) -> str:
        return "网易163"
