"""数据源抽象接口"""
from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """数据源配置"""
    source_type: str  # 'akshare' or 'tushare'
    enabled: bool = True
    priority: int = 1  # 优先级，数字越小优先级越高
    token: Optional[str] = None  # Tushare token


class BaseDataSource(ABC):
    """数据源基类"""

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        pass

    @abstractmethod
    def get_daily_quotes(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: Optional[str] = "qfq"
    ) -> pd.DataFrame:
        """获取日线行情"""
        pass

    @abstractmethod
    def get_minute_quotes(
        self,
        symbol: str,
        period: str = "1",
        adjust: Optional[str] = "qfq"
    ) -> pd.DataFrame:
        """获取分钟线行情"""
        pass

    @abstractmethod
    def get_financial(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取财务数据"""
        pass

    @abstractmethod
    def get_index_list(self) -> pd.DataFrame:
        """获取指数列表"""
        pass

    @abstractmethod
    def get_index_quotes(
        self,
        symbol: str,
        period: str = "daily"
    ) -> pd.DataFrame:
        """获取指数行情"""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """获取数据源名称"""
        pass

    def normalize_daily_data(self, df: pd.DataFrame, original_symbol: str) -> pd.DataFrame:
        """标准化日线数据为统一格式"""
        required_columns = [
            "ts_code", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "vol", "amount"
        ]

        df = df.copy()

        if "ts_code" not in df.columns:
            df["ts_code"] = original_symbol

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        return df[required_columns]
