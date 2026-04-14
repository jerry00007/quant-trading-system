"""双均线交叉策略"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Signal:
    """交易信号"""
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


class MAStrategy:
    """双均线交叉策略"""

    def __init__(
        self,
        fast_period: int = 5,
        slow_period: int = 20,
        position: Optional[str] = None
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position = position

    def calculate_signals(self, df: pd.DataFrame, position: Optional[str] = None) -> List[Signal]:
        """计算交易信号
        
        Args:
            df: 包含OHLCV数据的DataFrame
            position: 外部传入的持仓状态，避免状态泄漏
        """
        signals = []

        if df.empty or len(df) < self.slow_period:
            return signals

        df = df.copy()

        df[f"ma{self.fast_period}"] = df["close"].rolling(
            window=self.fast_period
        ).mean()

        df[f"ma{self.slow_period}"] = df["close"].rolling(
            window=self.slow_period
        ).mean()

        # 使用局部变量避免状态泄漏
        current_position = position if position is not None else "sell"

        for i in range(self.slow_period, len(df)):
            row = df.iloc[i]
            fast_ma = row[f"ma{self.fast_period}"]
            slow_ma = row[f"ma{self.slow_period}"]
            close_price = row["close"]

            if pd.isna(fast_ma) or pd.isna(slow_ma):
                continue

            if fast_ma > slow_ma and current_position != "buy":
                signals.append(Signal(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    action="buy",
                    price=close_price,
                    reason=f"MA{self.fast_period}上穿MA{self.slow_period}"
                ))
                current_position = "buy"

            elif fast_ma < slow_ma and current_position == "buy":
                signals.append(Signal(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    action="sell",
                    price=close_price,
                    reason=f"MA{self.fast_period}下穿MA{self.slow_period}"
                ))
                current_position = "sell"

        return signals

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        df[f"ma{self.fast_period}"] = df["close"].rolling(
            window=self.fast_period
        ).mean()

        df[f"ma{self.slow_period}"] = df["close"].rolling(
            window=self.slow_period
        ).mean()

        return df
