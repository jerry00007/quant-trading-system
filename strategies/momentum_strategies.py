"""动量类策略：布林带突破、RSI超买超卖、MACD金叉死叉
参数已通过网格搜索优化。
"""

import numpy as np
import pandas as pd
from typing import List

from strategies.benchmark_strategies import BaseSignal, _next_open_price


class BollingerBreakStrategy:
    """布林带突破策略
    价格突破下轨买入（超卖反弹），突破上轨卖出（超买回落）
    优化参数：period=25, std_dev=2.5
    """

    def __init__(
        self,
        period: int = 25,
        std_dev: float = 2.5,
        stop_pct: float = 0.05,
    ):
        self.period = period
        self.std_dev = std_dev
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.period + 2:
            return []

        close = df["close"].copy()
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(self.period, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]
            prev_p = close.iloc[i - 1]

            if self.position is None:
                if prev_p > lower.iloc[i - 1] and p <= lower.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason="突破布林带下轨 超卖反弹",
                            )
                        )
                        self.position = "buy"
                        self.entry_price = nx_price
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""
                exec_price = p
                exec_date = row["trade_date"]

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl * 100:+.1f}%"
                    exec_price = p * 0.999
                elif prev_p < upper.iloc[i - 1] and p >= upper.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        sell = True
                        reason = f"突破布林带上轨 {pnl * 100:+.1f}%"
                        exec_price = nx_price
                        exec_date = nx_date

                if sell:
                    signals.append(
                        BaseSignal(
                            ts_code=row["ts_code"],
                            trade_date=exec_date,
                            action="sell",
                            price=exec_price,
                            reason=reason,
                        )
                    )
                    self.position = None
                    self.entry_price = None

        return signals


class RSIMomentumStrategy:
    """RSI超买超卖策略
    RSI<25买入（超卖），RSI>80卖出（超买）
    优化参数：period=12, oversold=25, overbought=80
    """

    def __init__(
        self,
        period: int = 12,
        oversold: float = 25,
        overbought: float = 80,
        stop_pct: float = 0.05,
    ):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def _rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.period + 2:
            return []

        close = df["close"]
        rsi = self._rsi(close, self.period)

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(self.period + 1, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                if rsi.iloc[i] < self.oversold:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason=f"RSI={rsi.iloc[i]:.1f} 超卖区域",
                            )
                        )
                        self.position = "buy"
                        self.entry_price = nx_price
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""
                exec_price = p
                exec_date = row["trade_date"]

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl * 100:+.1f}%"
                    exec_price = p * 0.999
                elif rsi.iloc[i] > self.overbought:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        sell = True
                        reason = f"RSI={rsi.iloc[i]:.1f} 超买区域"
                        exec_price = nx_price
                        exec_date = nx_date

                if sell:
                    signals.append(
                        BaseSignal(
                            ts_code=row["ts_code"],
                            trade_date=exec_date,
                            action="sell",
                            price=exec_price,
                            reason=reason,
                        )
                    )
                    self.position = None
                    self.entry_price = None

        return signals


class MACDCrossStrategy:
    """MACD金叉死叉策略
    MACD金叉买入，死叉卖出
    优化参数：fast=15, slow=26, signal=13
    """

    def __init__(
        self,
        fast_period: int = 15,
        slow_period: int = 26,
        signal_period: int = 13,
        stop_pct: float = 0.05,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.slow_period + 2:
            return []

        close = df["close"]
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]

            if pd.isna(dif.iloc[i]) or pd.isna(dea.iloc[i]):
                continue

            p = close.iloc[i]

            if self.position is None:
                if dif.iloc[i - 1] <= dea.iloc[i - 1] and dif.iloc[i] > dea.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason="MACD金叉",
                            )
                        )
                        self.position = "buy"
                        self.entry_price = nx_price
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""
                exec_price = p
                exec_date = row["trade_date"]

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl * 100:+.1f}%"
                    exec_price = p * 0.999
                elif dif.iloc[i - 1] >= dea.iloc[i - 1] and dif.iloc[i] < dea.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        sell = True
                        reason = "MACD死叉"
                        exec_price = nx_price
                        exec_date = nx_date

                if sell:
                    signals.append(
                        BaseSignal(
                            ts_code=row["ts_code"],
                            trade_date=exec_date,
                            action="sell",
                            price=exec_price,
                            reason=reason,
                        )
                    )
                    self.position = None
                    self.entry_price = None

        return signals
