"""基准策略集合：双均线、布林带、RSI"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class BaseSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


class DualMAStrategy:
    def __init__(self, fast_period: int = 5, slow_period: int = 20, stop_pct: float = 0.05):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.slow_period + 1:
            return []

        close = df["close"]
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(self.slow_period, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                if fast_ma.iloc[i - 1] <= slow_ma.iloc[i - 1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="buy", price=p,
                        reason=f"金叉 MA{self.fast_period}>MA{self.slow_period}"
                    ))
                    self.position = "buy"
                    self.entry_price = p
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl*100:+.1f}%"
                elif fast_ma.iloc[i - 1] >= slow_ma.iloc[i - 1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                    sell = True
                    reason = f"死叉 MA{self.fast_period}<MA{self.slow_period}"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="sell", price=p, reason=reason
                    ))
                    self.position = None
                    self.entry_price = None

        return signals


class BollingerBandStrategy:
    def __init__(self, period: int = 20, std_mult: float = 2.0, stop_pct: float = 0.04):
        self.period = period
        self.std_mult = std_mult
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.period + 1:
            return []

        close = df["close"]
        ma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = ma + std * self.std_mult
        lower = ma - std * self.std_mult

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(self.period, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                if p <= lower.iloc[i]:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="buy", price=p,
                        reason=f"触及下轨 布林带({self.period},{self.std_mult})"
                    ))
                    self.position = "buy"
                    self.entry_price = p
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl*100:+.1f}%"
                elif p >= ma.iloc[i]:
                    sell = True
                    reason = f"回归中轨 {pnl*100:+.1f}%"
                elif p >= upper.iloc[i]:
                    sell = True
                    reason = f"触及上轨 {pnl*100:+.1f}%"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="sell", price=p, reason=reason
                    ))
                    self.position = None
                    self.entry_price = None

        return signals


class RSIStrategy:
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, stop_pct: float = 0.05):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.stop_pct = stop_pct
        self.position = None
        self.entry_price = None

    def _rsi(self, close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 0.001)
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
                if rsi.iloc[i - 1] < self.oversold and rsi.iloc[i] >= self.oversold:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="buy", price=p,
                        reason=f"RSI超卖回升 RSI={rsi.iloc[i]:.0f}"
                    ))
                    self.position = "buy"
                    self.entry_price = p
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl*100:+.1f}%"
                elif rsi.iloc[i - 1] > self.overbought and rsi.iloc[i] <= self.overbought:
                    sell = True
                    reason = f"RSI超买回落 RSI={rsi.iloc[i]:.0f}"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"], trade_date=row["trade_date"],
                        action="sell", price=p, reason=reason
                    ))
                    self.position = None
                    self.entry_price = None

        return signals
