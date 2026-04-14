"""顶底图策略（通达信顶底图指标转换）
基于好大哥提供的顶底图指标公式实现。
买入：VAR20红色柱 + 切手线低位向上
卖出：髑战线高位回落 + 三胖线拐头向下
"""

import numpy as np
import pandas as pd
from typing import List

from strategies.benchmark_strategies import BaseSignal, _next_open_price


class TopBottomStrategy:
    """顶底图策略 — 通达信顶底图指标

    核心指标线：髑战(战斗线)、三胖(胖手指线)、切手(切手线)
    买入：VAR20红色柱出现 + 切手线<30（低位）+ 三胖线>50
    卖出：止损-8% / 止盈+15% / 髑战高位拐头 / 三胖下穿VAR4
    """

    def __init__(
        self,
        var1: float = 1.0,
        winner_lookback: int = 250,
        stop_loss_pct: float = -0.08,
        take_profit_pct: float = 0.15,
    ):
        self.var1 = var1
        self.winner_lookback = winner_lookback
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position = None
        self.entry_price = None

    @staticmethod
    def _tdx_sma(x, n, m):
        y = np.empty(len(x))
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = (m * x[i] + (n - m) * y[i - 1]) / n
        return y

    def _winner(self, price: pd.Series, lookback: int = None) -> pd.Series:
        if lookback is None:
            lookback = self.winner_lookback
        if len(price) < lookback:
            lookback = len(price)
        winner_values = np.ones(len(price))
        for i in range(len(price)):
            start = max(0, i - lookback + 1)
            window = price.iloc[start : i + 1]
            if len(window) < 2:
                winner_values[i] = 0.5
            else:
                rank = (window <= price.iloc[i]).sum()
                winner_values[i] = rank / len(window)
        return pd.Series(winner_values, index=price.index)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = (
            df["vol"] if "vol" in df.columns else pd.Series(1000000, index=df.index)
        )
        capital = (
            df["vol"] * df["close"] * 10
            if "vol" in df.columns
            else pd.Series(1e9, index=df.index)
        )

        var1 = self.var1

        winner_close = self._winner(close)
        var2 = 1.0 / np.where(winner_close > 1e-10, winner_close, 1e-10)

        var3 = close.rolling(13, min_periods=1).mean()
        var4 = 100 - np.abs((close - var3) / np.where(var3 > 1e-10, var3, 1e-10) * 100)

        var5 = low.rolling(75, min_periods=1).min()
        var6 = high.rolling(75, min_periods=1).max()
        var7 = (var6 - var5) / 100.0

        raw_var8 = np.where(var7 > 1e-10, (close - var5) / var7, 0.0)
        var8 = self._tdx_sma(raw_var8, 20, 1)
        raw_var9 = np.where(var7 > 1e-10, (open_ - var5) / var7, 0.0)
        var9 = self._tdx_sma(raw_var9, 20, 1)

        vara = 3.0 * var8 - 2.0 * self._tdx_sma(var8, 15, 1)
        varb = 3.0 * var9 - 2.0 * self._tdx_sma(var9, 15, 1)

        fight = (100 - vara) * var1

        winner_close_95 = self._winner(close * 0.95)
        sanpang = (winner_close_95 * 100).rolling(3, min_periods=1).mean() * var1

        cond1 = var2 > 5
        cond2 = var2 < 100
        val_if = np.where(cond1, np.where(cond2, var2, var4 - 10), 0)
        qieshou = (100 - val_if) * var1

        vare = low.shift(1) * 0.9
        varf = low * 0.9
        var10 = (varf * volume + vare * (capital - volume)) / np.where(
            capital > 0, capital, 1
        )
        var11 = var10.ewm(span=30, adjust=False).mean()

        var12 = close - close.shift(1)
        var13 = np.maximum(var12, 0)
        var14 = np.abs(var12)
        var15 = np.where(
            self._tdx_sma(var14.values, 7, 1) > 1e-10,
            self._tdx_sma(var13.values, 7, 1) / self._tdx_sma(var14.values, 7, 1) * 100,
            50,
        )
        var16 = np.where(
            self._tdx_sma(var14.values, 13, 1) > 1e-10,
            self._tdx_sma(var13.values, 13, 1)
            / self._tdx_sma(var14.values, 13, 1)
            * 100,
            50,
        )
        var17 = pd.Series(range(1, len(close) + 1), index=close.index)

        var18_numer = self._tdx_sma(np.maximum(var12, 0).values, 6, 1)
        var18_denom = self._tdx_sma(np.abs(var12).values, 6, 1)
        var18 = np.where(var18_denom > 1e-10, var18_numer / var18_denom * 100, 50)

        hhv_60 = high.rolling(60, min_periods=1).max()
        llv_60 = low.rolling(60, min_periods=1).min()
        var19 = (-200) * (hhv_60 - close) / np.where(
            (hhv_60 - llv_60) > 1e-10, hhv_60 - llv_60, 1e-10
        ) + 100

        var1a = (
            (close - low.rolling(15, min_periods=1).min())
            / np.where(
                (
                    high.rolling(15, min_periods=1).max()
                    - low.rolling(15, min_periods=1).min()
                )
                > 1e-10,
                high.rolling(15, min_periods=1).max()
                - low.rolling(15, min_periods=1).min(),
                1e-10,
            )
            * 100
        )

        var1b = self._tdx_sma((self._tdx_sma(var1a.values, 4, 1) - 50) * 2, 3, 1)
        var1e = self._tdx_sma(self._tdx_sma(var1a.values, 4, 1), 3, 1)
        var1f = (
            (high.rolling(30, min_periods=1).max() - close)
            / np.where(close > 1e-10, close, 1e-10)
            * 100
        )

        var20 = (
            (var18 <= 25)
            & (var19 < -95)
            & (var1f > 20)
            & (var1b < -30)
            & (var1e < 30)
            & (var11 - close >= -0.25)
            & (var15 < 22)
            & (var16 < 28)
            & (var17 > 50)
        )

        df["fight"] = fight
        df["sanpang"] = sanpang
        df["qieshou"] = qieshou
        df["var4"] = var4
        df["var20"] = var20.astype(bool)
        return df

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < 100:
            return []

        df = df.sort_values("trade_date").copy()
        df = self._calculate_indicators(df)
        close = df["close"]
        fight = df["fight"]
        sanpang = df["sanpang"]
        qieshou = df["qieshou"]
        var20 = df["var20"]

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(100, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]
            qs = qieshou.iloc[i]
            ft = fight.iloc[i]
            sp = sanpang.iloc[i]
            is_red_bar = var20.iloc[i]

            if self.position is None:
                buy_cond = is_red_bar and qs < 30 and sp > 50
                if buy_cond:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason=f"顶底图红色柱+切手低位 qieshou={qs:.1f} fight={ft:.1f}",
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

                if pnl <= self.stop_loss_pct:
                    sell = True
                    reason = f"止损 {pnl * 100:+.1f}%"
                    exec_price = p * 0.999
                elif pnl >= self.take_profit_pct:
                    sell = True
                    reason = f"止盈 {pnl * 100:+.1f}%"
                elif (
                    ft > 70
                    and fight.iloc[i] < fight.iloc[i - 1]
                    and fight.iloc[i - 1] > fight.iloc[i - 2]
                ):
                    sell = True
                    reason = f"髑战线高位拐头 fight={ft:.1f}"
                elif (
                    sanpang.iloc[i] < df["var4"].iloc[i]
                    and sanpang.iloc[i - 1] >= df["var4"].iloc[i - 1]
                ):
                    sell = True
                    reason = f"三胖线下穿VAR4 sanpang={sp:.1f}"

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
