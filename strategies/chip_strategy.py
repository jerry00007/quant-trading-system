"""主力筹码趋向策略（通达信指标转换）"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class ChipSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


class ChipStrategy:
    """主力筹码趋向(ZLCMQ)策略

    买入条件（原通达信逻辑）:
      1. ZLCMQ 在 N_DAYS 内曾到达 MIN_HIGH 以上
      2. ZLCMQ 下穿 95
      3. 从高点回落至少 MIN_FALL，且在 2~N_DAYS-1 根K线内
      4. 价格企稳（阳线或收盘高于昨收）

    卖出条件:
      1. 止损: -8%
      2. 止盈: +15%
      3. 筹码极度分散: ZLCMQ < 15
    """

    def __init__(
        self,
        n_days: int = 5,
        min_high: float = 98,
        min_fall: float = 5,
        stop_loss_pct: float = -0.08,
        take_profit_pct: float = 0.15,
        chip_exit: float = 15,
    ):
        self.n_days = n_days
        self.min_high = min_high
        self.min_fall = min_fall
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.chip_exit = chip_exit

        self.position = None
        self.entry_price = None

    @staticmethod
    def _tdx_sma(x, n, m):
        """通达信 SMA(X, N, M): Y = (M*X + (N-M)*Y_prev) / N"""
        y = np.empty(len(x))
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = (m * x[i] + (n - m) * y[i - 1]) / n
        return y

    @staticmethod
    def _barslast_high(zlcmq, n_days):
        """BARSLAST(ZLCMQ = HHV(ZLCMQ, N_DAYS))"""
        result = np.full(len(zlcmq), np.nan)
        for i in range(len(zlcmq)):
            start = max(0, i - n_days + 1)
            window = zlcmq[start : i + 1]
            if len(window) == 0 or np.any(np.isnan(window)):
                continue
            max_val = np.max(window)
            for j in range(len(window) - 1, -1, -1):
                if not np.isnan(window[j]) and window[j] == max_val:
                    result[i] = len(window) - 1 - j
                    break
        return result

    def calculate_zlcmq(self, close, high, low):
        """计算主力筹码趋向指标"""
        c = close.values.astype(float)
        h = high.values.astype(float)
        lo = low.values.astype(float)

        var5 = pd.Series(lo).rolling(75, min_periods=1).min().values
        var6 = pd.Series(h).rolling(75, min_periods=1).max().values
        var7 = (var6 - var5) / 100.0

        raw = np.where(var7 > 1e-10, (c - var5) / var7, 0.0)
        raw = np.nan_to_num(raw, nan=0.0)

        var8 = self._tdx_sma(raw, 20, 1)
        var8_s = self._tdx_sma(var8, 15, 1)
        vara = 3.0 * var8 - 2.0 * var8_s

        return pd.Series(100.0 - vara, index=close.index)

    def calculate_signals(self, df: pd.DataFrame) -> List[ChipSignal]:
        if df.empty or len(df) < 80:
            return []

        close = df["close"]
        open_ = df["open"]
        zlcmq = self.calculate_zlcmq(close, df["high"], df["low"])

        zq_high = zlcmq.rolling(self.n_days, min_periods=1).max()
        was_high = zq_high >= self.min_high
        cross_95 = (zlcmq.shift(1) >= 95) & (zlcmq < 95)

        high_bars = self._barslast_high(zlcmq.values, self.n_days)
        fall_value = zq_high - zlcmq
        is_fast_fall = (
            (high_bars >= 2)
            & (high_bars <= self.n_days - 1)
            & (fall_value >= self.min_fall)
        )

        is_stable = (close > open_) | (close > close.shift(1))
        buy_cond = was_high & cross_95 & is_fast_fall & is_stable

        self.position = None
        self.entry_price = None
        signals = []

        for i in range(75, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]
            z = zlcmq.iloc[i]

            if self.position is None:
                if buy_cond.iloc[i]:
                    signals.append(
                        ChipSignal(
                            ts_code=row["ts_code"],
                            trade_date=row["trade_date"],
                            action="buy",
                            price=p,
                            reason=f"筹码高位回落企稳 ZLCMQ={z:.1f}",
                        )
                    )
                    self.position = "buy"
                    self.entry_price = p
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                if pnl <= self.stop_loss_pct:
                    sell = True
                    reason = f"止损 {pnl * 100:+.1f}%"
                elif pnl >= self.take_profit_pct:
                    sell = True
                    reason = f"止盈 {pnl * 100:+.1f}%"
                elif not np.isnan(z) and z < self.chip_exit:
                    sell = True
                    reason = f"筹码极度分散 ZLCMQ={z:.1f}"

                if sell:
                    signals.append(
                        ChipSignal(
                            ts_code=row["ts_code"],
                            trade_date=row["trade_date"],
                            action="sell",
                            price=p,
                            reason=reason,
                        )
                    )
                    self.position = None
                    self.entry_price = None

        return signals
