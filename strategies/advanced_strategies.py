"""高级策略集合：MACD背离、改良海龟、动量反转"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class BaseSignal:
    ts_code: str
    trade_date: str
    action: str  # "buy" or "sell"
    price: float
    reason: str


class MACDDivergenceStrategy:
    """MACD背离策略

    基于MACD指标与价格的背离信号进行交易：
    - 买入：价格创新低但MACD柱状图未创新低（看涨背离）
    - 卖出：止损-5%、MACD死叉（MACD线下穿信号线）、止盈+15%
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        lookback: int = 20,
        stop_pct: float = 0.05,
        take_profit_pct: float = 0.15,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.lookback = lookback
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct
        self.position = None
        self.entry_price = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        min_len = self.slow_period + self.signal_period + self.lookback
        if df.empty or len(df) < min_len:
            return []

        close = df["close"]
        # 向量化计算MACD
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        self.position = None
        self.entry_price = None
        signals = []

        start_idx = self.slow_period + self.signal_period
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                # 看涨背离检测：在lookback窗口内，价格创新低但MACD柱状图未创新低
                if i >= self.lookback:
                    window_start = i - self.lookback
                    price_recent_low = close.iloc[window_start:i].min()
                    hist_recent_vals = histogram.iloc[window_start:i]

                    # 当前价格低于窗口内最低价（价格创新低）
                    if p < price_recent_low:
                        # MACD柱状图：当前值 > 窗口内最小值（柱状图未创新低）
                        hist_current = histogram.iloc[i]
                        hist_window_min = hist_recent_vals.min()
                        if hist_current > hist_window_min:
                            signals.append(BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=row["trade_date"],
                                action="buy",
                                price=p,
                                reason=f"MACD看涨背离 价格新低{p:.2f}"
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
                elif pnl >= self.take_profit_pct:
                    sell = True
                    reason = f"止盈 {pnl*100:+.1f}%"
                elif (
                    macd_line.iloc[i - 1] >= signal_line.iloc[i - 1]
                    and macd_line.iloc[i] < signal_line.iloc[i]
                ):
                    sell = True
                    reason = f"MACD死叉 {pnl*100:+.1f}%"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"],
                        trade_date=row["trade_date"],
                        action="sell",
                        price=p,
                        reason=reason,
                    ))
                    self.position = None
                    self.entry_price = None

        return signals


class ModifiedTurtleStrategy:
    """改良海龟策略

    基于通道突破的趋势跟踪策略：
    - 入场：20日通道突破（收盘价突破过去20日最高价）
    - 出场：10日通道突破（收盘价跌破过去10日最低价）
    - 止损：2倍ATR(20)从入场价
    - 加仓：价格从入场价上涨0.5倍ATR时加仓一次（仅一次）
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        stop_atr_mult: float = 2.0,
        add_atr_mult: float = 0.5,
    ):
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        self.add_atr_mult = add_atr_mult
        self.position = None
        self.entry_price = None
        self.added = False
        self.atr_at_entry = None

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        min_len = max(self.entry_period, self.atr_period) + 1
        if df.empty or len(df) < min_len:
            return []

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 向量化计算指标
        entry_channel = high.shift(1).rolling(self.entry_period).max()
        exit_channel = low.shift(1).rolling(self.exit_period).min()
        atr_series = self._atr(high, low, close, self.atr_period)

        self.position = None
        self.entry_price = None
        self.added = False
        self.atr_at_entry = None
        signals = []

        for i in range(min_len, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]
            atr_val = atr_series.iloc[i]

            if np.isnan(atr_val) or np.isnan(entry_channel.iloc[i]):
                continue

            if self.position is None:
                # 20日通道突破入场
                if p > entry_channel.iloc[i]:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"],
                        trade_date=row["trade_date"],
                        action="buy",
                        price=p,
                        reason=f"20日突破 入场{p:.2f} 通道{entry_channel.iloc[i]:.2f}"
                    ))
                    self.position = "buy"
                    self.entry_price = p
                    self.added = False
                    self.atr_at_entry = atr_val
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""
                stop_price = self.entry_price - self.stop_atr_mult * self.atr_at_entry

                # 止损：2倍ATR
                if p <= stop_price:
                    sell = True
                    reason = f"ATR止损 {pnl*100:+.1f}%"
                # 10日通道出场
                elif p < exit_channel.iloc[i]:
                    sell = True
                    reason = f"10日通道出场 {pnl*100:+.1f}%"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"],
                        trade_date=row["trade_date"],
                        action="sell",
                        price=p,
                        reason=reason,
                    ))
                    self.position = None
                    self.entry_price = None
                    self.added = False
                    self.atr_at_entry = None
                elif not self.added:
                    # 加仓：价格从入场价上涨0.5倍ATR
                    add_threshold = self.entry_price + self.add_atr_mult * self.atr_at_entry
                    if p >= add_threshold:
                        signals.append(BaseSignal(
                            ts_code=row["ts_code"],
                            trade_date=row["trade_date"],
                            action="buy",
                            price=p,
                            reason=f"加仓 +{self.add_atr_mult}ATR 价格{p:.2f}"
                        ))
                        self.added = True

        return signals


class MomentumReversalStrategy:
    """动量反转策略

    基于动量排名与反转形态的短线策略：
    - 买入：股票处于近期收益底部四分位（超卖）+ 反转K线（收阳且收盘>前收）+ 成交量>1.5倍20日均量
    - 卖出：止损-5%、10日新高出场（动量恢复止盈）、最大持仓15根K线
    """

    def __init__(
        self,
        lookback: int = 20,
        volume_mult: float = 1.5,
        stop_pct: float = 0.05,
        max_hold: int = 15,
    ):
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.stop_pct = stop_pct
        self.max_hold = max_hold
        self.position = None
        self.entry_price = None
        self.hold_bars = 0

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        min_len = self.lookback + 1
        if df.empty or len(df) < min_len:
            return []

        close = df["close"]
        open_col = df["open"]
        vol = df["vol"]
        high = df["high"]

        # 向量化计算指标
        returns = close.pct_change(self.lookback)
        avg_volume = vol.rolling(self.lookback).mean()
        high_10 = high.shift(1).rolling(10).max()

        self.position = None
        self.entry_price = None
        self.hold_bars = 0
        signals = []

        for i in range(min_len, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                # 动量排名：检查是否处于底部四分位
                ret_window = returns.iloc[i - self.lookback + 1 : i + 1]
                if len(ret_window.dropna()) < self.lookback:
                    continue
                percentile_rank = (ret_window.dropna() < returns.iloc[i]).sum() / len(
                    ret_window.dropna()
                )

                # 底部四分位 + 反转K线 + 放量
                is_oversold = percentile_rank <= 0.25
                is_reversal_candle = (p > open_col.iloc[i]) and (
                    p > close.iloc[i - 1]
                )
                is_volume_surge = (
                    vol.iloc[i] > self.volume_mult * avg_volume.iloc[i]
                )

                if is_oversold and is_reversal_candle and is_volume_surge:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"],
                        trade_date=row["trade_date"],
                        action="buy",
                        price=p,
                        reason=f"动量反转 排名{percentile_rank:.0%} 量比{vol.iloc[i]/avg_volume.iloc[i]:.1f}"
                    ))
                    self.position = "buy"
                    self.entry_price = p
                    self.hold_bars = 0
            else:
                self.hold_bars += 1
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                if pnl <= -self.stop_pct:
                    sell = True
                    reason = f"止损 {pnl*100:+.1f}%"
                elif i >= 10 and not np.isnan(high_10.iloc[i]) and p > high_10.iloc[i]:
                    sell = True
                    reason = f"10日新高出场 {pnl*100:+.1f}%"
                elif self.hold_bars >= self.max_hold:
                    sell = True
                    reason = f"最大持仓{self.max_hold}根 {pnl*100:+.1f}%"

                if sell:
                    signals.append(BaseSignal(
                        ts_code=row["ts_code"],
                        trade_date=row["trade_date"],
                        action="sell",
                        price=p,
                        reason=reason,
                    ))
                    self.position = None
                    self.entry_price = None
                    self.hold_bars = 0

        return signals
