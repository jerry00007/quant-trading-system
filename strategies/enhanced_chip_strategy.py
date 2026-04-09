"""增强型主力筹码趋向策略（方案A：多因子确认+动态风控）"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class EnhancedChipSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str
    zlcmq: float
    vol_surge: bool
    low_turnover: bool
    trend_ok: bool


class EnhancedChipStrategy:
    """增强型主力筹码趋向策略

    买入条件（原ZLCMQ逻辑 + A股实证优化）:
      1. ZLCMQ 在 N_DAYS 内曾到达 MIN_HIGH 以上
      2. ZLCMQ 下穿 95
      3. 从高点回落至少 MIN_FALL，且在 2~N_DAYS-1 根K线内
      4. 价格企稳（阳线或收盘高于昨收）
      5. [新增] 成交量放大确认（20日均量1.5倍以上）
      6. [新增] 换手率过滤（日均换手率<8%，避开炒作股）
      7. [新增] 趋势过滤（收盘价在60日均线上方2%以内）

    卖出条件（动态风控）:
      1. [新增] ATR动态止损（2.5倍ATR，适应波动率）
      2. [新增] 移动止盈加速（盈利>10%后用移动止损锁定利润）
      3. 止盈 +20%（从原15%提高）
      4. 筹码极度分散：ZLCMQ < 15（连续2日确认）
    """

    def __init__(
        self,
        n_days: int = 5,
        min_high: float = 98,
        min_fall: float = 5,
        stop_loss_atr_mult: float = 2.5,
        take_profit_pct: float = 0.20,
        chip_exit: float = 15,
        vol_surge_mult: float = 1.5,
        max_turnover: float = 8.0,
        trend_ma_period: int = 60,
        trailing_profit_start: float = 0.10,
        trailing_atr_mult: float = 1.5,
    ):
        self.n_days = n_days
        self.min_high = min_high
        self.min_fall = min_fall
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_pct = take_profit_pct
        self.chip_exit = chip_exit
        self.vol_surge_mult = vol_surge_mult
        self.max_turnover = max_turnover
        self.trend_ma_period = trend_ma_period
        self.trailing_profit_start = trailing_profit_start
        self.trailing_atr_mult = trailing_atr_mult

        self.position = None
        self.entry_price = None
        self.trailing_stop = None

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

    @staticmethod
    def _atr(high, low, close, period=14):
        """平均真实波幅"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

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

    def calculate_signals(self, df: pd.DataFrame) -> List[EnhancedChipSignal]:
        if df.empty or len(df) < 80:
            return []

        close = df["close"]
        open_ = df["open"]
        volume = df["vol"]
        zlcmq = self.calculate_zlcmq(close, df["high"], df["low"])

        vol_avg_20 = volume.rolling(20, min_periods=1).mean()
        vol_surge = volume > vol_avg_20 * self.vol_surge_mult

        ma_trend = close.rolling(self.trend_ma_period, min_periods=1).mean()
        trend_ok = close > ma_trend * (1.0 - 0.02)

        turnover_1m = df["pct_chg"].abs().rolling(20, min_periods=1).mean()
        low_turnover = turnover_1m < self.max_turnover

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

        buy_cond = was_high & cross_95 & is_fast_fall & is_stable & vol_surge & low_turnover & trend_ok

        atr = self._atr(df["high"], df["low"], close)
        atr_mult_2 = atr * self.stop_loss_atr_mult
        atr_mult_trailing = atr * self.trailing_atr_mult

        self.position = None
        self.entry_price = None
        self.trailing_stop = None
        signals = []

        for i in range(75, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]
            z = zlcmq.iloc[i]
            z_prev = zlcmq.iloc[i - 1] if i > 0 else np.nan
            vol_ok = vol_surge.iloc[i]
            turn_ok = low_turnover.iloc[i]
            trend_ok_val = trend_ok.iloc[i]

            if self.position is None:
                if buy_cond.iloc[i]:
                    self.trailing_stop = p - atr_mult_trailing.iloc[i]
                    signals.append(
                        EnhancedChipSignal(
                            ts_code=row["ts_code"],
                            trade_date=row["trade_date"],
                            action="buy",
                            price=p,
                            reason=f"筹码高位回落企稳 ZLCMQ={z:.1f} | 放量{vol_ok} | 低换手{turn_ok} | 趋势好{trend_ok_val}",
                            zlcmq=z,
                            vol_surge=vol_ok,
                            low_turnover=turn_ok,
                            trend_ok=trend_ok_val,
                        )
                    )
                    self.position = "buy"
                    self.entry_price = p
            else:
                pnl = (p - self.entry_price) / self.entry_price
                sell = False
                reason = ""

                atr_stop = p <= (self.entry_price - atr_mult_2.iloc[i])
                take_profit = pnl >= self.take_profit_pct
                chip_exit_1 = not np.isnan(z) and z < self.chip_exit
                chip_exit_2 = (
                    not np.isnan(z)
                    and z < self.chip_exit
                    and z_prev < self.chip_exit
                )

                if atr_stop:
                    sell = True
                    reason = f"ATR止损 {pnl * 100:+.1f}%"
                elif pnl > self.trailing_profit_start:
                    self.trailing_stop = max(self.trailing_stop, p - atr_mult_trailing.iloc[i])
                    if p <= self.trailing_stop:
                        sell = True
                        reason = f"移动止盈 {pnl * 100:+.1f}%"
                elif take_profit:
                    sell = True
                    reason = f"止盈 {pnl * 100:+.1f}%"
                elif chip_exit_2:
                    sell = True
                    reason = f"筹码极度分散 ZLCMQ={z:.1f}"

                if sell:
                    signals.append(
                        EnhancedChipSignal(
                            ts_code=row["ts_code"],
                            trade_date=row["trade_date"],
                            action="sell",
                            price=p,
                            reason=reason,
                            zlcmq=z,
                            vol_surge=vol_ok,
                            low_turnover=turn_ok,
                            trend_ok=trend_ok_val,
                        )
                    )
                    self.position = None
                    self.entry_price = None
                    self.trailing_stop = None

        return signals
