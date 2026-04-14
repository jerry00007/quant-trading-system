"""锋芒实战策略集：爆量突破、龙头首阴反抽、均线趋势跟踪
来源：锋芒爆点盈利系统 / 波段实战课程
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from strategies.benchmark_strategies import BaseSignal, _next_open_price


class VolumeBreakoutStrategy:
    """爆量突破 — 锋芒爆点盈利系统

    核心逻辑：低位横盘后爆量突破20日高点，换手率放大

    买入条件:
      1. 20日波动率 < 15%（横盘整理）
      2. 今日成交量 > 2倍20日均量（爆量）
      3. 收盘价突破20日最高价（突破）
      4. 价格在20日均线上方（趋势确认）

    卖出条件:
      1. 止损: -8%
      2. 止盈: +20%
      3. 移动止盈: 盈利>10%后回撤5%
    """

    def __init__(
        self,
        box_days: int = 20,
        vol_mult: float = 2.0,
        stop_loss_pct: float = -0.08,
        take_profit_pct: float = 0.20,
        trail_start: float = 0.10,
        trail_pct: float = 0.05,
    ):
        self.box_days = box_days
        self.vol_mult = vol_mult
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trail_start = trail_start
        self.trail_pct = trail_pct
        self.position = None
        self.entry_price = None
        self.trailing_stop = None

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < self.box_days + 10:
            return []

        df = df.sort_values("trade_date").copy()
        bd = self.box_days
        close = df["close"]
        high = df["high"]
        vol = df["vol"] if "vol" in df.columns else pd.Series(0, index=df.index)

        ma20 = close.rolling(bd).mean()
        avg_vol = vol.rolling(bd).mean()
        volatility = close.pct_change().rolling(bd).std() * np.sqrt(bd)
        high20 = high.rolling(bd).max().shift(1)

        is_box = volatility < 0.15
        is_vol = vol > avg_vol * self.vol_mult
        is_break = close > high20
        above_ma = close > ma20

        buy_cond = is_box & is_vol & is_break & above_ma

        self.position = None
        self.entry_price = None
        self.trailing_stop = None
        signals = []

        for i in range(bd, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                if buy_cond.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        vol_ratio = vol.iloc[i] / max(avg_vol.iloc[i], 1)
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason=f"爆量突破 量比{vol_ratio:.1f} 突破{high20.iloc[i]:.2f}",
                            )
                        )
                        self.position = "buy"
                        self.entry_price = nx_price
                        self.trailing_stop = None
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
                elif pnl > self.trail_start:
                    trail_price = p * (1 - self.trail_pct)
                    if self.trailing_stop is None or trail_price > self.trailing_stop:
                        self.trailing_stop = trail_price
                    if p <= self.trailing_stop:
                        sell = True
                        reason = f"移动止盈 {pnl * 100:+.1f}%"

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
                    self.trailing_stop = None

        return signals


class DragonFirstYinStrategy:
    """龙头首阴反抽 — 锋芒爆点盈利系统

    核心逻辑：连续涨停后首次收大阴线，次日低吸博弈反抽

    买入条件:
      1. 连续涨停 >= 2天
      2. 首日收阴线（跌幅>3%，收盘<开盘）
      3. 次日低吸买入

    卖出条件:
      1. 止损: -5%
      2. 止盈: +10%
      3. 持仓超过3天保利出局
    """

    def __init__(
        self,
        limit_pct: float = 0.095,
        min_limits: int = 2,
        yin_pct: float = -0.03,
        stop_loss_pct: float = -0.05,
        take_profit_pct: float = 0.10,
        max_hold_days: int = 3,
    ):
        self.limit_pct = limit_pct
        self.min_limits = min_limits
        self.yin_pct = yin_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_days = max_hold_days

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < 30:
            return []

        df = df.sort_values("trade_date").copy()
        close = df["close"]
        open_ = df["open"]
        pct = (
            df["pct_chg"] if "pct_chg" in df.columns else df["close"].pct_change() * 100
        )

        lp = self.limit_pct * 100
        is_limit = pct > lp

        # 连续涨停天数
        limit_streak = pd.Series(0, index=df.index, dtype=int)
        streak = 0
        for i in range(len(df)):
            if is_limit.iloc[i]:
                streak += 1
                limit_streak.iloc[i] = streak
            else:
                streak = 0

        # 首阴: 昨日连续涨停>=N，今日收阴且跌幅>3%
        yp = self.yin_pct * 100
        first_yin = (
            (limit_streak.shift(1) >= self.min_limits) & (pct < yp) & (close < open_)
        )
        # 次日买入信号
        buy_signal = first_yin.shift(1).fillna(False)

        in_pos = False
        entry_price = 0
        hold_days = 0
        signals = []

        for i in range(30, len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if not in_pos and buy_signal.iloc[i]:
                nx_price, nx_date = _next_open_price(df, i)
                if nx_price is not None:
                    signals.append(
                        BaseSignal(
                            ts_code=row["ts_code"],
                            trade_date=nx_date,
                            action="buy",
                            price=nx_price,
                            reason="首阴次日低吸",
                        )
                    )
                    in_pos = True
                    entry_price = nx_price
                    hold_days = 0
            elif in_pos:
                hold_days += 1
                pnl = (p - entry_price) / entry_price
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
                elif hold_days >= self.max_hold_days:
                    sell = True
                    reason = f"持仓{self.max_hold_days}日保利 {pnl * 100:+.1f}%"

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
                    in_pos = False

        return signals


class TrendMAStrategy:
    """均线趋势跟踪 — 锋芒波段实战

    三阶段均线系统：酝势→起势→趋势

    买入条件:
      1. MA5 > MA10 > MA20 > MA30 多头排列首次形成（起势信号）

    卖出条件:
      1. 固定止损: -5%
      2. 浮动保利: 盈利>5%后回撤3%
      3. 趋势保利: 多头排列破位（收盘破5均）
    """

    def __init__(
        self,
        ma_periods: Optional[list] = None,
        stop_loss_pct: float = -0.05,
        trail_start: float = 0.05,
        trail_pct: float = 0.03,
    ):
        self.ma_periods = ma_periods or [5, 10, 20, 30]
        self.stop_loss_pct = stop_loss_pct
        self.trail_start = trail_start
        self.trail_pct = trail_pct
        self.position = None
        self.entry_price = None
        self.trailing_stop = None
        self.prev_bull = False

    def calculate_signals(self, df: pd.DataFrame) -> List[BaseSignal]:
        if df.empty or len(df) < 40:
            return []

        df = df.sort_values("trade_date").copy()
        close = df["close"]
        ms = self.ma_periods

        for m in ms:
            df[f"ma{m}"] = close.rolling(m).mean()

        # 多头排列: MA5 > MA10 > MA20 > MA30
        bull = pd.Series(True, index=df.index)
        for j in range(len(ms) - 1):
            bull = bull & (df[f"ma{ms[j]}"] > df[f"ma{ms[j + 1]}"])

        # 首次多头: 之前非多头，今天多头
        first_bull = bull & ~bull.shift(1).fillna(False)

        self.position = None
        self.entry_price = None
        self.trailing_stop = None
        self.prev_bull = False
        signals = []

        for i in range(max(ms), len(df)):
            row = df.iloc[i]
            p = close.iloc[i]

            if self.position is None:
                if first_bull.iloc[i]:
                    nx_price, nx_date = _next_open_price(df, i)
                    if nx_price is not None:
                        signals.append(
                            BaseSignal(
                                ts_code=row["ts_code"],
                                trade_date=nx_date,
                                action="buy",
                                price=nx_price,
                                reason="均线起势(多头排列)",
                            )
                        )
                        self.position = "buy"
                        self.entry_price = nx_price
                        self.trailing_stop = None
                        self.prev_bull = bool(bull.iloc[i])
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
                elif pnl > self.trail_start:
                    trail_price = p * (1 - self.trail_pct)
                    if self.trailing_stop is None or trail_price > self.trailing_stop:
                        self.trailing_stop = trail_price
                    if p <= self.trailing_stop:
                        sell = True
                        reason = f"浮动保利 {pnl * 100:+.1f}%"
                elif not bull.iloc[i] and self.prev_bull:
                    sell = True
                    reason = f"趋势破位 {pnl * 100:+.1f}%"

                self.prev_bull = bool(bull.iloc[i])

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
                    self.trailing_stop = None

        return signals
