"""网格仓位管理系统
基于锋芒课程第八章"网格体系"，结合顶底图评分动态调整仓位。
90分（极低）→ 满仓进攻；60-90分 → 半仓滚动；<60分 → 空仓观望
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass


class PositionLevel(str, Enum):
    EXTREMELY_LOW = "极低位"
    LOW = "偏低"
    NEUTRAL = "中性"
    HIGH = "偏高"
    EXTREMELY_HIGH = "极高"


@dataclass
class PositionAdvice:
    score: float
    level: PositionLevel
    total_position_pct: float
    single_stock_pct: float
    max_stocks: int
    action: str
    reason: str


class TopBottomScorer:
    """顶底图评分计算器（0-100）
    评分越高代表市场越处于低位（越适合买入）。
    """

    def __init__(self, var1: float = 1.0, winner_lookback: int = 250):
        self.var1 = var1
        self.winner_lookback = winner_lookback

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

    def calculate_score(self, df: pd.DataFrame) -> float:
        if df.empty or len(df) < 100:
            return 50.0

        df = df.sort_values("trade_date").copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        var1 = self.var1

        var3 = close.rolling(13, min_periods=1).mean()
        var4 = 100 - np.abs((close - var3) / np.where(var3 > 1e-10, var3, 1e-10) * 100)

        var5 = low.rolling(75, min_periods=1).min()
        var6 = high.rolling(75, min_periods=1).max()
        var7 = (var6 - var5) / 100.0
        raw_var8 = np.where(var7 > 1e-10, (close - var5) / var7, 0.0)
        var8 = self._tdx_sma(raw_var8, 20, 1)
        vara = 3.0 * var8 - 2.0 * self._tdx_sma(var8, 15, 1)
        fight = (100 - vara) * var1

        winner_close_95 = self._winner(close * 0.95)
        sanpang = (winner_close_95 * 100).rolling(3, min_periods=1).mean() * var1

        winner_close = self._winner(close)
        var2 = 1.0 / np.where(winner_close > 1e-10, winner_close, 1e-10)
        cond1 = var2 > 5
        cond2 = var2 < 100
        val_if = np.where(cond1, np.where(cond2, var2, var4 - 10), 0)
        qieshou = (100 - val_if) * var1

        idx = -1
        fight_val = float(fight.iloc[idx])
        sanpang_val = float(sanpang.iloc[idx])
        qieshou_val = float(qieshou.iloc[idx])

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

        is_red_bar = bool(
            var18[idx] <= 25
            and var19.iloc[idx] < -95
            and var17.iloc[idx] > 50
            and var15[idx] < 22
            and var16[idx] < 28
        )

        fight_score = max(0, min(40, (100 - fight_val) * 0.4))
        qieshou_score = max(0, min(30, (100 - qieshou_val) * 0.3))
        sanpang_score = max(0, min(20, sanpang_val * 0.2))
        red_bar_score = 10.0 if is_red_bar else 0.0

        total_score = max(
            0, min(100, fight_score + qieshou_score + sanpang_score + red_bar_score)
        )

        logger.debug(
            f"顶底图评分: {total_score:.1f} "
            f"(fight={fight_val:.1f}→{fight_score:.1f}, "
            f"qieshou={qieshou_val:.1f}→{qieshou_score:.1f}, "
            f"sanpang={sanpang_val:.1f}→{sanpang_score:.1f}, "
            f"red_bar={'Y' if is_red_bar else 'N'}→{red_bar_score:.1f})"
        )

        return round(total_score, 1)


class GridPositionSizer:
    """网格仓位管理器 — 锋芒第八章网格体系
    根据顶底图评分分配仓位。
    """

    POSITION_CONFIG = {
        PositionLevel.EXTREMELY_LOW: {
            "total_pct": 1.0,
            "single_pct": 0.20,
            "max_stocks": 5,
            "action": "满仓进攻",
            "reason": "市场处于极低位，顶底图评分≥90，历史级别底部区域",
        },
        PositionLevel.LOW: {
            "total_pct": 0.8,
            "single_pct": 0.20,
            "max_stocks": 4,
            "action": "重仓操作",
            "reason": "市场偏低，顶底图评分75-90，适合积极布局",
        },
        PositionLevel.NEUTRAL: {
            "total_pct": 0.5,
            "single_pct": 0.15,
            "max_stocks": 4,
            "action": "半仓滚动",
            "reason": "市场中位，顶底图评分60-75，半仓操作降低风险",
        },
        PositionLevel.HIGH: {
            "total_pct": 0.3,
            "single_pct": 0.10,
            "max_stocks": 3,
            "action": "轻仓防守",
            "reason": "市场偏高，顶底图评分40-60，控制仓位等待回调",
        },
        PositionLevel.EXTREMELY_HIGH: {
            "total_pct": 0.0,
            "single_pct": 0.0,
            "max_stocks": 0,
            "action": "空仓观望",
            "reason": "市场处于极高位，顶底图评分<40，风险极大，建议空仓",
        },
    }

    def __init__(self, scorer: TopBottomScorer = None):
        self.scorer = scorer or TopBottomScorer()

    def _score_to_level(self, score: float) -> PositionLevel:
        if score >= 90:
            return PositionLevel.EXTREMELY_LOW
        elif score >= 75:
            return PositionLevel.LOW
        elif score >= 60:
            return PositionLevel.NEUTRAL
        elif score >= 40:
            return PositionLevel.HIGH
        else:
            return PositionLevel.EXTREMELY_HIGH

    def get_position_advice(self, score: float) -> PositionAdvice:
        level = self._score_to_level(score)
        config = self.POSITION_CONFIG[level]
        return PositionAdvice(
            score=score,
            level=level,
            total_position_pct=config["total_pct"],
            single_stock_pct=config["single_pct"],
            max_stocks=config["max_stocks"],
            action=config["action"],
            reason=config["reason"],
        )

    def evaluate_market(self, df: pd.DataFrame) -> PositionAdvice:
        score = self.scorer.calculate_score(df)
        return self.get_position_advice(score)

    def allocate_positions(
        self,
        total_cash: float,
        advice: PositionAdvice,
        stock_candidates: List[Dict],
    ) -> List[Dict]:
        if advice.total_position_pct <= 0 or not stock_candidates:
            return []

        position_cash = total_cash * advice.total_position_pct
        max_stocks = min(advice.max_stocks, len(stock_candidates))
        per_stock_budget = position_cash / max_stocks

        allocations = []
        for stock in stock_candidates[:max_stocks]:
            price = stock.get("price", 0)
            if price <= 0:
                continue
            shares = int(per_stock_budget / price / 100) * 100
            budget = shares * price
            allocations.append(
                {
                    "ts_code": stock["ts_code"],
                    "name": stock.get("name", ""),
                    "shares": shares,
                    "budget": round(budget, 2),
                    "budget_pct": round(budget / total_cash * 100, 1),
                }
            )

        logger.info(
            f"仓位分配: 总资金{total_cash:,.0f} → 仓位{advice.total_position_pct * 100:.0f}% "
            f"({position_cash:,.0f}) → {len(allocations)}只股票 "
            f"每只约{per_stock_budget:,.0f}"
        )

        return allocations

    def should_rebalance(
        self,
        current_score: float,
        previous_score: float,
        threshold: float = 10.0,
    ) -> Tuple[bool, str]:
        diff = current_score - previous_score
        current_level = self._score_to_level(current_score)
        previous_level = self._score_to_level(previous_score)

        if current_level != previous_level:
            return True, (
                f"仓位等级变化: {previous_level.value}→{current_level.value} "
                f"(评分{previous_score:.1f}→{current_score:.1f})"
            )

        if abs(diff) >= threshold:
            return (
                True,
                f"评分大幅波动: {diff:+.1f} ({previous_score:.1f}→{current_score:.1f})",
            )

        return False, f"评分变化不大: {diff:+.1f}，维持当前仓位"

    def get_portfolio_status(
        self,
        advice: PositionAdvice,
        current_positions: List[Dict],
        total_cash: float,
    ) -> Dict:
        current_total = total_cash + sum(
            p.get("market_value", 0) for p in current_positions
        )
        current_position_value = sum(
            p.get("market_value", 0) for p in current_positions
        )
        current_position_pct = (
            current_position_value / current_total if current_total > 0 else 0
        )

        target_position_value = current_total * advice.total_position_pct
        diff = target_position_value - current_position_value
        diff_pct = advice.total_position_pct - current_position_pct

        if abs(diff_pct) < 0.05:
            direction = "维持"
        elif diff_pct > 0:
            direction = "加仓"
        else:
            direction = "减仓"

        return {
            "score": advice.score,
            "level": advice.level.value,
            "action": advice.action,
            "reason": advice.reason,
            "current_total_assets": round(current_total, 2),
            "current_position_pct": round(current_position_pct * 100, 1),
            "target_position_pct": round(advice.total_position_pct * 100, 1),
            "diff_pct": round(diff_pct * 100, 1),
            "diff_amount": round(diff, 2),
            "direction": direction,
            "position_count": len(current_positions),
            "target_count": advice.max_stocks,
            "single_stock_limit_pct": round(advice.single_stock_pct * 100, 1),
        }
