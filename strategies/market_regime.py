"""市场状态检测模块 — 识别趋势/震荡行情，动态调整策略参数

基于3个维度判断市场状态:
1. ADX (Average Directional Index): 趋势强度
2. 波动率体制 (Volatility Regime): VIX/ATD高低
3. 均线排列 (MA Alignment): 多头/空头/缠绕

状态映射:
- 强趋势 → 激进参数 (短lookback, 高仓位)
- 弱趋势 → 标准参数 (默认配置)
- 震荡   → 防守参数 (长lookback, 低仓位, 严格止损)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent

REGIME_TRENDING = "trending"
REGIME_RANGING = "ranging"
REGIME_NEUTRAL = "neutral"

PARAM_PRESETS = {
    REGIME_TRENDING: {
        "etf_lookback": 15, "etf_top_k": 2, "etf_rebalance": 15,
        "etf_stop_loss": 0.10, "etf_take_profit": 0.35,
        "position_scale": 1.0,
        "description": "强趋势 — 短周期快进快出，高仓位",
    },
    REGIME_NEUTRAL: {
        "etf_lookback": 20, "etf_top_k": 3, "etf_rebalance": 20,
        "etf_stop_loss": 0.08, "etf_take_profit": 0.30,
        "position_scale": 0.85,
        "description": "中性 — 标准参数",
    },
    REGIME_RANGING: {
        "etf_lookback": 30, "etf_top_k": 4, "etf_rebalance": 30,
        "etf_stop_loss": 0.05, "etf_take_profit": 0.15,
        "position_scale": 0.60,
        "description": "震荡 — 长周期低仓位，严格止损",
    },
}


@dataclass
class RegimeResult:
    regime: str
    confidence: float
    adx: float
    volatility_rank: float
    ma_alignment: str
    recommended_params: dict
    description: str


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]

        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx


def compute_volatility_rank(close: np.ndarray, lookback: int = 60,
                             window: int = 252) -> float:
    if len(close) < lookback + 1:
        return 0.5

    returns = np.diff(close[-lookback - 1:]) / close[-lookback - 1:-1]
    current_vol = np.std(returns)

    all_returns = np.diff(close) / close[:-1]
    n_windows = len(all_returns) // lookback
    if n_windows < 3:
        return 0.5

    hist_vols = []
    for i in range(n_windows):
        chunk = all_returns[i * lookback:(i + 1) * lookback]
        if len(chunk) == lookback:
            hist_vols.append(np.std(chunk))

    if not hist_vols:
        return 0.5

    rank = sum(1 for v in hist_vols if v < current_vol) / len(hist_vols)
    return rank


def compute_ma_alignment(close: np.ndarray, ma20: np.ndarray = None,
                          ma60: np.ndarray = None) -> Tuple[str, float]:
    if len(close) < 60:
        return "neutral", 0.0

    ma5 = np.mean(close[-5:])
    if ma20 is None:
        ma20 = np.mean(close[-20:])
    if ma60 is None:
        ma60 = np.mean(close[-60:])

    above_ma20 = close[-1] > ma20
    above_ma60 = close[-1] > ma60
    ma5_above_ma20 = ma5 > ma20
    ma5_above_ma60 = ma5 > ma60
    ma20_above_ma60 = ma20 > ma60

    bullish_count = sum([above_ma20, above_ma60, ma5_above_ma20, ma5_above_ma60, ma20_above_ma60])
    bearish_count = 5 - bullish_count

    if bullish_count >= 4:
        return "bullish", bullish_count / 5.0
    elif bearish_count >= 4:
        return "bearish", bearish_count / 5.0
    else:
        return "intertwined", 0.5


def detect_market_regime(df: pd.DataFrame, date: str = None) -> RegimeResult:
    df = df.sort_values("trade_date").reset_index(drop=True)

    if date:
        mask = df["trade_date"] <= date
        df = df.loc[mask].reset_index(drop=True)

    if len(df) < 80:
        return RegimeResult(
            regime=REGIME_NEUTRAL, confidence=0.0,
            adx=0, volatility_rank=0.5, ma_alignment="neutral",
            recommended_params=PARAM_PRESETS[REGIME_NEUTRAL],
            description="数据不足，使用标准参数",
        )

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    adx_full = compute_adx(high, low, close, period=14)
    adx_current = float(adx_full[-1]) if not np.isnan(adx_full[-1]) else 20.0

    vol_rank = compute_volatility_rank(close, lookback=20)

    ma_alignment, alignment_score = compute_ma_alignment(close)

    trend_score = 0.0
    if adx_current > 30:
        trend_score += 0.4
    elif adx_current > 25:
        trend_score += 0.2

    if ma_alignment == "bullish":
        trend_score += 0.3 * alignment_score
    elif ma_alignment == "bearish":
        trend_score += 0.15 * alignment_score

    if vol_rank > 0.7:
        trend_score += 0.2
    elif vol_rank < 0.3:
        trend_score -= 0.1

    if trend_score >= 0.5:
        regime = REGIME_TRENDING
    elif trend_score >= 0.25:
        regime = REGIME_NEUTRAL
    else:
        regime = REGIME_RANGING

    confidence = min(abs(trend_score - 0.375) / 0.375, 1.0)

    return RegimeResult(
        regime=regime,
        confidence=round(confidence, 3),
        adx=round(adx_current, 2),
        volatility_rank=round(vol_rank, 3),
        ma_alignment=ma_alignment,
        recommended_params=PARAM_PRESETS[regime],
        description=f"ADX={adx_current:.1f} | 波动率排名={vol_rank:.0%} | 均线={ma_alignment} → {PARAM_PRESETS[regime]['description']}",
    )


def get_regime_with_params(conn, index_code: str = "000300") -> RegimeResult:
    df = pd.read_sql(
        f"SELECT * FROM etf_daily_quotes WHERE etf_code='sz159919' ORDER BY trade_date",
        conn
    )
    if df.empty or len(df) < 100:
        return RegimeResult(
            regime=REGIME_NEUTRAL, confidence=0.0,
            adx=0, volatility_rank=0.5, ma_alignment="neutral",
            recommended_params=PARAM_PRESETS[REGIME_NEUTRAL],
            description="使用沪深300ETF代理，数据不足",
        )
    return detect_market_regime(df)
