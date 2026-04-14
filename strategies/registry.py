"""策略注册表"""

from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = {}


@dataclass
class StrategyInfo:
    name: str
    cls: Type
    config_schema: Type[BaseModel]
    description: str = ""
    supported_features: List[str] = field(default_factory=list)


STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {}


def register_strategy(
    name: str,
    cls: Type,
    config_schema: Type[BaseModel] = StrategyConfig,
    description: str = "",
    supported_features: Optional[List[str]] = None,
):
    STRATEGY_REGISTRY[name] = StrategyInfo(
        name=name,
        cls=cls,
        config_schema=config_schema,
        description=description,
        supported_features=supported_features or [],
    )


def get_strategy(name: str) -> Optional[StrategyInfo]:
    return STRATEGY_REGISTRY.get(name)


def list_strategies() -> List[Dict[str, str]]:
    return [
        {
            "name": info.name,
            "description": info.description,
            "supported_features": ", ".join(info.supported_features),
        }
        for info in STRATEGY_REGISTRY.values()
    ]


def validate_strategy_config(name: str, config: dict) -> BaseModel:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"策略 {name} 不存在")
    schema = STRATEGY_REGISTRY[name].config_schema
    return schema(**config)


def register_builtin_strategies():
    from strategies.ma_strategy import MAStrategy
    from strategies.etf_rotation_strategy import ETFRotationStrategy
    from strategies.enhanced_chip_strategy import EnhancedChipStrategy
    from strategies.multifactor_strategy import MultifactorStrategy

    class MAConfig(BaseModel):
        fast_period: int = Field(default=5, ge=1, le=200)
        slow_period: int = Field(default=20, ge=1, le=500)
        position: Optional[str] = None

    class ETFConfig(BaseModel):
        lookback: int = Field(default=20, ge=5, le=100)
        top_k: int = Field(default=3, ge=1, le=10)
        rebalance_days: int = Field(default=20, ge=1, le=60)
        stop_loss: float = Field(default=0.08, ge=0, le=0.5)
        take_profit: float = Field(default=0.30, ge=0, le=1.0)

    class ChipConfig(BaseModel):
        atr_period: int = Field(default=14, ge=5, le=50)
        stop_loss_atr: float = Field(default=2.5, ge=0.5, le=10)
        stop_loss_fixed: float = Field(default=0.10, ge=0, le=0.5)
        take_profit_move: float = Field(default=0.05, ge=0, le=0.3)
        take_profit_fixed: float = Field(default=0.20, ge=0, le=1.0)

    class MultifactorConfig(BaseModel):
        top_n: int = Field(default=10, ge=1, le=50)
        rebalance_days: int = Field(default=20, ge=5, le=60)
        momentum_days: int = Field(default=20, ge=5, le=100)
        stop_loss: float = Field(default=0.10, ge=0, le=0.5)
        w_cap: float = Field(default=0.333, ge=0, le=1)
        w_quality: float = Field(default=0.333, ge=0, le=1)
        w_momentum: float = Field(default=0.333, ge=0, le=1)

    register_strategy(
        "ma",
        MAStrategy,
        MAConfig,
        description="双均线交叉策略",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "etf_rotation",
        ETFRotationStrategy,
        ETFConfig,
        description="ETF动量轮动策略",
        supported_features=["backtest", "signals", "portfolio"],
    )
    register_strategy(
        "enhanced_chip",
        EnhancedChipStrategy,
        ChipConfig,
        description="增强筹码选股策略",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "multifactor",
        MultifactorStrategy,
        MultifactorConfig,
        description="多因子选股策略",
        supported_features=["backtest", "signals"],
    )

    # --- 新增策略 ---

    from strategies.fengmang_strategies import (
        VolumeBreakoutStrategy,
        DragonFirstYinStrategy,
        TrendMAStrategy,
    )
    from strategies.top_bottom_strategy import TopBottomStrategy
    from strategies.momentum_strategies import (
        BollingerBreakStrategy,
        RSIMomentumStrategy,
        MACDCrossStrategy,
    )

    class VolumeBreakoutConfig(BaseModel):
        box_days: int = Field(default=20, ge=5, le=60)
        vol_mult: float = Field(default=2.0, ge=1.0, le=5.0)
        stop_loss_pct: float = Field(default=-0.08, ge=-0.2, le=0.0)
        take_profit_pct: float = Field(default=0.20, ge=0.0, le=0.5)
        trail_start: float = Field(default=0.10, ge=0.0, le=0.3)
        trail_pct: float = Field(default=0.05, ge=0.01, le=0.2)

    class DragonFirstYinConfig(BaseModel):
        limit_pct: float = Field(default=0.095, ge=0.05, le=0.15)
        min_limits: int = Field(default=2, ge=1, le=5)
        yin_pct: float = Field(default=-0.03, ge=-0.10, le=0.0)
        stop_loss_pct: float = Field(default=-0.05, ge=-0.15, le=0.0)
        take_profit_pct: float = Field(default=0.10, ge=0.0, le=0.3)
        max_hold_days: int = Field(default=3, ge=1, le=10)

    class TrendMAConfig(BaseModel):
        ma_periods: str = Field(default="5,10,20,30")
        stop_loss_pct: float = Field(default=-0.05, ge=-0.15, le=0.0)
        trail_start: float = Field(default=0.05, ge=0.0, le=0.2)
        trail_pct: float = Field(default=0.03, ge=0.01, le=0.1)

    class TopBottomConfig(BaseModel):
        var1: float = Field(default=1.0, ge=0.5, le=2.0)
        winner_lookback: int = Field(default=250, ge=50, le=500)
        stop_loss_pct: float = Field(default=-0.08, ge=-0.2, le=0.0)
        take_profit_pct: float = Field(default=0.15, ge=0.0, le=0.5)

    class BollingerBreakConfig(BaseModel):
        period: int = Field(default=25, ge=5, le=60)
        std_dev: float = Field(default=2.5, ge=1.0, le=4.0)
        stop_pct: float = Field(default=0.05, ge=0.01, le=0.2)

    class RSIMomentumConfig(BaseModel):
        period: int = Field(default=12, ge=5, le=30)
        oversold: float = Field(default=25, ge=10, le=40)
        overbought: float = Field(default=80, ge=60, le=95)
        stop_pct: float = Field(default=0.05, ge=0.01, le=0.2)

    class MACDCrossConfig(BaseModel):
        fast_period: int = Field(default=15, ge=5, le=30)
        slow_period: int = Field(default=26, ge=10, le=60)
        signal_period: int = Field(default=13, ge=5, le=30)
        stop_pct: float = Field(default=0.05, ge=0.01, le=0.2)

    register_strategy(
        "vol_breakout",
        VolumeBreakoutStrategy,
        VolumeBreakoutConfig,
        description="爆量突破策略(锋芒) — 低位横盘后爆量突破20日高点",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "dragon_first_yin",
        DragonFirstYinStrategy,
        DragonFirstYinConfig,
        description="龙头首阴反抽策略(锋芒) — 连续涨停后首阴次日低吸",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "trend_ma",
        TrendMAStrategy,
        TrendMAConfig,
        description="均线趋势跟踪策略(锋芒) — 三阶段均线系统捕捉完整波段",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "top_bottom",
        TopBottomStrategy,
        TopBottomConfig,
        description="顶底图策略 — 通达信顶底图指标识别顶部底部区域",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "bollinger_break",
        BollingerBreakStrategy,
        BollingerBreakConfig,
        description="布林带突破策略 — 突破下轨买入，突破上轨卖出",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "rsi_momentum",
        RSIMomentumStrategy,
        RSIMomentumConfig,
        description="RSI超买超卖策略 — RSI<25买入，RSI>80卖出",
        supported_features=["backtest", "signals"],
    )
    register_strategy(
        "macd_cross",
        MACDCrossStrategy,
        MACDCrossConfig,
        description="MACD金叉死叉策略 — MACD金叉买入，死叉卖出",
        supported_features=["backtest", "signals"],
    )
