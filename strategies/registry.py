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
