"""策略模块"""
from .ma_strategy import MAStrategy, Signal
from .registry import (
    register_strategy,
    get_strategy,
    list_strategies,
    validate_strategy_config,
    register_builtin_strategies,
)

register_builtin_strategies()

__all__ = [
    "MAStrategy",
    "Signal",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "validate_strategy_config",
]
