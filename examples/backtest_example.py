"""回测示例脚本"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.ma_strategy import MAStrategy
from backtest.simple_backtest import SimpleBacktest


def generate_mock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """生成模拟数据"""
    np.random.seed(42)

    dates = pd.date_range(
        end=datetime.now(),
        periods=days,
        freq="D"
    ).strftime("%Y%m%d").tolist()

    data = []
    base_price = 100.0
    prev_close = base_price

    for i, date in enumerate(dates):
        ret = np.random.normal(0.001, 0.02)
        close = prev_close * (1 + ret)
        open_price = close * (1 + np.random.uniform(-0.01, 0.01))
        high_price = close * (1 + abs(np.random.uniform(0, 0.02)))
        low_price = close * (1 - abs(np.random.uniform(0, 0.02)))
        vol = np.random.randint(100000, 500000)
        amount = np.random.randint(10000000, 50000000)

        data.append({
            "ts_code": symbol,
            "trade_date": date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close,
            "pre_close": prev_close,
            "change": close - prev_close,
            "pct_chg": ((close / prev_close) - 1) * 100,
            "vol": vol,
            "amount": amount
        })

        prev_close = close

    return pd.DataFrame(data)


def run_example():
    """运行示例回测"""
    print("=" * 60)
    print("双均线交叉策略回测示例")
    print("=" * 60)

    symbol = "600000"
    days = 252

    print(f"\n生成模拟数据: {symbol}, {days} 个交易日")
    df = generate_mock_data(symbol, days)
    print(f"数据日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
    print(f"数据预览:\n{df.head(10)}")

    print("\n" + "-" * 60)
    print("初始化策略: MA5 / MA20")
    strategy = MAStrategy(fast_period=5, slow_period=20)

    print("\n" + "-" * 60)
    print("开始回测...")
    print("-" * 60)

    backtest = SimpleBacktest(initial_capital=100000)
    result = backtest.run(df, strategy)

    print("\n" + "=" * 60)
    print("回测结果汇总:")
    print("=" * 60)
    print(f"初始资金: {backtest.initial_capital:.2f}")
    print(f"最终资金: {result.final_equity:.2f}")
    print(f"总收益: {result.total_profit:.2f}")
    print(f"总收益率: {result.total_profit_pct:.2f}%")
    print(f"交易次数: {result.total_trades}")
    print(f"胜率: {result.win_rate:.2f}%")

    backtest.print_trades(result)

    print("\n" + "=" * 60)
    print("策略参数优化建议:")
    print("=" * 60)
    print("当前参数: MA5 / MA20")
    print("建议优化方向:")
    print("  1. 调整快慢均线周期 (如 MA10 / MA30)")
    print("  2. 添加止损止盈规则")
    print("  3. 考虑交易手续费影响")
    print("=" * 60)


if __name__ == "__main__":
    run_example()
