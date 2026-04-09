"""简单的回测引擎"""
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
import logging

from strategies.ma_strategy import MAStrategy, Signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    entry_date: str
    exit_date: str
    action: str
    entry_price: float
    exit_price: float
    quantity: int
    profit: float
    profit_pct: float


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade]
    total_trades: int
    total_profit: float
    total_profit_pct: float
    win_rate: float
    final_equity: float


class SimpleBacktest:
    """简单回测引擎"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

    def run(self, df: pd.DataFrame, strategy: MAStrategy) -> BacktestResult:
        """执行回测"""
        logger.info("开始回测...")

        if df.empty:
            return BacktestResult(
                trades=[],
                total_trades=0,
                total_profit=0.0,
                total_profit_pct=0.0,
                win_rate=0.0,
                final_equity=self.initial_capital
            )

        signals = strategy.calculate_signals(df)
        trades = []
        equity_curve = [self.initial_capital]

        position = None
        position_size = 1000

        for i, row in df.iterrows():
            trade_date = row["trade_date"]
            close_price = row["close"]

            if position is None:
                for signal in signals:
                    if signal.trade_date == trade_date and signal.action == "buy":
                        position = {
                            "entry_price": close_price,
                            "entry_date": trade_date,
                            "quantity": position_size
                        }
                        self.current_capital -= close_price * position_size
                        logger.info(f"买入: {trade_date}, 价格: {close_price:.2f}")
                        break
            else:
                for signal in signals:
                    if signal.trade_date == trade_date and signal.action == "sell" and position:
                        profit = (close_price - position["entry_price"]) * position["quantity"]
                        profit_pct = (close_price / position["entry_price"] - 1) * 100

                        trade = Trade(
                            entry_date=position["entry_date"],
                            exit_date=trade_date,
                            action="buy",
                            entry_price=position["entry_price"],
                            exit_price=close_price,
                            quantity=position["quantity"],
                            profit=profit,
                            profit_pct=profit_pct
                        )

                        trades.append(trade)
                        self.current_capital += close_price * position["quantity"]
                        equity_curve.append(self.current_capital)

                        logger.info(
                            f"卖出: {trade_date}, 价格: {close_price:.2f}, "
                            f"盈利: {profit:.2f} ({profit_pct:.2f}%)"
                        )

                        position = None
                        break

            equity_curve.append(self.current_capital)

        win_trades = [t for t in trades if t.profit > 0]
        total_profit = sum(t.profit for t in trades)
        win_rate = len(win_trades) / len(trades) * 100 if trades else 0

        result = BacktestResult(
            trades=trades,
            total_trades=len(trades),
            total_profit=total_profit,
            total_profit_pct=(self.current_capital / self.initial_capital - 1) * 100,
            win_rate=win_rate,
            final_equity=self.current_capital
        )

        logger.info("=" * 50)
        logger.info(f"回测完成！")
        logger.info(f"总交易次数: {result.total_trades}")
        logger.info(f"总收益: {result.total_profit:.2f}")
        logger.info(f"总收益率: {result.total_profit_pct:.2f}%")
        logger.info(f"胜率: {result.win_rate:.2f}%")
        logger.info(f"最终资金: {result.final_equity:.2f}")
        logger.info("=" * 50)

        return result

    def print_trades(self, result: BacktestResult):
        """打印交易明细"""
        if not result.trades:
            print("无交易记录")
            return

        print("\n" + "=" * 80)
        print("交易明细:")
        print("=" * 80)
        print(f"{'序号':<6} {'买入日期':<12} {'卖出日期':<12} {'买入价':<10} {'卖出价':<10} {'数量':<8} {'盈利':<12} {'收益率':<10}")
        print("-" * 80)

        for i, trade in enumerate(result.trades, 1):
            print(
                f"{i:<6} {trade.entry_date:<12} {trade.exit_date:<12} "
                f"{trade.entry_price:<10.2f} {trade.exit_price:<10.2f} "
                f"{trade.quantity:<8} {trade.profit:>10.2f} {trade.profit_pct:>9.2f}%"
            )

        print("=" * 80)
