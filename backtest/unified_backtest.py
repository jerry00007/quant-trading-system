"""统一策略回测模块 - 支持所有注册策略（预计算信号优化版）"""

import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    ts_code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_days: int
    reason: str


@dataclass
class BacktestResult:
    strategy_name: str
    initial_capital: float
    final_value: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    avg_holding_days: float
    annual_return: float
    status: str = "success"
    error: str = ""


class UnifiedBacktester:
    def __init__(
        self,
        db_path: str = "data/sqlite/stock_data.db",
        initial_capital: float = 1_000_000,
        position_size: int = 10_000,
        max_positions: int = 5,
        commission_rate: float = 0.0003,
    ):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.commission_rate = commission_rate

    def load_data(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2026-04-10",
        top_stocks: int = 100,
    ) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        stock_count = """
            SELECT ts_code, COUNT(*) as cnt
            FROM daily_quotes
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY ts_code
            HAVING cnt >= 200
            ORDER BY cnt DESC
            LIMIT ?
        """
        active_stocks = conn.execute(
            stock_count, (start_date, end_date, top_stocks)
        ).fetchall()
        active_codes = [s[0] for s in active_stocks]
        if not active_codes:
            conn.close()
            return pd.DataFrame()

        placeholders = ",".join("?" * len(active_codes))
        query = f"""
            SELECT ts_code, trade_date, open, high, low, close, vol,
                   pct_chg, change, amount
            FROM daily_quotes
            WHERE ts_code IN ({placeholders})
              AND trade_date BETWEEN ? AND ?
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql(query, conn, params=active_codes + [start_date, end_date])
        conn.close()

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        for col in ["open", "high", "low", "close", "vol", "pct_chg", "change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            f"数据: {len(df)} 条, {df['ts_code'].nunique()} 股, {df['trade_date'].min().date()}~{df['trade_date'].max().date()}"
        )
        return df

    def precompute_signals(
        self, data: pd.DataFrame, strategy, lookback: int = 60
    ) -> Dict[str, List]:
        stock_signals: Dict[str, List] = {}
        stocks = data["ts_code"].unique()
        for i, ts in enumerate(stocks):
            stock_data = data[data["ts_code"] == ts].sort_values("trade_date")
            if len(stock_data) < 60:
                stock_signals[ts] = []
                continue
            try:
                signals = strategy.calculate_signals(stock_data)
                stock_signals[ts] = signals
            except Exception:
                stock_signals[ts] = []
            if (i + 1) % 50 == 0:
                logger.info(f"  信号预计算: {i + 1}/{len(stocks)}")
        return stock_signals

    def run(
        self,
        strategy,
        strategy_name: str,
        data: pd.DataFrame,
        lookback: int = 60,
    ) -> BacktestResult:
        try:
            return self._run_backtest(strategy, strategy_name, data, lookback)
        except Exception as e:
            logger.exception(f"回测失败: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0,
                total_return_pct=0,
                sharpe_ratio=0,
                max_drawdown_pct=0,
                win_rate=0,
                total_trades=0,
                winning_trades=0,
                avg_holding_days=0,
                annual_return=0,
                status="error",
                error=str(e),
            )

    def _run_backtest(
        self,
        strategy,
        strategy_name: str,
        data: pd.DataFrame,
        lookback: int,
    ) -> BacktestResult:
        logger.info(f"  预计算信号...")
        all_signals = self.precompute_signals(data, strategy, lookback)

        cash = self.initial_capital
        positions: Dict[str, Dict] = {}
        trades: List[Trade] = []
        equity_curve: List[float] = [self.initial_capital]
        daily_returns_list: List[float] = [0.0]

        trading_dates = sorted(data["trade_date"].unique())
        lookback_dates = [
            d for d in trading_dates if d >= trading_dates[0] + timedelta(days=lookback)
        ]

        for date in lookback_dates:
            date_str = str(date.date())
            date_data = data[data["trade_date"] == date]
            if date_data.empty:
                equity_curve.append(equity_curve[-1])
                daily_returns_list.append(0)
                continue

            pos_value = (
                sum(
                    pos["current_price"] * pos["qty"]
                    for pos in positions.values()
                    if pos.get("current_price")
                )
                or 0
            )
            prev_total = cash + pos_value

            for ts_code, pos in list(positions.items()):
                sd = date_data[date_data["ts_code"] == ts_code]
                if not sd.empty:
                    pos["current_price"] = sd.iloc[0]["close"]

            sell_keys = []
            for ts_code, pos in positions.items():
                sigs = all_signals.get(ts_code, [])
                for s in sigs:
                    s_date = (
                        str(s.trade_date)[:10]
                        if hasattr(s.trade_date, "date")
                        else str(s.trade_date)
                    )
                    if s.action == "sell" and s_date == date_str:
                        current_price = pos.get("current_price", pos["entry_price"])
                        entry_dt = datetime.strptime(pos["entry_date"], "%Y-%m-%d")
                        exit_dt = datetime.strptime(date_str, "%Y-%m-%d")
                        hold_days = (exit_dt - entry_dt).days
                        qty = pos["qty"]
                        gross = (current_price - pos["entry_price"]) * qty
                        comm = current_price * qty * self.commission_rate
                        net = gross - comm * 2
                        cash += current_price * qty - comm
                        trades.append(
                            Trade(
                                ts_code,
                                pos["entry_date"],
                                date_str,
                                pos["entry_price"],
                                current_price,
                                qty,
                                net,
                                (current_price / pos["entry_price"] - 1) * 100,
                                hold_days,
                                s.reason or "",
                            )
                        )
                        sell_keys.append(ts_code)
                        break
            for k in sell_keys:
                positions.pop(k, None)

            if len(positions) < self.max_positions:
                avail = self.max_positions - len(positions)
                candidates = []
                for ts_code in date_data["ts_code"].values:
                    if ts_code in positions:
                        continue
                    sigs = all_signals.get(ts_code, [])
                    for s in sigs:
                        s_date = (
                            str(s.trade_date)[:10]
                            if hasattr(s.trade_date, "date")
                            else str(s.trade_date)
                        )
                        if s.action == "buy" and s_date == date_str:
                            price = date_data[date_data["ts_code"] == ts_code].iloc[0][
                                "close"
                            ]
                            if price > 0:
                                candidates.append(
                                    (ts_code, price, getattr(s, "confidence", 0.8))
                                )
                            break

                candidates.sort(key=lambda x: -x[2])
                for ts_code, price, conf in candidates[:avail]:
                    qty = int(self.position_size / price / 100) * 100
                    if qty < 100:
                        continue
                    cost = price * qty * (1 + self.commission_rate)
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[ts_code] = {
                        "qty": qty,
                        "entry_price": price,
                        "entry_date": date_str,
                        "entry_reason": f"conf{conf:.2f}",
                        "current_price": price,
                    }

            pos_value = sum(
                pos.get("current_price", pos["entry_price"]) * pos["qty"]
                for pos in positions.values()
            )
            total = cash + pos_value
            equity_curve.append(total)
            daily_ret = (total / prev_total - 1) if prev_total > 0 else 0
            daily_returns_list.append(daily_ret)

        equity = np.array(equity_curve)
        daily_rets = np.array(daily_returns_list[1:])
        final_value = equity[-1]
        total_ret = final_value - self.initial_capital
        total_ret_pct = (final_value / self.initial_capital - 1) * 100
        n_years = len(lookback_dates) / 252
        ann_ret = (
            ((final_value / self.initial_capital) ** (1 / n_years) - 1) * 100
            if n_years > 0
            else 0
        )
        sharpe = (
            (daily_rets.mean() / daily_rets.std() * np.sqrt(252))
            if daily_rets.std() > 1e-10
            else 0
        )
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        max_dd = abs(dd.min())
        winners = [t for t in trades if t.pnl > 0]
        win_rate = len(winners) / len(trades) * 100 if trades else 0
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0

        return BacktestResult(
            strategy_name=strategy_name,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_ret,
            total_return_pct=total_ret_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=len(winners),
            avg_holding_days=avg_hold,
            annual_return=ann_ret,
            status="success",
        )


STRATEGY_CLASSES = {
    "ma": "strategies.ma_strategy:MAStrategy",
    "enhanced_chip": "strategies.enhanced_chip_strategy:EnhancedChipStrategy",
    "vol_breakout": "strategies.fengmang_strategies:VolumeBreakoutStrategy",
    "dragon_first_yin": "strategies.fengmang_strategies:DragonFirstYinStrategy",
    "trend_ma": "strategies.fengmang_strategies:TrendMAStrategy",
    "top_bottom": "strategies.top_bottom_strategy:TopBottomStrategy",
    "bollinger_break": "strategies.momentum_strategies:BollingerBreakStrategy",
    "rsi_momentum": "strategies.momentum_strategies:RSIMomentumStrategy",
    "macd_cross": "strategies.momentum_strategies:MACDCrossStrategy",
}

STRATEGY_NAMES = {
    "ma": "双均线",
    "enhanced_chip": "增强筹码",
    "vol_breakout": "爆量突破",
    "dragon_first_yin": "龙头首阴",
    "trend_ma": "均线趋势",
    "top_bottom": "顶底图",
    "bollinger_break": "布林带",
    "rsi_momentum": "RSI动量",
    "macd_cross": "MACD交叉",
}


def run_all_strategies_backtest(
    start_date: str = "2023-01-01",
    end_date: str = "2026-04-10",
    db_path: str = "data/sqlite/stock_data.db",
    top_stocks: int = 100,
) -> List[BacktestResult]:
    import sys

    sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")

    tester = UnifiedBacktester(db_path=db_path)
    data = tester.load_data(
        start_date=start_date, end_date=end_date, top_stocks=top_stocks
    )
    if data.empty:
        logger.error("无数据!")
        return []

    results = []
    for name, path in STRATEGY_CLASSES.items():
        module_path, class_name = path.split(":")
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        strategy = cls()
        full_name = f"{name} ({STRATEGY_NAMES.get(name, name)})"
        logger.info(f"\n{'=' * 60}\n回测: {full_name}")
        result = tester.run(strategy, name, data)
        results.append(result)
        if result.status == "success":
            logger.info(
                f"  收益: {result.total_return_pct:+.2f}% | 年化: {result.annual_return:+.2f}% | "
                f"夏普: {result.sharpe_ratio:.2f} | 回撤: {result.max_drawdown_pct:.2f}% | "
                f"交易: {result.total_trades} | 胜率: {result.win_rate:.1f}%"
            )
        else:
            logger.error(f"  失败: {result.error}")

    return results


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")

    results = run_all_strategies_backtest()

    print("\n" + "=" * 85)
    print("📊 策略回测汇总报告 (2023-01-01 ~ 2026-04-10, 100只股票)")
    print("=" * 85)
    print(
        f"{'策略':<18} {'总收益':>10} {'年化':>8} {'夏普':>6} {'最大回撤':>10} {'交易数':>6} {'胜率':>8} {'均持仓天':>8}"
    )
    print("-" * 85)

    sorted_results = sorted(results, key=lambda x: x.total_return_pct, reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(sorted_results):
        medal = medals[i] if i < 3 else "  "
        print(
            f"{medal} {r.strategy_name:<16} {r.total_return_pct:>+9.2f}% "
            f"{r.annual_return:>+7.2f}% {r.sharpe_ratio:>6.2f} "
            f"{r.max_drawdown_pct:>9.2f}% {r.total_trades:>6} {r.win_rate:>7.1f}% "
            f"{r.avg_holding_days:>7.1f}"
        )
    print("=" * 85)
