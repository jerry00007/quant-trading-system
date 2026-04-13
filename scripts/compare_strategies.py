"""多策略对比回测（修复前视偏差）"""
import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.storage import DataStorage
from strategies.enhanced_chip_strategy import EnhancedChipStrategy
from strategies.benchmark_strategies import DualMAStrategy, BollingerBandStrategy, RSIStrategy
from strategies.advanced_strategies import MACDDivergenceStrategy, ModifiedTurtleStrategy, MomentumReversalStrategy


INITIAL_CAPITAL = 100_000
POSITION_PCT = 0.20
BOARD_LOT = 100
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005


def run_backtest(df, strategy):
    capital = INITIAL_CAPITAL
    trades = []
    position = None
    pending_buy = None
    pending_sell_reason = None

    signals = strategy.calculate_signals(df)
    sig_map = {}
    for s in signals:
        sig_map.setdefault(s.trade_date, []).append(s)

    dates = df["trade_date"].tolist()
    opens = df["open"].tolist()

    for idx in range(len(dates)):
        date = dates[idx]
        sigs = sig_map.get(date, [])

        if pending_buy is not None:
            exec_price = opens[idx]
            max_cost = capital * POSITION_PCT
            shares = int(max_cost / exec_price) // BOARD_LOT * BOARD_LOT
            if shares >= BOARD_LOT:
                cost = shares * exec_price * (1 + COMMISSION_RATE)
                if cost <= capital:
                    capital -= cost
                    position = {"entry_date": date, "entry_price": exec_price, "shares": shares}
            pending_buy = None

        if position is not None and pending_sell_reason is not None:
            exec_price = opens[idx]
            revenue = position["shares"] * exec_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            capital += revenue
            pnl_pct = (exec_price / position["entry_price"] - 1) * 100
            trades.append({"pnl_pct": pnl_pct, "reason": pending_sell_reason})
            position = None
            pending_sell_reason = None

        for sig in sigs:
            if position is None and pending_buy is None and sig.action == "buy":
                pending_buy = sig
            elif position is not None and pending_sell_reason is None and sig.action == "sell":
                pending_sell_reason = sig.reason
                break

    if position is not None:
        last_price = df.iloc[-1]["close"]
        capital += position["shares"] * last_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
        trades.append({"pnl_pct": (last_price / position["entry_price"] - 1) * 100, "reason": "期末平仓"})

    if not trades:
        return None

    returns = [t["pnl_pct"] for t in trades]
    wins = sum(1 for r in returns if r > 0)
    sharpe = (np.mean(returns) / (np.std(returns) + 0.001)) * np.sqrt(252) if len(returns) > 1 else 0

    return {
        "total_trades": len(trades),
        "win_rate": wins / len(trades) * 100,
        "avg_return": np.mean(returns),
        "return_pct": (capital / INITIAL_CAPITAL - 1) * 100,
        "sharpe": sharpe,
        "profitable": 1 if capital > INITIAL_CAPITAL else 0,
    }


def main():
    storage = DataStorage()
    import sqlite3
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"))
    codes = [r[0] for r in conn.execute("SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code").fetchall()]
    conn.close()

    strategies = {
        "增强筹码(ZLCMQ)": EnhancedChipStrategy(),
        "双均线(5/20)": DualMAStrategy(),
        "布林带(20,2)": BollingerBandStrategy(),
        "RSI(14,30/70)": RSIStrategy(),
        "MACD背离(12,26,9)": MACDDivergenceStrategy(),
        "改良海龟(20/10)": ModifiedTurtleStrategy(),
        "动量反转(20日)": MomentumReversalStrategy(),
    }

    print("=" * 100)
    print("  多策略对比回测（修复前视偏差，次日开盘价执行）")
    print(f"  股票池: 中证100 ({len(codes)}只)  初始资金: {INITIAL_CAPITAL:,}  仓位: {POSITION_PCT*100:.0f}%")
    print("=" * 100)

    all_results = {name: [] for name in strategies}

    for code in codes:
        df = storage.get_daily_quotes(code)
        if df.empty or len(df) < 80:
            continue
        for name, strategy in strategies.items():
            result = run_backtest(df, strategy)
            if result:
                all_results[name].append(result)

    print(f"\n{'策略':<20} {'股票数':>6} {'总交易':>6} {'盈利比':>6} {'平均收益':>10} {'胜率':>8} {'平均夏普':>8}")
    print("-" * 100)

    summary = []
    for name, results in all_results.items():
        if not results:
            continue
        total_stocks = len(results)
        total_trades = sum(r["total_trades"] for r in results)
        profitable = sum(r["profitable"] for r in results)
        avg_return = np.mean([r["return_pct"] for r in results])
        avg_win = np.mean([r["win_rate"] for r in results if r["total_trades"] > 0])
        avg_sharpe = np.mean([r["sharpe"] for r in results if r["total_trades"] > 1])

        summary.append({
            "name": name,
            "stocks": total_stocks,
            "trades": total_trades,
            "profitable": profitable,
            "avg_return": avg_return,
            "win_rate": avg_win,
            "sharpe": avg_sharpe,
        })

        print(f"{name:<20} {total_stocks:>6} {total_trades:>6} {profitable/total_stocks*100:>5.0f}% {avg_return:>+9.2f}% {avg_win:>7.1f}% {avg_sharpe:>8.2f}")

    summary.sort(key=lambda x: x["avg_return"], reverse=True)

    print("-" * 100)
    print(f"\n排名:")
    for i, s in enumerate(summary, 1):
        print(f"  {i}. {s['name']}: 平均收益 {s['avg_return']:+.2f}%  胜率 {s['win_rate']:.1f}%  夏普 {s['sharpe']:.2f}")

    print("=" * 100)


if __name__ == "__main__":
    main()
