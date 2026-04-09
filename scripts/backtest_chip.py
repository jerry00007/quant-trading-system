"""主力筹码趋向策略回测（中证100成分股）"""
import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.storage import DataStorage
from strategies.chip_strategy import ChipStrategy


INITIAL_CAPITAL = 100_000
POSITION_PCT = 0.20
BOARD_LOT = 100
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    profit: float
    profit_pct: float
    exit_reason: str


def calc_shares(capital, price):
    max_cost = capital * POSITION_PCT
    shares = int(max_cost / price) // BOARD_LOT * BOARD_LOT
    return shares if shares >= BOARD_LOT else 0


def run_backtest(df, strategy):
    capital = INITIAL_CAPITAL
    trades = []
    position = None

    signals = strategy.calculate_signals(df)
    sig_map = {}
    for s in signals:
        sig_map.setdefault(s.trade_date, []).append(s)

    for _, row in df.iterrows():
        date = row["trade_date"]
        price = row["close"]
        sigs = sig_map.get(date, [])

        if position is None:
            for sig in sigs:
                if sig.action == "buy":
                    shares = calc_shares(capital, price)
                    if shares == 0:
                        continue
                    cost = shares * price * (1 + COMMISSION_RATE)
                    if cost > capital:
                        continue
                    capital -= cost
                    position = {
                        "entry_date": date,
                        "entry_price": price,
                        "shares": shares,
                    }
                    break
        else:
            for sig in sigs:
                if sig.action == "sell":
                    revenue = position["shares"] * price * (
                        1 - COMMISSION_RATE - STAMP_TAX_RATE
                    )
                    capital += revenue
                    pnl = revenue - position["shares"] * position["entry_price"]
                    pnl_pct = (price / position["entry_price"] - 1) * 100
                    trades.append(
                        Trade(
                            entry_date=position["entry_date"],
                            exit_date=date,
                            entry_price=position["entry_price"],
                            exit_price=price,
                            shares=position["shares"],
                            profit=pnl,
                            profit_pct=pnl_pct,
                            exit_reason=sig.reason,
                        )
                    )
                    position = None
                    break

    if position is not None:
        last_price = df.iloc[-1]["close"]
        revenue = position["shares"] * last_price * (
            1 - COMMISSION_RATE - STAMP_TAX_RATE
        )
        capital += revenue
        pnl = revenue - position["shares"] * position["entry_price"]
        pnl_pct = (last_price / position["entry_price"] - 1) * 100
        trades.append(
            Trade(
                entry_date=position["entry_date"],
                exit_date=df.iloc[-1]["trade_date"],
                entry_price=position["entry_price"],
                exit_price=last_price,
                shares=position["shares"],
                profit=pnl,
                profit_pct=pnl_pct,
                exit_reason="期末强制平仓",
            )
        )

    total_profit = sum(t.profit for t in trades)
    wins = [t for t in trades if t.profit > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    return_pct = (capital / INITIAL_CAPITAL - 1) * 100

    stop_losses = [t for t in trades if "止损" in t.exit_reason]
    take_profits = [t for t in trades if "止盈" in t.exit_reason]
    chip_exits = [t for t in trades if "筹码" in t.exit_reason]
    forced = [t for t in trades if "期末" in t.exit_reason]

    return {
        "trades": trades,
        "total_trades": len(trades),
        "win_trades": len(wins),
        "win_rate": win_rate,
        "total_profit": total_profit,
        "return_pct": return_pct,
        "final_capital": capital,
        "stop_losses": len(stop_losses),
        "take_profits": len(take_profits),
        "chip_exits": len(chip_exits),
        "forced": len(forced),
    }


def main():
    storage = DataStorage()
    import sqlite3

    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"))
    codes = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
        ).fetchall()
    ]
    conn.close()

    print("=" * 80)
    print("  主力筹码趋向(ZLCMQ)策略回测")
    print(f"  初始资金: {INITIAL_CAPITAL:,}  仓位: {POSITION_PCT*100:.0f}%  手续费: {COMMISSION_RATE*100:.2f}%")
    print(f"  买入: 高位回落企稳 | 卖出: 止损-8% / 止盈+15% / ZLCMQ<15")
    print(f"  股票池: 中证100成分股 ({len(codes)}只)")
    print("=" * 80)

    strategy = ChipStrategy()
    results = []

    for code in codes:
        df = storage.get_daily_quotes(code)
        if df.empty or len(df) < 80:
            continue

        r = run_backtest(df, strategy)
        r["ts_code"] = code
        results.append(r)

    results.sort(key=lambda x: x["return_pct"], reverse=True)

    print(f"\n{'股票':<14} {'交易':>4} {'胜率':>6} {'收益率':>8} {'盈亏':>10} {'终值':>10} {'止损':>4} {'止盈':>4} {'筹码':>4}")
    print("-" * 80)

    for r in results:
        print(
            f"{r['ts_code']:<14} {r['total_trades']:>4} {r['win_rate']:>5.1f}% "
            f"{r['return_pct']:>+7.1f}% {r['total_profit']:>+10.0f} "
            f"{r['final_capital']:>10.0f} {r['stop_losses']:>4} {r['take_profits']:>4} {r['chip_exits']:>4}"
        )

    if not results:
        print("无有效回测结果")
        return

    avg_return = np.mean([r["return_pct"] for r in results])
    avg_win = np.mean([r["win_rate"] for r in results if r["total_trades"] > 0])
    profitable = sum(1 for r in results if r["return_pct"] > 0)
    total_trades_all = sum(r["total_trades"] for r in results)

    print("-" * 80)
    print(f"  股票数: {len(results)}  总交易: {total_trades_all}")
    print(f"  平均收益率: {avg_return:+.2f}%  平均胜率: {avg_win:.1f}%")
    print(f"  盈利股票: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)")
    print("=" * 80)

    winners = [r for r in results if r["total_trades"] > 0][:3]
    losers = [r for r in results if r["total_trades"] > 0][-3:]

    for label, picks in [("最佳3只", winners), ("最差3只", losers)]:
        print(f"\n  {label}:")
        for r in picks:
            print(f"  {r['ts_code']} ({r['return_pct']:+.1f}%)")
            for i, t in enumerate(r["trades"], 1):
                print(
                    f"    {i}. {t.entry_date}→{t.exit_date} "
                    f"{t.entry_price:.2f}→{t.exit_price:.2f} "
                    f"{t.profit_pct:+.1f}% [{t.exit_reason}]"
                )


if __name__ == "__main__":
    main()
