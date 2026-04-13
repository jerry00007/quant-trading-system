"""增强型筹码策略参数网格搜索"""
import sys
from pathlib import Path
from itertools import product
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from data.storage import DataStorage
from strategies.enhanced_chip_strategy import EnhancedChipStrategy


INITIAL_CAPITAL = 100_000
POSITION_PCT = 0.20
BOARD_LOT = 100
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005


@dataclass
class GridSearchResult:
    params: dict
    avg_return: float
    avg_win_rate: float
    total_trades: int
    profitable_ratio: float
    sharpe_like: float


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
                    position = {"entry_date": date, "entry_price": price, "shares": shares}
                    break
        else:
            for sig in sigs:
                if sig.action == "sell":
                    revenue = position["shares"] * price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
                    capital += revenue
                    pnl_pct = (price / position["entry_price"] - 1) * 100
                    trades.append({"profit_pct": pnl_pct, "entry_date": position["entry_date"], "exit_date": date})
                    position = None
                    break

    if position is not None:
        last_price = df.iloc[-1]["close"]
        capital += position["shares"] * last_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
        trades.append({
            "profit_pct": (last_price / position["entry_price"] - 1) * 100,
            "entry_date": position["entry_date"],
            "exit_date": df.iloc[-1]["trade_date"]
        })

    return trades, capital


def evaluate_params(df, params):
    strategy = EnhancedChipStrategy(**params)
    trades, final_capital = run_backtest(df, strategy)

    if not trades:
        return None

    returns = [t["profit_pct"] for t in trades]
    wins = sum(1 for r in returns if r > 0)

    return {
        "avg_return": np.mean(returns),
        "win_rate": wins / len(returns) * 100 if returns else 0,
        "total_trades": len(trades),
        "profitable_ratio": wins / len(returns) if returns else 0,
        "final_capital": final_capital,
        "returns_std": np.std(returns) if len(returns) > 1 else 1,
    }


def grid_search(storage, stock_codes, param_grid):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    print(f"参数网格搜索")
    print(f"参数组合数: {len(combinations)}")
    print(f"参数范围:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print()

    all_results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))
        print(f"[{i}/{len(combinations)}] 测试: {params}")

        stock_results = []
        for code in stock_codes:
            df = storage.get_daily_quotes(code)
            if df.empty or len(df) < 80:
                continue

            result = evaluate_params(df, params)
            if result and result["total_trades"] > 0:
                stock_results.append(result)

        if not stock_results:
            print(f"  -> 无有效结果")
            continue

        avg_return = np.mean([r["avg_return"] for r in stock_results])
        avg_win_rate = np.mean([r["win_rate"] for r in stock_results])
        total_trades = sum(r["total_trades"] for r in stock_results)
        profitable_ratio = sum(1 for r in stock_results if r["avg_return"] > 0) / len(stock_results)

        sharpe_like = avg_return / (np.std([r["avg_return"] for r in stock_results]) + 0.1)

        grid_result = GridSearchResult(
            params=params,
            avg_return=avg_return,
            avg_win_rate=avg_win_rate,
            total_trades=total_trades,
            profitable_ratio=profitable_ratio,
            sharpe_like=sharpe_like,
        )
        all_results.append(grid_result)

        print(f"  -> 平均收益: {avg_return:+.2f}%  胜率: {avg_win_rate:.1f}%  交易数: {total_trades}  盈利比: {profitable_ratio:.1%}")

    return all_results


def print_top_results(all_results, top_n=10):
    if not all_results:
        print("无有效结果")
        return

    all_results.sort(key=lambda x: (x.avg_return, x.sharpe_like), reverse=True)

    print(f"\n{'排名':>4} {'参数组合':<60} {'收益':>8} {'胜率':>6} {'夏普':>6} {'交易数':>6}")
    print("-" * 100)

    for i, r in enumerate(all_results[:top_n], 1):
        param_str = str(r.params)
        if len(param_str) > 58:
            param_str = param_str[:55] + "..."
        print(f"{i:>4} {param_str:<60} {r.avg_return:>+7.2f}% {r.avg_win_rate:>5.1f}% {r.sharpe_like:>6.2f} {r.total_trades:>6}")


def main():
    storage = DataStorage()
    import sqlite3

    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"))
    codes = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code LIMIT 20"
        ).fetchall()
    ]
    conn.close()

    print(f"股票池: {len(codes)}只 (前10只用于快速搜索)")
    codes = codes[:10]
    print("=" * 100)

    param_grid = {
        "n_days": [3, 5],
        "min_high": [95, 98],
        "min_fall": [3, 5],
        "stop_loss_atr_mult": [2.0, 2.5],
        "take_profit_pct": [0.15, 0.20],
        "chip_exit": [12, 15],
    }

    results = grid_search(storage, codes, param_grid)

    print("\n" + "=" * 100)
    print("TOP 10 最优参数组合:")
    print_top_results(results, 10)

    if results:
        best = max(results, key=lambda x: (x.avg_return, x.sharpe_like))
        print("\n" + "=" * 100)
        print("最优参数:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print(f"  -> 平均收益: {best.avg_return:+.2f}%")
        print(f"  -> 平均胜率: {best.avg_win_rate:.1f}%")
        print(f"  -> 总交易数: {best.total_trades}")


if __name__ == "__main__":
    main()
