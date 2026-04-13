"""Walk-Forward 验证框架 — 对任何策略进行滚动窗口验证"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


@dataclass
class BaseSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


def run_backtest(df: pd.DataFrame, strategy) -> Optional[dict]:
    """对单个股票运行回测（复制自 compare_strategies.py 的逻辑）"""
    capital = INITIAL_CAPITAL
    trades = []
    position = None
    pending_buy = None
    pending_sell_reason = None

    signals = strategy.calculate_signals(df)
    sig_map: Dict[str, list] = {}
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
                    position = {
                        "entry_date": date,
                        "entry_price": exec_price,
                        "shares": shares,
                    }
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
    sharpe = (np.mean(returns) / (np.std(returns) + 0.001)) * np.sqrt(252) if len(returns) > 1 else 0.0

    return {
        "total_trades": len(trades),
        "win_rate": wins / len(trades) * 100,
        "avg_return": np.mean(returns),
        "return_pct": (capital / INITIAL_CAPITAL - 1) * 100,
        "sharpe": sharpe,
        "profitable": 1 if capital > INITIAL_CAPITAL else 0,
    }


def run_single_window(
    df: pd.DataFrame,
    strategy,
    start_idx: int,
    end_idx: int,
) -> Optional[dict]:
    """对单个时间窗口运行策略回测"""
    window_df = df.iloc[start_idx:end_idx].copy()

    if len(window_df) < 50:
        return None

    return run_backtest(window_df, strategy)


def walk_forward_validate(
    df: pd.DataFrame,
    strategy,
    train_window: int = 500,
    test_window: int = 100,
    step: int = 100,
) -> dict:
    """Walk-Forward 验证

    Args:
        df: 单只股票的日线数据
        strategy: 策略实例（需实现 calculate_signals）
        train_window: 训练窗口大小（交易日数）
        test_window: 测试窗口大小（交易日数）
        step: 滑动步长（交易日数）

    Returns:
        dict with per-window IS/OOS metrics and aggregate stats
    """
    n = len(df)
    if n < train_window + test_window:
        return {"windows": [], "aggregate": None}

    windows = []
    start = 0

    while start + train_window + test_window <= n:
        train_start = start
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window

        train_result = run_single_window(df, strategy, train_start, train_end)
        test_result = run_single_window(df, strategy, test_start, test_end)

        window_info = {
            "window_id": len(windows) + 1,
            "train_start": df.iloc[train_start]["trade_date"],
            "train_end": df.iloc[train_end - 1]["trade_date"],
            "test_start": df.iloc[test_start]["trade_date"],
            "test_end": df.iloc[min(test_end, n) - 1]["trade_date"],
            "is_result": train_result,
            "oos_result": test_result,
        }
        windows.append(window_info)
        start += step

    # 计算汇总指标
    is_results = [w["is_result"] for w in windows if w["is_result"] is not None]
    oos_results = [w["oos_result"] for w in windows if w["oos_result"] is not None]

    aggregate = _compute_aggregate(is_results, oos_results)

    return {"windows": windows, "aggregate": aggregate}


def _compute_aggregate(is_results: List[dict], oos_results: List[dict]) -> dict:
    """计算汇总统计"""
    if not is_results and not oos_results:
        return {
            "is_avg_return": 0.0,
            "is_avg_sharpe": 0.0,
            "is_win_rate": 0.0,
            "oos_avg_return": 0.0,
            "oos_avg_sharpe": 0.0,
            "oos_win_rate": 0.0,
            "overfitting_ratio": 0.0,
            "is_windows": 0,
            "oos_windows": 0,
        }

    is_avg_return = np.mean([r["return_pct"] for r in is_results]) if is_results else 0.0
    is_avg_sharpe = np.mean([r["sharpe"] for r in is_results]) if is_results else 0.0
    is_win_rate = np.mean([r["win_rate"] for r in is_results]) if is_results else 0.0

    oos_avg_return = np.mean([r["return_pct"] for r in oos_results]) if oos_results else 0.0
    oos_avg_sharpe = np.mean([r["sharpe"] for r in oos_results]) if oos_results else 0.0
    oos_win_rate = np.mean([r["win_rate"] for r in oos_results]) if oos_results else 0.0

    # 过拟合比率：IS性能 / OOS性能
    # 使用绝对值来避免符号问题
    if abs(oos_avg_return) > 0.01:
        overfitting_ratio = abs(is_avg_return / oos_avg_return)
    else:
        overfitting_ratio = float("inf") if abs(is_avg_return) > 0.01 else 1.0

    return {
        "is_avg_return": is_avg_return,
        "is_avg_sharpe": is_avg_sharpe,
        "is_win_rate": is_win_rate,
        "oos_avg_return": oos_avg_return,
        "oos_avg_sharpe": oos_avg_sharpe,
        "oos_win_rate": oos_win_rate,
        "overfitting_ratio": overfitting_ratio,
        "is_windows": len(is_results),
        "oos_windows": len(oos_results),
    }


def print_walk_forward_report(strategy_name: str, wf_result: dict):
    """打印Walk-Forward验证报告"""
    windows = wf_result["windows"]
    aggregate = wf_result["aggregate"]

    if not windows:
        print(f"\n  {strategy_name}: 无有效窗口（数据不足）")
        return

    print(f"\n  策略: {strategy_name}")
    print(f"  {'窗口':>4}  {'训练期':<22}  {'测试期':<22}  {'IS收益':>10}  {'OOS收益':>10}  {'IS夏普':>8}  {'OOS夏普':>8}")
    print("  " + "-" * 100)

    for w in windows:
        is_r = w["is_result"]
        oos_r = w["oos_result"]

        is_ret = f"{is_r['return_pct']:+.2f}%" if is_r else "N/A"
        oos_ret = f"{oos_r['return_pct']:+.2f}%" if oos_r else "N/A"
        is_sh = f"{is_r['sharpe']:.2f}" if is_r else "N/A"
        oos_sh = f"{oos_r['sharpe']:.2f}" if oos_r else "N/A"

        print(f"  {w['window_id']:>4}  {w['train_start']}~{w['train_end']:<11}  "
              f"{w['test_start']}~{w['test_end']:<11}  {is_ret:>10}  {oos_ret:>10}  {is_sh:>8}  {oos_sh:>8}")

    if aggregate:
        print("  " + "-" * 100)
        ratio_str = f"{aggregate['overfitting_ratio']:.2f}" if aggregate["overfitting_ratio"] != float("inf") else "∞"
        print(f"  汇总  IS平均收益: {aggregate['is_avg_return']:+.2f}%  "
              f"OOS平均收益: {aggregate['oos_avg_return']:+.2f}%  "
              f"过拟合比: {ratio_str}")
        print(f"       IS平均夏普: {aggregate['is_avg_sharpe']:.2f}  "
              f"OOS平均夏普: {aggregate['oos_avg_sharpe']:.2f}  "
              f"IS胜率: {aggregate['is_win_rate']:.1f}%  OOS胜率: {aggregate['oos_win_rate']:.1f}%")


def main():
    storage = DataStorage()
    import sqlite3
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"))
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
    ).fetchall()]
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

    train_window = 500
    test_window = 100
    step = 100

    print("=" * 110)
    print("  Walk-Forward 验证报告")
    print(f"  训练窗口: {train_window}日  测试窗口: {test_window}日  步长: {step}日")
    print(f"  股票池: {len(codes)}只  策略数: {len(strategies)}")
    print("=" * 110)

    # 用前几只股票做演示
    sample_codes = codes[:5]

    summary_table = []

    for name, strategy in strategies.items():
        all_is_returns = []
        all_oos_returns = []
        all_is_sharpes = []
        all_oos_sharpes = []

        for code in sample_codes:
            df = storage.get_daily_quotes(code)
            if df.empty or len(df) < train_window + test_window:
                continue

            wf = walk_forward_validate(df, strategy, train_window, test_window, step)
            agg = wf.get("aggregate")

            if agg and agg.get("is_windows", 0) > 0:
                all_is_returns.append(agg["is_avg_return"])
                all_oos_returns.append(agg["oos_avg_return"])
                all_is_sharpes.append(agg["is_avg_sharpe"])
                all_oos_sharpes.append(agg["oos_avg_sharpe"])

        if all_is_returns:
            avg_is = np.mean(all_is_returns)
            avg_oos = np.mean(all_oos_returns)
            avg_is_sh = np.mean(all_is_sharpes)
            avg_oos_sh = np.mean(all_oos_sharpes)
            ratio = abs(avg_is / avg_oos) if abs(avg_oos) > 0.01 else float("inf")

            summary_table.append({
                "name": name,
                "stocks": len(all_is_returns),
                "is_return": avg_is,
                "oos_return": avg_oos,
                "is_sharpe": avg_is_sh,
                "oos_sharpe": avg_oos_sh,
                "overfitting_ratio": ratio,
            })

    if summary_table:
        print(f"\n{'策略':<22} {'股票':>4} {'IS平均收益':>12} {'OOS平均收益':>12} {'IS夏普':>8} {'OOS夏普':>8} {'过拟合比':>8}")
        print("-" * 110)
        for s in summary_table:
            ratio_str = f"{s['overfitting_ratio']:.2f}" if s["overfitting_ratio"] != float("inf") else "∞"
            print(f"{s['name']:<22} {s['stocks']:>4} {s['is_return']:>+11.2f}% {s['oos_return']:>+11.2f}% "
                  f"{s['is_sharpe']:>8.2f} {s['oos_sharpe']:>8.2f} {ratio_str:>8}")

        print("-" * 110)
        print("\n过拟合比说明: IS收益/OOS收益的绝对值比，越大表示过拟合越严重，接近1.0最理想")

    # 对第一只股票打印详细报告
    if sample_codes:
        first_code = sample_codes[0]
        df = storage.get_daily_quotes(first_code)
        if not df.empty and len(df) >= train_window + test_window:
            print(f"\n{'=' * 110}")
            print(f"  详细报告 — {first_code}")
            print("=" * 110)

            for name, strategy in strategies.items():
                wf = walk_forward_validate(df, strategy, train_window, test_window, step)
                print_walk_forward_report(name, wf)

    print("\n" + "=" * 110)
    print("  Walk-Forward 验证完成")
    print("=" * 110)


if __name__ == "__main__":
    main()
