"""Walk-Forward 验证 — ETF行业轮动策略

滚动窗口验证框架：
- 训练窗口: 500交易日 (~2年)
- 测试窗口: 100交易日 (~5个月)
- 步长: 100交易日

每个窗口：
1. 在训练窗口上运行简化网格搜索，找最优参数
2. 将最优参数应用于测试窗口
3. 计算过拟合比率 = IS平均收益 / OOS平均收益

过拟合判断标准：
- Ratio < 1  → OOS优于IS (好)
- Ratio 1~5  → 轻度过拟合 (可接受)
- Ratio > 5  → 严重过拟合 (不通过)

通过条件: overfitting_ratio < 5 AND OOS收益 > 0

Usage:
    python scripts/walk_forward_etf.py
"""
import sqlite3
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"

sys.path.insert(0, str(PROJECT_ROOT))

from strategies.etf_rotation_strategy import ETFRotationStrategy, INITIAL_CAPITAL

# ── Walk-Forward 配置 ─────────────────────────────────────────────
TRAIN_WINDOW = 500     # 训练窗口 (交易日, ~2年)
TEST_WINDOW = 100      # 测试窗口 (交易日, ~5个月)
STEP = 100             # 滚动步长 (交易日)

# 网格搜索空间 (共6种组合)
PARAM_GRID = list(product(
    [10, 20, 40],   # lookback
    [3],             # top_k
    [20, 40],        # rebalance_days
))

# 固定风控参数 (与策略默认配置一致)
FIXED_RISK = {
    "stop_loss": 0.08,
    "take_profit": 0.30,
}

# 静态参数基准 (最优历史参数)
STATIC_PARAMS = {"lookback": 20, "top_k": 3, "rebalance": 20}

# 过拟合阈值
OVERFITTING_THRESHOLD = 5.0


# ── 数据加载 ──────────────────────────────────────────────────────

def load_etf_data(conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
    """从数据库加载所有ETF日线数据"""
    df = pd.read_sql(
        "SELECT * FROM etf_daily_quotes ORDER BY etf_code, trade_date",
        conn,
    )
    if df.empty:
        return {}
    result = {}
    for etf_code, group in df.groupby("etf_code"):
        group = group.sort_values("trade_date").reset_index(drop=True)
        group = group.rename(columns={"etf_code": "ts_code"})
        result[etf_code] = group
    return result


def get_all_trading_dates(etf_data: Dict[str, pd.DataFrame]) -> List[str]:
    """获取所有ETF的合并交易日列表"""
    dates = set()
    for df in etf_data.values():
        dates.update(df["trade_date"].tolist())
    return sorted(dates)


def slice_etf_data(
    etf_data: Dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    min_rows: int = 50,
) -> Dict[str, pd.DataFrame]:
    """按日期范围截取ETF数据，过滤掉数据不足的ETF"""
    sliced = {}
    for code, df in etf_data.items():
        mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
        sub = df.loc[mask].reset_index(drop=True)
        if len(sub) >= min_rows:
            sliced[code] = sub
    return sliced


# ── 回测与网格搜索 ────────────────────────────────────────────────

def run_backtest_with_params(
    etf_data: Dict[str, pd.DataFrame],
    lookback: int,
    top_k: int,
    rebalance: int,
) -> dict:
    """用指定参数运行ETF轮动回测，返回关键指标"""
    strategy = ETFRotationStrategy(
        lookback=lookback,
        top_k=top_k,
        rebalance_days=rebalance,
        **FIXED_RISK,
    )
    result = strategy.run_rotation_backtest(etf_data)
    return {
        "total_return": result["total_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "cagr": result["cagr"],
        "win_rate": result["win_rate"],
        "total_trades": result["total_trades"],
    }


def grid_search_train(
    train_data: Dict[str, pd.DataFrame],
) -> Tuple[dict, dict]:
    """在训练数据上运行网格搜索，返回(最优参数, 最优指标)

    选择标准: 夏普比率最高
    """
    best_sharpe = -999.0
    best_params = None
    best_metrics = None

    for lookback, top_k, rebalance in PARAM_GRID:
        metrics = run_backtest_with_params(train_data, lookback, top_k, rebalance)
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_params = {
                "lookback": lookback,
                "top_k": top_k,
                "rebalance": rebalance,
            }
            best_metrics = metrics

    return best_params, best_metrics


# ── Walk-Forward 主逻辑 ───────────────────────────────────────────

def walk_forward_validate(etf_data: Dict[str, pd.DataFrame]) -> dict:
    """Walk-Forward 验证主函数

    Returns:
        {
            "windows": [per-window results],
            "aggregate": {summary statistics},
        }
    """
    all_dates = get_all_trading_dates(etf_data)
    n_dates = len(all_dates)

    print(f"\n  总交易日: {n_dates}  ({all_dates[0]} ~ {all_dates[-1]})")
    print(f"  ETF数量: {len(etf_data)}")
    print(f"  训练窗口: {TRAIN_WINDOW}日  测试窗口: {TEST_WINDOW}日  步长: {STEP}日")
    print(f"  网格搜索: {len(PARAM_GRID)} 种参数组合")

    if n_dates < TRAIN_WINDOW + TEST_WINDOW:
        print("  ✗ 数据不足以进行Walk-Forward验证")
        return {"windows": [], "aggregate": None}

    # 生成所有窗口
    window_specs = []
    start = 0
    while start + TRAIN_WINDOW + TEST_WINDOW <= n_dates:
        train_start = all_dates[start]
        train_end = all_dates[start + TRAIN_WINDOW - 1]
        test_start = all_dates[start + TRAIN_WINDOW]
        test_end_idx = min(start + TRAIN_WINDOW + TEST_WINDOW - 1, n_dates - 1)
        test_end = all_dates[test_end_idx]

        window_specs.append({
            "window_id": len(window_specs) + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        start += STEP

    print(f"  总窗口数: {len(window_specs)}")

    # 逐窗口运行
    window_results = []
    t_total = time.time()

    for spec in window_specs:
        wid = spec["window_id"]
        ts, te = spec["train_start"], spec["train_end"]
        tst_s, tst_e = spec["test_start"], spec["test_end"]

        # 截取数据
        train_data = slice_etf_data(etf_data, ts, te, min_rows=60)
        test_data = slice_etf_data(etf_data, tst_s, tst_e, min_rows=40)

        # 跳过数据不足的窗口
        if not train_data or not test_data:
            print(f"  窗口 {wid:>2}: 跳过 (数据不足 train={len(train_data)} test={len(test_data)})")
            continue

        test_dates = get_all_trading_dates(test_data)
        if len(test_dates) < 50:
            print(f"  窗口 {wid:>2}: 跳过 (测试交易日不足: {len(test_dates)})")
            continue

        t0 = time.time()

        # ── Step 1: 训练 — 网格搜索 ──
        best_params, train_metrics = grid_search_train(train_data)
        is_return = train_metrics["total_return"]

        # ── Step 2: 测试 — 用最优参数 ──
        test_metrics = run_backtest_with_params(
            test_data,
            best_params["lookback"],
            best_params["top_k"],
            best_params["rebalance"],
        )
        oos_return = test_metrics["total_return"]

        # ── Step 3: 静态基准 — 用固定参数 ──
        static_metrics = run_backtest_with_params(
            test_data,
            STATIC_PARAMS["lookback"],
            STATIC_PARAMS["top_k"],
            STATIC_PARAMS["rebalance"],
        )

        elapsed = time.time() - t0

        # ── 过拟合比率 ──
        if abs(oos_return) > 0.01:
            overfitting_ratio = abs(is_return / oos_return)
        else:
            overfitting_ratio = float("inf") if abs(is_return) > 0.01 else 1.0

        passed = overfitting_ratio < OVERFITTING_THRESHOLD and oos_return > 0
        status_mark = "✓ PASS" if passed else "✗ FAIL"
        ratio_str = f"{overfitting_ratio:.2f}" if overfitting_ratio != float("inf") else "∞"

        print(f"  窗口 {wid:>2}: 训练[{ts}~{te}] 测试[{tst_s}~{tst_e}] "
              f"| IS={is_return:+.1f}% OOS={oos_return:+.1f}% "
              f"| 比率={ratio_str} 参数=L{best_params['lookback']}R{best_params['rebalance']} "
              f"| {status_mark} ({elapsed:.1f}s)")

        window_results.append({
            "window_id": wid,
            "train_period": f"{ts}~{te}",
            "test_period": f"{tst_s}~{tst_e}",
            # 最优参数
            "lookback": best_params["lookback"],
            "rebalance": best_params["rebalance"],
            # 训练(IS)指标
            "is_return": is_return,
            "is_sharpe": train_metrics["sharpe"],
            "is_max_dd": train_metrics["max_drawdown"],
            # 测试(OOS)指标 — Walk-Forward
            "oos_return": oos_return,
            "oos_sharpe": test_metrics["sharpe"],
            "oos_max_dd": test_metrics["max_drawdown"],
            # 测试(OOS)指标 — 静态参数
            "static_return": static_metrics["total_return"],
            "static_sharpe": static_metrics["sharpe"],
            # 过拟合
            "overfitting_ratio": overfitting_ratio,
            "passed": passed,
        })

    elapsed_total = time.time() - t_total
    print(f"\n  总耗时: {elapsed_total:.1f}秒")

    aggregate = compute_aggregate(window_results)
    return {"windows": window_results, "aggregate": aggregate}


def compute_aggregate(windows: List[dict]) -> Optional[dict]:
    """计算汇总统计"""
    if not windows:
        return None

    n = len(windows)
    is_returns = [w["is_return"] for w in windows]
    oos_returns = [w["oos_return"] for w in windows]
    static_returns = [w["static_return"] for w in windows]

    avg_is = float(np.mean(is_returns))
    avg_oos = float(np.mean(oos_returns))
    avg_static = float(np.mean(static_returns))

    # 全局过拟合比率 (IS平均 / OOS平均)
    if abs(avg_oos) > 0.01:
        global_ratio = abs(avg_is / avg_oos)
    else:
        global_ratio = float("inf") if abs(avg_is) > 0.01 else 1.0

    # 窗口平均过拟合比 (排除inf)
    finite_ratios = [w["overfitting_ratio"] for w in windows
                     if w["overfitting_ratio"] != float("inf")]
    avg_ratio = float(np.mean(finite_ratios)) if finite_ratios else float("inf")

    # 通过窗口
    passed_count = sum(1 for w in windows if w["passed"])
    overall_pass = global_ratio < OVERFITTING_THRESHOLD and avg_oos > 0

    # 自适应 vs 静态
    wf_better = sum(1 for w in windows if w["oos_return"] > w["static_return"])

    return {
        "avg_is_return": avg_is,
        "avg_oos_return": avg_oos,
        "avg_static_return": avg_static,
        "global_overfitting_ratio": global_ratio,
        "avg_overfitting_ratio": avg_ratio,
        "total_windows": n,
        "passed_windows": passed_count,
        "pass_rate": passed_count / n * 100,
        "overall_pass": overall_pass,
        "wf_better_than_static": wf_better,
        "wf_better_pct": wf_better / n * 100,
    }


# ── 报告输出 ──────────────────────────────────────────────────────

def print_per_window_table(windows: List[dict]):
    """打印逐窗口明细表"""
    if not windows:
        return

    print(f"\n{'=' * 130}")
    print("  逐窗口明细")
    print(f"{'=' * 130}")
    print(f"  {'#':>2}  {'训练期':<24}  {'测试期':<24}  "
          f"{'IS收益':>8}  {'OOS收益':>8}  {'过拟合比':>7}  "
          f"{'参数':>6}  {'状态':>6}")
    print("  " + "-" * 126)

    for w in windows:
        ratio_str = f"{w['overfitting_ratio']:.2f}" if w["overfitting_ratio"] != float("inf") else "∞"
        params_str = f"L{w['lookback']}R{w['rebalance']}"
        status = "✓" if w["passed"] else "✗"

        print(f"  {w['window_id']:>2}  {w['train_period']:<24}  {w['test_period']:<24}  "
              f"{w['is_return']:>+7.1f}%  {w['oos_return']:>+7.1f}%  {ratio_str:>7}  "
              f"{params_str:>6}  {status:>6}")

    print(f"{'=' * 130}")


def print_aggregate_report(aggregate: dict):
    """打印汇总报告"""
    if not aggregate:
        return

    print(f"\n{'=' * 80}")
    print("  汇总统计")
    print(f"{'=' * 80}")

    ratio_str = (f"{aggregate['global_overfitting_ratio']:.2f}"
                 if aggregate["global_overfitting_ratio"] != float("inf") else "∞")
    avg_ratio_str = (f"{aggregate['avg_overfitting_ratio']:.2f}"
                     if aggregate["avg_overfitting_ratio"] != float("inf") else "∞")

    print(f"  窗口总数:         {aggregate['total_windows']}")
    print(f"  通过窗口:         {aggregate['passed_windows']}/{aggregate['total_windows']}  "
          f"({aggregate['pass_rate']:.1f}%)")
    print(f"  ────────────────────────────────")
    print(f"  IS平均收益:       {aggregate['avg_is_return']:>+8.2f}%")
    print(f"  OOS平均收益:      {aggregate['avg_oos_return']:>+8.2f}%")
    print(f"  静态参数OOS收益:  {aggregate['avg_static_return']:>+8.2f}%")
    print(f"  ────────────────────────────────")
    print(f"  全局过拟合比:     {ratio_str}  (IS平均/OOS平均)")
    print(f"  窗口平均过拟合比: {avg_ratio_str}")
    print(f"  ────────────────────────────────")
    print(f"  自适应>静态参数:  {aggregate['wf_better_than_static']}/{aggregate['total_windows']}  "
          f"({aggregate['wf_better_pct']:.1f}%)")

    # 最终判定
    print(f"\n  {'━' * 76}")
    overall = "✓ 通过" if aggregate["overall_pass"] else "✗ 不通过"
    print(f"  最终判定: {overall}")
    print(f"  条件: 过拟合比 < {OVERFITTING_THRESHOLD:.0f} AND OOS收益 > 0")
    print(f"{'=' * 80}")


def print_param_consistency(windows: List[dict]):
    """打印参数一致性分析"""
    if not windows:
        return

    print(f"\n{'=' * 60}")
    print("  参数一致性分析")
    print(f"{'=' * 60}")

    param_counts = {}
    for w in windows:
        key = f"lookback={w['lookback']}, rebalance={w['rebalance']}"
        param_counts[key] = param_counts.get(key, 0) + 1

    n = len(windows)
    for key, count in sorted(param_counts.items(), key=lambda x: -x[1]):
        pct = count / n * 100
        bar = "█" * int(pct / 5)
        print(f"  {key:<30}  {count:>2}次 ({pct:>5.1f}%) {bar}")

    print(f"{'=' * 60}")


def compare_with_single_stock():
    """与单股策略的Walk-Forward结果对比"""
    print(f"\n{'=' * 100}")
    print("  与单股策略Walk-Forward对比 (参考)")
    print(f"{'=' * 100}")

    try:
        from data.storage import DataStorage

        conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"))
        codes = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code LIMIT 5"
        ).fetchall()]
        conn.close()

        if not codes:
            print("  未找到单股数据，跳过对比")
            return

        storage = DataStorage()
        df = storage.get_daily_quotes(codes[0])
        if df.empty or len(df) < TRAIN_WINDOW + TEST_WINDOW:
            print(f"  {codes[0]}数据不足({len(df)}条)，跳过对比")
            return

        from strategies.enhanced_chip_strategy import EnhancedChipStrategy
        from strategies.benchmark_strategies import DualMAStrategy, RSIStrategy
        from scripts.walk_forward_validation import run_backtest

        strategies = {
            "增强筹码": EnhancedChipStrategy(),
            "双均线(5/20)": DualMAStrategy(),
            "RSI(14)": RSIStrategy(),
        }

        print(f"\n  对比股票: {codes[0]}  数据量: {len(df)}条")
        print(f"  {'策略':<16} {'股票':>4} {'IS平均收益':>12} {'OOS平均收益':>12} {'过拟合比':>10} {'通过':>6}")
        print("  " + "-" * 64)

        for name, strategy in strategies.items():
            is_returns = []
            oos_returns = []
            stock_count = 0

            # 对前5只股票取平均
            for code in codes[:5]:
                stock_df = storage.get_daily_quotes(code)
                if stock_df.empty or len(stock_df) < TRAIN_WINDOW + TEST_WINDOW:
                    continue
                stock_count += 1

                n = len(stock_df)
                start = 0
                while start + TRAIN_WINDOW + TEST_WINDOW <= n:
                    train_df = stock_df.iloc[start:start + TRAIN_WINDOW]
                    test_df = stock_df.iloc[start + TRAIN_WINDOW:start + TRAIN_WINDOW + TEST_WINDOW]

                    is_r = run_backtest(train_df, strategy)
                    oos_r = run_backtest(test_df, strategy)

                    if is_r and oos_r:
                        is_returns.append(is_r["return_pct"])
                        oos_returns.append(oos_r["return_pct"])

                    start += STEP

            if is_returns:
                avg_is = float(np.mean(is_returns))
                avg_oos = float(np.mean(oos_returns))
                if abs(avg_oos) > 0.01:
                    ratio = abs(avg_is / avg_oos)
                else:
                    ratio = float("inf") if abs(avg_is) > 0.01 else 1.0
                passed = ratio < OVERFITTING_THRESHOLD and avg_oos > 0

                ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "∞"
                status = "✓" if passed else "✗"
                print(f"  {name:<16} {stock_count:>4} {avg_is:>+11.2f}% {avg_oos:>+11.2f}% "
                      f"{ratio_str:>10} {status:>6}")

    except ImportError as e:
        print(f"  策略模块导入失败: {e}，跳过对比")
    except Exception as e:
        print(f"  对比失败: {e}")

    print(f"{'=' * 100}")


# ── 主入口 ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  ETF行业轮动策略 — Walk-Forward 验证")
    print("=" * 80)

    if not DB_PATH.exists():
        print(f"数据库不存在: {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    etf_data = load_etf_data(conn)
    conn.close()

    if not etf_data:
        print("未找到ETF数据")
        return

    print(f"\n  加载了 {len(etf_data)} 个ETF:")
    for code, df in etf_data.items():
        print(f"    {code}: {len(df)}条 ({df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]})")

    # 运行Walk-Forward验证
    wf_result = walk_forward_validate(etf_data)

    # 打印报告
    print_per_window_table(wf_result["windows"])
    print_aggregate_report(wf_result["aggregate"])
    print_param_consistency(wf_result["windows"])

    # 与单股策略对比
    compare_with_single_stock()

    # 最终总结
    aggregate = wf_result.get("aggregate")
    if aggregate:
        print(f"\n{'=' * 80}")
        print("  最终结论")
        print(f"{'=' * 80}")

        ratio_str = (f"{aggregate['global_overfitting_ratio']:.2f}"
                     if aggregate["global_overfitting_ratio"] != float("inf") else "∞")

        print(f"  1. 通过窗口数:  {aggregate['passed_windows']}/{aggregate['total_windows']} "
              f"({aggregate['pass_rate']:.1f}%)")
        print(f"  2. 平均过拟合比: {ratio_str}")
        print(f"  3. ETF策略Walk-Forward: "
              f"{'✓ 通过' if aggregate['overall_pass'] else '✗ 不通过'}")
        print(f"     - IS平均收益:  {aggregate['avg_is_return']:+.2f}%")
        print(f"     - OOS平均收益: {aggregate['avg_oos_return']:+.2f}%")

        if aggregate["overall_pass"]:
            print(f"\n  📊 ETF轮动策略通过了Walk-Forward验证，过拟合风险可控。")
        else:
            if aggregate["avg_oos_return"] <= 0:
                print(f"\n  ⚠️  OOS平均收益为负，策略在样本外表现不佳。")
            else:
                print(f"\n  ⚠️  过拟合比率过高({ratio_str})，存在过拟合风险。")

        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
