"""用真实数据跑双均线策略回测"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.storage import DataStorage
from strategies.ma_strategy import MAStrategy
from backtest.simple_backtest import SimpleBacktest


# 选几只代表性股票回测
STOCKS = [
    ("600519.SH", "贵州茅台"),
    ("000858.SZ", "五粮液"),   # 可能数据不全
    ("600036.SH", "招商银行"),
    ("002594.SZ", "比亚迪"),
    ("601318.SH", "中国平安"),
    ("000333.SZ", "美的集团"),
    ("300750.SZ", "宁德时代"),
    ("600276.SH", "恒瑞医药"),
]


def run_single_backtest(storage, ts_code, name, fast=5, slow=20):
    """跑单只股票回测"""
    df = storage.get_daily_quotes(ts_code)

    if df.empty:
        print(f"  ⚠ {ts_code} ({name}): 无数据")
        return None

    if len(df) < slow + 10:
        print(f"  ⚠ {ts_code} ({name}): 数据不足 ({len(df)}条)")
        return None

    strategy = MAStrategy(fast_period=fast, slow_period=slow)
    backtest = SimpleBacktest(initial_capital=100000)
    result = backtest.run(df, strategy)

    return {
        "ts_code": ts_code,
        "name": name,
        "data_range": f"{df['trade_date'].min()} ~ {df['trade_date'].max()}",
        "data_count": len(df),
        "total_trades": result.total_trades,
        "total_profit": result.total_profit,
        "total_profit_pct": result.total_profit_pct,
        "win_rate": result.win_rate,
        "final_equity": result.final_equity,
        "trades": result.trades,
    }


def main():
    print("=" * 80)
    print("  中证100成分股 · 双均线交叉策略回测")
    print("  策略: MA5/MA20 | 初始资金: 10万 | 固定仓位: 1000股")
    print("=" * 80)

    storage = DataStorage()

    results = []
    for ts_code, name in STOCKS:
        print(f"\n{'─' * 60}")
        print(f"  回测: {ts_code} ({name})")
        r = run_single_backtest(storage, ts_code, name)
        if r:
            results.append(r)
            print(f"  数据: {r['data_count']}条 ({r['data_range']})")
            print(f"  交易: {r['total_trades']}次 | 胜率: {r['win_rate']:.1f}%")
            print(f"  收益: {r['total_profit']:+.2f} ({r['total_profit_pct']:+.2f}%)")
            print(f"  终值: {r['final_equity']:.2f}")
        print(f"{'─' * 60}")

    # 汇总
    print("\n" + "=" * 80)
    print("  回测汇总")
    print("=" * 80)
    print(f"  {'股票':<12} {'名称':<10} {'交易数':<8} {'胜率':<8} {'收益率':<10} {'最终资金':<12}")
    print("  " + "-" * 60)

    for r in results:
        print(
            f"  {r['ts_code']:<12} {r['name']:<10} "
            f"{r['total_trades']:<8} {r['win_rate']:<7.1f}% "
            f"{r['total_profit_pct']:>+8.2f}% {r['final_equity']:<12.2f}"
        )

    if results:
        avg_profit_pct = sum(r['total_profit_pct'] for r in results) / len(results)
        avg_win_rate = sum(r['win_rate'] for r in results) / len(results)
        print("  " + "-" * 60)
        print(f"  平均收益率: {avg_profit_pct:+.2f}%  |  平均胜率: {avg_win_rate:.1f}%")

    print("=" * 80)

    # 打印最赚钱和最亏钱的股票的交易明细
    if results:
        best = max(results, key=lambda r: r['total_profit_pct'])
        worst = min(results, key=lambda r: r['total_profit_pct'])

        for label, r in [("最佳", best), ("最差", worst)]:
            print(f"\n  {label}交易明细: {r['ts_code']} ({r['name']}) - 收益 {r['total_profit_pct']:+.2f}%")
            if r['trades']:
                print(f"  {'序号':<5} {'买入日期':<12} {'卖出日期':<12} {'买价':<10} {'卖价':<10} {'收益':<10} {'收益率':<8}")
                print("  " + "-" * 70)
                for i, t in enumerate(r['trades'], 1):
                    print(
                        f"  {i:<5} {t.entry_date:<12} {t.exit_date:<12} "
                        f"{t.entry_price:<10.2f} {t.exit_price:<10.2f} "
                        f"{t.profit:>+9.2f} {t.profit_pct:>+7.2f}%"
                    )
            else:
                print("  无交易记录")


if __name__ == "__main__":
    main()
