"""ETF行业轮动策略 — 基于动量排名在多个行业ETF之间轮动

已修复前视偏差：信号在T日收盘后生成，交易在T+1日开盘价执行。
ETF免印花税，卖出时仅收取佣金。
"""
import sqlite3
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"

INITIAL_CAPITAL = 100_000
COMMISSION_RATE = 0.0003


@dataclass
class BaseSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


class ETFRotationStrategy:
    """ETF行业轮动策略

    基于N日动量排名在多个行业ETF之间轮动：
    - 计算每个ETF的N日收益率（使用收盘价）
    - 按收益率排名
    - 买入排名前K的ETF（等权分配）
    - 每M个交易日重新平衡
    - 止损：从买入价下跌超过stop_loss比例
    - 止盈：持仓盈利超过take_profit比例时卖出
    - 所有信号在T日生成，T+1日以开盘价执行（消除前视偏差）
    """

    def __init__(
        self,
        lookback: int = 20,
        top_k: int = 3,
        rebalance_days: int = 20,
        stop_loss: float = 0.08,
        take_profit: float = 0.30,
    ):
        self.lookback = lookback
        self.top_k = top_k
        self.rebalance_days = rebalance_days
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def _load_etf_data(self, conn: sqlite3.Connection) -> Dict[str, pd.DataFrame]:
        try:
            df = pd.read_sql(
                "SELECT * FROM etf_daily_quotes ORDER BY etf_code, trade_date",
                conn,
            )
        except Exception:
            return {}

        if df.empty:
            return {}

        result = {}
        for etf_code, group in df.groupby("etf_code"):
            group = group.sort_values("trade_date").reset_index(drop=True)
            group = group.rename(columns={"etf_code": "ts_code"})
            result[etf_code] = group

        return result

    def _compute_momentum(self, etf_data: Dict[str, pd.DataFrame], date: str) -> pd.DataFrame:
        rows = []
        for etf_code, df in etf_data.items():
            mask = df["trade_date"] <= date
            subset = df.loc[mask]

            if len(subset) < self.lookback + 1:
                continue

            close = subset["close"].values
            ret = (close[-1] - close[-self.lookback - 1]) / close[-self.lookback - 1]

            if np.isnan(ret):
                continue

            rows.append({
                "etf_code": etf_code,
                "close": close[-1],
                "momentum": ret,
                "trade_date": date,
            })

        if not rows:
            return pd.DataFrame()

        momentum_df = pd.DataFrame(rows)
        momentum_df = momentum_df.sort_values("momentum", ascending=False).reset_index(drop=True)
        momentum_df["rank"] = range(1, len(momentum_df) + 1)
        return momentum_df

    def _get_open_price(self, etf_data: Dict[str, pd.DataFrame],
                        etf_code: str, date: str) -> Optional[float]:
        df = etf_data.get(etf_code)
        if df is None:
            return None
        mask = df["trade_date"] == date
        if not mask.any():
            return None
        return float(df.loc[mask, "open"].values[0])

    def _get_close_price(self, etf_data: Dict[str, pd.DataFrame],
                         etf_code: str, date: str) -> Optional[float]:
        df = etf_data.get(etf_code)
        if df is None:
            return None
        mask = df["trade_date"] == date
        if not mask.any():
            return None
        return float(df.loc[mask, "close"].values[0])

    def run_rotation_backtest(self, etf_data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
        """运行ETF轮动回测（T日信号，T+1日开盘价执行）

        执行顺序（每个交易日）：
        1. 以开盘价执行前一日挂单（先卖后买）
        2. 以收盘价检查止损/止盈/调仓条件 → 生成新挂单
        3. 新挂单在下一个交易日以开盘价执行
        """
        if not etf_data:
            return self._empty_result()

        all_dates_set: set[str] = set()
        for df in etf_data.values():
            all_dates_set.update(df["trade_date"].tolist())
        all_dates = sorted(all_dates_set)

        if len(all_dates) < self.lookback + self.rebalance_days:
            return self._empty_result()

        capital = float(INITIAL_CAPITAL)
        holdings: Dict[str, Dict[str, object]] = {}
        trades: List[Dict[str, object]] = []
        days_since_rebalance = 0
        portfolio_values: List[float] = []
        portfolio_dates: List[str] = []

        pending_sells: List[Dict[str, object]] = []
        pending_buys: List[Dict[str, object]] = []

        for idx, date in enumerate(all_dates):
            # Step 1: Execute pending orders at today's open price
            for order in pending_sells:
                etf_code = order["etf_code"]
                reason = order["reason"]
                if etf_code not in holdings:
                    continue
                open_price = self._get_open_price(etf_data, etf_code, date)
                if open_price is None:
                    continue
                holding = holdings[etf_code]
                sell_price = open_price

                # 支持部分卖出（再平衡减仓）和全部卖出
                reduce_shares = order.get("reduce_shares")
                if reduce_shares is not None and reduce_shares < holding["shares"]:
                    # 部分卖出：只卖出指定数量
                    shares_to_sell = reduce_shares
                    revenue = shares_to_sell * sell_price * (1 - COMMISSION_RATE)
                    capital += revenue
                    pnl_pct = (sell_price / holding["entry_price"] - 1) * 100
                    trades.append({
                        "etf_code": etf_code,
                        "action": "sell",
                        "date": date,
                        "price": sell_price,
                        "pnl_pct": pnl_pct,
                        "reason": reason,
                    })
                    # 更新持仓数量而非删除
                    holdings[etf_code] = {
                        "shares": holding["shares"] - shares_to_sell,
                        "entry_price": holding["entry_price"],
                        "entry_date": holding["entry_date"],
                    }
                else:
                    # 全部卖出
                    revenue = holding["shares"] * sell_price * (1 - COMMISSION_RATE)
                    capital += revenue
                    pnl_pct = (sell_price / holding["entry_price"] - 1) * 100
                    trades.append({
                        "etf_code": etf_code,
                        "action": "sell",
                        "date": date,
                        "price": sell_price,
                        "pnl_pct": pnl_pct,
                        "reason": reason,
                    })
                    del holdings[etf_code]

            new_buy_orders = [o for o in pending_buys if "add_shares" not in o]
            rebalance_add_orders = [o for o in pending_buys if "add_shares" in o]
            n_new = len(new_buy_orders)

            for order in new_buy_orders:
                etf_code = order["etf_code"]
                reason = order["reason"]
                open_price = self._get_open_price(etf_data, etf_code, date)
                if open_price is None:
                    continue
                alloc = capital / n_new if n_new > 0 else 0
                shares = int(alloc / open_price) // 100 * 100
                if shares >= 100:
                    cost = shares * open_price * (1 + COMMISSION_RATE)
                    if cost <= capital:
                        capital -= cost
                        holdings[etf_code] = {
                            "shares": shares,
                            "entry_price": open_price,
                            "entry_date": date,
                        }
                        trades.append({
                            "etf_code": etf_code,
                            "action": "buy",
                            "date": date,
                            "price": open_price,
                            "pnl_pct": 0.0,
                            "reason": reason,
                        })

            for order in rebalance_add_orders:
                etf_code = order["etf_code"]
                reason = order["reason"]
                add_shares = order.get("add_shares", 0)
                open_price = self._get_open_price(etf_data, etf_code, date)
                if open_price is None or add_shares < 100:
                    continue
                cost = add_shares * open_price * (1 + COMMISSION_RATE)
                if cost <= capital:
                    capital -= cost
                    if etf_code in holdings:
                        h = holdings[etf_code]
                        old_cost = h["shares"] * h["entry_price"]
                        new_total_shares = h["shares"] + add_shares
                        holdings[etf_code] = {
                            "shares": new_total_shares,
                            "entry_price": (old_cost + cost) / new_total_shares,
                            "entry_date": h["entry_date"],
                        }
                    else:
                        holdings[etf_code] = {
                            "shares": add_shares,
                            "entry_price": open_price,
                            "entry_date": date,
                        }
                    trades.append({
                        "etf_code": etf_code,
                        "action": "buy",
                        "date": date,
                        "price": open_price,
                        "pnl_pct": 0.0,
                        "reason": reason,
                    })

            pending_sells = []
            pending_buys = []

            # Step 2: Check stop-loss / take-profit / rebalance using close prices
            prices_close: Dict[str, float] = {}
            for etf_code in etf_data:
                c = self._get_close_price(etf_data, etf_code, date)
                if c is not None:
                    prices_close[etf_code] = c

            for etf_code, holding in list(holdings.items()):
                if etf_code not in prices_close:
                    continue
                current_price = prices_close[etf_code]
                pnl = (current_price - holding["entry_price"]) / holding["entry_price"]

                if pnl <= -self.stop_loss:
                    pending_sells.append({
                        "etf_code": etf_code,
                        "reason": f"止损 {pnl*100:+.1f}%",
                    })
                elif self.take_profit > 0 and pnl >= self.take_profit:
                    pending_sells.append({
                        "etf_code": etf_code,
                        "reason": f"止盈 {pnl*100:+.1f}%",
                    })

            days_since_rebalance += 1

            if days_since_rebalance >= self.rebalance_days:
                momentum_df = self._compute_momentum(etf_data, date)

                if not momentum_df.empty:
                    top_etfs = momentum_df.head(self.top_k)["etf_code"].tolist()

                    pending_sell_codes = {o["etf_code"] for o in pending_sells}

                    for etf_code in list(holdings.keys()):
                        if etf_code in pending_sell_codes:
                            continue
                        if etf_code not in top_etfs:
                            pending_sells.append({
                                "etf_code": etf_code,
                                "reason": "调仓卖出 排名落后",
                            })

                    etf_to_buy = [c for c in top_etfs if c not in holdings]
                    for etf_code in etf_to_buy:
                        rank = momentum_df.loc[momentum_df["etf_code"] == etf_code, "rank"]
                        rank_val = int(rank.values[0]) if len(rank) > 0 else 0
                        pending_buys.append({
                            "etf_code": etf_code,
                            "reason": f"调仓买入 动量排名{rank_val}",
                            "rank": rank_val,
                        })

                    # Rebalance existing holdings in top_k
                    for etf_code in list(holdings.keys()):
                        if etf_code in pending_sell_codes:
                            continue
                        if etf_code not in top_etfs:
                            continue
                        if etf_code not in prices_close:
                            continue

                        total_assets = capital
                        for c, h in holdings.items():
                            if c in prices_close and c not in pending_sell_codes:
                                total_assets += h["shares"] * prices_close[c]

                        target_per_etf = total_assets / self.top_k if self.top_k > 0 else 0
                        current_value = holdings[etf_code]["shares"] * prices_close[etf_code]

                        if target_per_etf > 0 and abs(current_value - target_per_etf) / target_per_etf > 0.2:
                            target_shares = int(target_per_etf / prices_close[etf_code]) // 100 * 100
                            current_shares = holdings[etf_code]["shares"]
                            diff = target_shares - current_shares

                            if diff > 0:
                                pending_buys.append({
                                    "etf_code": etf_code,
                                    "reason": f"再平衡加仓 +{diff}股",
                                    "rank": 0,
                                    "add_shares": diff,
                                })
                            elif diff < 0:
                                pending_sells.append({
                                    "etf_code": etf_code,
                                    "reason": f"再平衡减仓 {diff}股",
                                    "reduce_shares": abs(diff),
                                })

                days_since_rebalance = 0

            # Step 3: Calculate portfolio value at close
            total_value = capital
            for etf_code, holding in holdings.items():
                if etf_code in prices_close:
                    total_value += holding["shares"] * prices_close[etf_code]
                else:
                    total_value += holding["shares"] * holding["entry_price"]
            portfolio_values.append(total_value)
            portfolio_dates.append(date)

        # Liquidate all holdings at last close price
        for etf_code, holding in list(holdings.items()):
            last_price = holding["entry_price"]
            df = etf_data.get(etf_code)
            if df is not None and not df.empty:
                last_price = float(df.iloc[-1]["close"])
            revenue = holding["shares"] * last_price * (1 - COMMISSION_RATE)
            pnl_pct = (last_price / holding["entry_price"] - 1) * 100
            capital += revenue
            trades.append({
                "etf_code": etf_code,
                "action": "sell",
                "date": all_dates[-1] if all_dates else "",
                "price": last_price,
                "pnl_pct": pnl_pct,
                "reason": "期末平仓",
            })

        total_return = (capital / INITIAL_CAPITAL - 1) * 100

        sell_trades = [t for t in trades if t["action"] == "sell" and t["reason"] != "期末平仓"]
        if sell_trades:
            returns = [t["pnl_pct"] for t in sell_trades]
            wins = sum(1 for r in returns if r > 0)
            win_rate = wins / len(returns) * 100
            avg_return = float(np.mean(returns))
            sharpe = float((np.mean(returns) / (np.std(returns) + 0.001)) * np.sqrt(252)) if len(returns) > 1 else 0.0
        else:
            win_rate = 0.0
            avg_return = 0.0
            sharpe = 0.0

        if portfolio_values:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = float(drawdown.min() * 100)
        else:
            max_drawdown = 0.0

        if portfolio_dates and len(portfolio_dates) > 1:
            first_date = portfolio_dates[0]
            last_date = portfolio_dates[-1]
            years = float((pd.Timestamp(last_date) - pd.Timestamp(first_date)).days) / 365.25
            cagr = ((capital / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0.0
        else:
            cagr = 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_trades": len(trades),
            "sell_trades": len(sell_trades),
            "max_drawdown": max_drawdown,
            "trades": trades,
            "final_capital": capital,
            "portfolio_values": portfolio_values,
            "portfolio_dates": portfolio_dates,
        }

    @staticmethod
    def _empty_result() -> Dict[str, object]:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_trades": 0,
            "sell_trades": 0,
            "max_drawdown": 0.0,
            "trades": [],
            "final_capital": INITIAL_CAPITAL,
            "portfolio_values": [],
            "portfolio_dates": [],
        }

    def print_summary(self, result: Dict[str, object]):
        print("=" * 70)
        print("  ETF行业轮动策略 回测结果")
        print(f"  参数: 动量周期={self.lookback}日  Top-K={self.top_k}  "
              f"调仓周期={self.rebalance_days}日  止损={self.stop_loss*100:.0f}%  "
              f"止盈={self.take_profit*100:.0f}%")
        print("  执行: T日信号 → T+1日开盘价成交 (无前视偏差)")
        print("=" * 70)
        print(f"  初始资金:     {INITIAL_CAPITAL:>12,.0f}")
        print(f"  期末资金:     {result['final_capital']:>12,.2f}")
        print(f"  总收益率:     {result['total_return']:>+11.2f}%")
        print(f"  年化收益:     {result['cagr']:>+11.2f}%")
        print(f"  最大回撤:     {result['max_drawdown']:>11.2f}%")
        print(f"  夏普比率:     {result['sharpe']:>11.2f}")
        print(f"  交易次数:     {result['total_trades']:>12d}  (卖出: {result['sell_trades']})")
        print(f"  胜率:         {result['win_rate']:>11.1f}%")
        print(f"  平均收益:     {result['avg_return']:>+11.2f}%")
        print("=" * 70)

        trades = result.get("trades", [])
        if trades:
            print(f"\n  最近20笔交易:")
            print(f"  {'日期':<12} {'ETF':<10} {'动作':<6} {'价格':>10} {'收益':>10} {'原因'}")
            print("  " + "-" * 66)
            for t in trades[-20:]:
                print(f"  {t['date']:<12} {t['etf_code']:<10} {t['action']:<6} "
                      f"{t['price']:>10.3f} {t['pnl_pct']:>+9.1f}% {t['reason']}")


def print_annual_breakdown(result: Dict[str, object]):
    portfolio_dates = result.get("portfolio_dates", [])
    portfolio_values = result.get("portfolio_values", [])

    if not portfolio_dates or not portfolio_values:
        print("  无足够数据生成年度分解表")
        return

    df = pd.DataFrame({
        "date": portfolio_dates,
        "value": portfolio_values,
    })
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())

    print("\n" + "=" * 70)
    print("  年度收益分解")
    print("=" * 70)
    print(f"  {'年份':<6} {'期初值':>12} {'期末值':>12} {'年度收益':>10} {'最大回撤':>10}")
    print("  " + "-" * 52)

    overall_start_value = float(df.iloc[0]["value"])
    overall_end_value = float(df.iloc[-1]["value"])

    for year in years:
        year_data = df[df["year"] == year]
        start_val = float(year_data.iloc[0]["value"])
        end_val = float(year_data.iloc[-1]["value"])

        year_return = (end_val / start_val - 1) * 100

        year_vals = np.array(year_data["value"].tolist(), dtype=float)
        peak = np.maximum.accumulate(year_vals)
        dd = (year_vals - peak) / peak
        year_max_dd = float(dd.min() * 100)

        print(f"  {year:<6} {start_val:>12,.0f} {end_val:>12,.0f} "
              f"{year_return:>+9.1f}% {year_max_dd:>9.1f}%")

    total_return = (overall_end_value / overall_start_value - 1) * 100
    years_count = len(years)
    if years_count > 1:
        years_span = float((pd.Timestamp(portfolio_dates[-1]) - pd.Timestamp(portfolio_dates[0])).days) / 365.25
        cagr = ((overall_end_value / overall_start_value) ** (1 / years_span) - 1) * 100
    else:
        cagr = total_return

    print("  " + "-" * 52)
    print(f"  {'汇总':<6} {overall_start_value:>12,.0f} {overall_end_value:>12,.0f} "
          f"{total_return:>+9.1f}%")
    print(f"\n  年化收益率(CAGR): {cagr:>+.2f}%")
    print("=" * 70)


def grid_search_etf(etf_data: Dict[str, pd.DataFrame]):
    lookbacks = [10, 15, 20, 30, 40, 60]
    top_ks = [2, 3, 4, 5]
    rebalance_days_list = [10, 15, 20, 30, 40]

    total_combos = len(lookbacks) * len(top_ks) * len(rebalance_days_list)
    print(f"\n{'=' * 70}")
    print(f"  参数网格搜索 — 共 {total_combos} 种组合")
    print(f"{'=' * 70}")

    results: List[Dict[str, object]] = []
    combo_count = 0

    for lookback, top_k, rebalance_days in product(lookbacks, top_ks, rebalance_days_list):
        combo_count += 1
        strategy = ETFRotationStrategy(
            lookback=lookback,
            top_k=top_k,
            rebalance_days=rebalance_days,
            stop_loss=0.08,
            take_profit=0.30,
        )
        result = strategy.run_rotation_backtest(etf_data)

        results.append({
            "lookback": lookback,
            "top_k": top_k,
            "rebalance": rebalance_days,
            "total_return": result["total_return"],
            "sharpe": result["sharpe"],
            "max_drawdown": result["max_drawdown"],
            "cagr": result["cagr"],
        })

        if combo_count % 20 == 0 or combo_count == total_combos:
            print(f"  进度: {combo_count}/{total_combos} ({combo_count/total_combos*100:.0f}%)")

    results.sort(key=lambda x: x["total_return"], reverse=True)

    print(f"\n{'=' * 90}")
    print(f"  参数网格搜索结果 (按总收益排序)")
    print(f"{'=' * 90}")
    print(f"  {'排名':>4}  {'lookback':>8}  {'top_k':>5}  {'rebalance':>9}  "
          f"{'总收益':>8}  {'夏普':>6}  {'最大回撤':>8}  {'年化':>8}")
    print("  " + "-" * 82)

    for i, r in enumerate(results):
        marker = " ★" if i == 0 else "  "
        print(f"{marker}{i+1:>3}  {r['lookback']:>8}  {r['top_k']:>5}  {r['rebalance']:>9}  "
              f"{r['total_return']:>+7.1f}%  {r['sharpe']:>6.2f}  "
              f"{r['max_drawdown']:>7.1f}%  {r['cagr']:>+7.1f}%")

    best = results[0]
    print(f"\n  ★ 最优参数组合:")
    print(f"    lookback={best['lookback']}, top_k={best['top_k']}, "
          f"rebalance={best['rebalance']}")
    print(f"    总收益: {best['total_return']:+.1f}%  夏普: {best['sharpe']:.2f}  "
          f"最大回撤: {best['max_drawdown']:.1f}%  年化: {best['cagr']:+.1f}%")
    print("=" * 90)


def main():
    db_path = DB_PATH
    if not db_path.exists():
        print(f"数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))

    strategy = ETFRotationStrategy(
        lookback=20,
        top_k=3,
        rebalance_days=20,
        stop_loss=0.08,
        take_profit=0.30,
    )

    etf_data = strategy._load_etf_data(conn)
    conn.close()

    if not etf_data:
        print("未找到ETF数据 (etf_daily_quotes表为空或不存在)")
        return

    print(f"加载了 {len(etf_data)} 个ETF的数据:")
    for code, df in etf_data.items():
        print(f"  {code}: {len(df)} 条记录 ({df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]})")

    print("\n运行默认策略 (lookback=20, top_k=3, rebalance=20, stop=8%, take_profit=30%)...")
    result = strategy.run_rotation_backtest(etf_data)
    strategy.print_summary(result)
    print_annual_breakdown(result)

    if "--grid" in sys.argv:
        grid_search_etf(etf_data)
    else:
        print("\n  提示: 使用 --grid 参数运行参数网格搜索")
        print("  例: python strategies/etf_rotation_strategy.py --grid")


if __name__ == "__main__":
    main()
