"""多因子选股策略 — 基于市值+质量+动量三因子选股"""
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"

INITIAL_CAPITAL = 100_000
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005
BOARD_LOT = 100


@dataclass
class BaseSignal:
    ts_code: str
    trade_date: str
    action: str
    price: float
    reason: str


class MultifactorStrategy:
    """多因子选股策略

    三因子模型：
    - 因子1：市值因子（越小越好，小市值溢价）
    - 因子2：质量因子（ROE排名，越高越好）
    - 因子3：动量因子（20日收益率排名，越高越好）
    - 综合得分 = 等权平均排名
    - 买入排名前N的股票
    - 每M个交易日调仓
    - 掉出前2N名时卖出
    - 止损-10%
    """

    def __init__(
        self,
        top_n: int = 10,
        rebalance_days: int = 20,
        momentum_days: int = 20,
        stop_loss: float = 0.10,
        w_cap: float = 1.0 / 3,
        w_quality: float = 1.0 / 3,
        w_momentum: float = 1.0 / 3,
    ):
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.momentum_days = momentum_days
        self.stop_loss = stop_loss
        self.w_cap = w_cap
        self.w_quality = w_quality
        self.w_momentum = w_momentum

    def _load_data(self, conn: sqlite3.Connection) -> dict:
        """从数据库加载所有因子数据"""
        data = {}

        try:
            data["daily"] = pd.read_sql(
                "SELECT ts_code, trade_date, open, close, high, low FROM daily_quotes ORDER BY ts_code, trade_date",
                conn,
            )
        except Exception:
            data["daily"] = pd.DataFrame()

        try:
            data["valuation"] = pd.read_sql(
                "SELECT ts_code, trade_date, total_market_cap, circulating_market_cap FROM stock_valuation ORDER BY ts_code, trade_date",
                conn,
            )
        except Exception:
            data["valuation"] = pd.DataFrame()

        try:
            data["fundamental"] = pd.read_sql(
                "SELECT ts_code, report_date, roe FROM stock_fundamental ORDER BY ts_code, report_date",
                conn,
            )
        except Exception:
            data["fundamental"] = pd.DataFrame()

        return data

    def _get_latest_fundamental(self, fundamental: pd.DataFrame, date: str) -> pd.DataFrame:
        """获取指定日期前最近的财务数据"""
        if fundamental.empty:
            return pd.DataFrame()

        mask = fundamental["report_date"] <= date
        subset = fundamental.loc[mask]

        if subset.empty:
            return pd.DataFrame()

        # 每只股票取最新一期
        idx = subset.groupby("ts_code")["report_date"].idxmax()
        return subset.loc[idx].reset_index(drop=True)

    def _get_latest_valuation(self, valuation: pd.DataFrame, date: str) -> pd.DataFrame:
        """获取指定日期的市值数据（取最近的）"""
        if valuation.empty:
            return pd.DataFrame()

        mask = valuation["trade_date"] <= date
        subset = valuation.loc[mask]

        if subset.empty:
            return pd.DataFrame()

        idx = subset.groupby("ts_code")["trade_date"].idxmax()
        return subset.loc[idx].reset_index(drop=True)

    def _build_factor_scores(
        self,
        daily: pd.DataFrame,
        valuation: pd.DataFrame,
        fundamental: pd.DataFrame,
        date: str,
    ) -> pd.DataFrame:
        """在指定日期构建因子得分"""
        if daily.empty:
            return pd.DataFrame()

        # 动量因子：计算每只股票的N日收益率
        daily_up_to = daily[daily["trade_date"] <= date].copy()
        if daily_up_to.empty:
            return pd.DataFrame()

        momentum_rows = []
        for ts_code, group in daily_up_to.groupby("ts_code"):
            group = group.sort_values("trade_date")
            if len(group) < self.momentum_days + 1:
                continue
            close = group["close"].values
            ret = (close[-1] - close[-self.momentum_days]) / close[-self.momentum_days]
            if np.isnan(ret):
                continue
            momentum_rows.append({
                "ts_code": ts_code,
                "momentum": ret,
                "close": close[-1],
            })

        if not momentum_rows:
            return pd.DataFrame()

        scores = pd.DataFrame(momentum_rows)

        # 市值因子排名（越小越好 → 排名越小得分越高）
        if not valuation.empty:
            val_data = self._get_latest_valuation(valuation, date)
            if not val_data.empty:
                val_data = val_data[["ts_code", "circulating_market_cap"]].rename(
                    columns={"circulating_market_cap": "market_cap"}
                )
                scores = scores.merge(val_data, on="ts_code", how="left")
                scores["cap_rank"] = scores["market_cap"].rank(ascending=True, na_option="bottom")
            else:
                scores["cap_rank"] = np.nan
        else:
            scores["cap_rank"] = np.nan

        # 质量因子排名（ROE越高越好 → 排名越大得分越高）
        if not fundamental.empty:
            fund_data = self._get_latest_fundamental(fundamental, date)
            if not fund_data.empty:
                fund_data = fund_data[["ts_code", "roe"]]
                scores = scores.merge(fund_data, on="ts_code", how="left")
                scores["quality_rank"] = scores["roe"].rank(ascending=False, na_option="bottom")
            else:
                scores["quality_rank"] = np.nan
        else:
            scores["quality_rank"] = np.nan

        # 动量因子排名
        scores["momentum_rank"] = scores["momentum"].rank(ascending=False, na_option="bottom")

        # 综合得分：等权平均排名（越小越好）
        ranks = []
        weights = []
        if scores["cap_rank"].notna().any():
            ranks.append(scores["cap_rank"])
            weights.append(self.w_cap)
        if scores["quality_rank"].notna().any():
            ranks.append(scores["quality_rank"])
            weights.append(self.w_quality)
        ranks.append(scores["momentum_rank"])
        weights.append(self.w_momentum)

        total_weight = sum(weights)
        if total_weight > 0:
            combined = sum(r * w for r, w in zip(ranks, weights)) / total_weight
        else:
            combined = scores["momentum_rank"]

        scores["combined_score"] = combined
        scores = scores.sort_values("combined_score").reset_index(drop=True)
        scores["score_rank"] = range(1, len(scores) + 1)

        return scores

    def run_multifactor_backtest(self, data: Optional[dict] = None) -> dict:
        """运行多因子选股回测

        Args:
            data: 可选，预加载数据dict (daily, valuation, fundamental)

        Returns:
            dict with performance metrics
        """
        if data is None:
            if not DB_PATH.exists():
                return self._empty_result()
            conn = sqlite3.connect(str(DB_PATH))
            data = self._load_data(conn)
            conn.close()

        daily = data.get("daily", pd.DataFrame())
        valuation = data.get("valuation", pd.DataFrame())
        fundamental = data.get("fundamental", pd.DataFrame())

        if daily.empty:
            return self._empty_result()

        # 获取所有交易日
        all_dates = sorted(daily["trade_date"].unique())

        if len(all_dates) < self.rebalance_days + self.momentum_days:
            return self._empty_result()

        capital = float(INITIAL_CAPITAL)
        # holdings: dict of ts_code -> {"shares": int, "entry_price": float, "entry_date": str}
        holdings: Dict[str, dict] = {}
        trades: List[dict] = []
        days_since_rebalance = 0
        portfolio_values: List[float] = []

        for date in all_dates:
            # 当天价格
            day_data = daily[daily["trade_date"] == date]
            prices_today = dict(zip(day_data["ts_code"], day_data["close"]))

            # 止损检查
            to_stop = []
            for ts_code, holding in holdings.items():
                if ts_code in prices_today:
                    pnl = (prices_today[ts_code] - holding["entry_price"]) / holding["entry_price"]
                    if pnl <= -self.stop_loss:
                        to_stop.append(ts_code)

            for ts_code in to_stop:
                holding = holdings[ts_code]
                sell_price = prices_today[ts_code]
                revenue = holding["shares"] * sell_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
                capital += revenue
                pnl_pct = (sell_price / holding["entry_price"] - 1) * 100
                trades.append({
                    "ts_code": ts_code,
                    "action": "sell",
                    "date": date,
                    "price": sell_price,
                    "pnl_pct": pnl_pct,
                    "reason": f"止损 {pnl_pct:+.1f}%",
                })
                del holdings[ts_code]

            days_since_rebalance += 1

            # 调仓逻辑
            if days_since_rebalance >= self.rebalance_days:
                scores = self._build_factor_scores(daily, valuation, fundamental, date)

                if not scores.empty:
                    top_stocks = scores.head(self.top_n)["ts_code"].tolist()
                    cutoff = min(2 * self.top_n, len(scores))
                    pool_stocks = set(scores.head(cutoff)["ts_code"].tolist())

                    # 卖出掉出前2N的持仓
                    to_sell = [c for c in list(holdings.keys()) if c not in pool_stocks]
                    # 卖出掉出top N且不在pool中的持仓
                    to_sell += [c for c in list(holdings.keys()) if c not in top_stocks and c in pool_stocks]

                    for ts_code in to_sell:
                        if ts_code in prices_today and ts_code in holdings:
                            sell_price = prices_today[ts_code]
                            holding = holdings[ts_code]
                            revenue = holding["shares"] * sell_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
                            capital += revenue
                            pnl_pct = (sell_price / holding["entry_price"] - 1) * 100
                            trades.append({
                                "ts_code": ts_code,
                                "action": "sell",
                                "date": date,
                                "price": sell_price,
                                "pnl_pct": pnl_pct,
                                "reason": "调仓卖出 排名落后",
                            })
                            del holdings[ts_code]

                    # 买入新进top N的股票
                    to_buy = [c for c in top_stocks if c not in holdings]
                    if to_buy:
                        per_stock = capital / len(to_buy) if to_buy else 0
                        for ts_code in to_buy:
                            if ts_code in prices_today:
                                buy_price = prices_today[ts_code]
                                shares = int(per_stock / buy_price) // BOARD_LOT * BOARD_LOT
                                if shares >= BOARD_LOT:
                                    cost = shares * buy_price * (1 + COMMISSION_RATE)
                                    if cost <= capital:
                                        capital -= cost
                                        holdings[ts_code] = {
                                            "shares": shares,
                                            "entry_price": buy_price,
                                            "entry_date": date,
                                        }
                                        rank = scores.loc[scores["ts_code"] == ts_code, "score_rank"].values[0]
                                        trades.append({
                                            "ts_code": ts_code,
                                            "action": "buy",
                                            "date": date,
                                            "price": buy_price,
                                            "pnl_pct": 0.0,
                                            "reason": f"调仓买入 因子排名{rank}",
                                        })

                    days_since_rebalance = 0

            # 计算当日组合价值
            total_value = capital
            for ts_code, holding in holdings.items():
                if ts_code in prices_today:
                    total_value += holding["shares"] * prices_today[ts_code]
                else:
                    total_value += holding["shares"] * holding["entry_price"]
            portfolio_values.append(total_value)

        # 期末清算
        for ts_code, holding in list(holdings.items()):
            last_price = holding["entry_price"]
            stock_data = daily[daily["ts_code"] == ts_code]
            if not stock_data.empty:
                last_price = stock_data.iloc[-1]["close"]
            revenue = holding["shares"] * last_price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            pnl_pct = (last_price / holding["entry_price"] - 1) * 100
            capital += revenue
            trades.append({
                "ts_code": ts_code,
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
            avg_return = np.mean(returns)
            sharpe = (np.mean(returns) / (np.std(returns) + 0.001)) * np.sqrt(252) if len(returns) > 1 else 0.0
        else:
            win_rate = 0.0
            avg_return = 0.0
            sharpe = 0.0

        if portfolio_values:
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0.0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "total_trades": len(trades),
            "sell_trades": len(sell_trades),
            "max_drawdown": max_drawdown,
            "trades": trades,
            "final_capital": capital,
            "portfolio_values": portfolio_values,
        }

    @staticmethod
    def _empty_result() -> dict:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_trades": 0,
            "sell_trades": 0,
            "max_drawdown": 0.0,
            "trades": [],
            "final_capital": INITIAL_CAPITAL,
            "portfolio_values": [],
        }

    def print_summary(self, result: dict):
        """打印回测结果摘要"""
        print("=" * 70)
        print("  多因子选股策略 回测结果")
        print(f"  参数: Top-N={self.top_n}  调仓={self.rebalance_days}日  "
              f"动量={self.momentum_days}日  止损={self.stop_loss*100:.0f}%")
        print(f"  因子权重: 市值={self.w_cap:.2f}  质量={self.w_quality:.2f}  动量={self.w_momentum:.2f}")
        print("=" * 70)
        print(f"  初始资金:     {INITIAL_CAPITAL:>12,.0f}")
        print(f"  期末资金:     {result['final_capital']:>12,.2f}")
        print(f"  总收益率:     {result['total_return']:>+11.2f}%")
        print(f"  最大回撤:     {result['max_drawdown']:>11.2f}%")
        print(f"  夏普比率:     {result['sharpe']:>11.2f}")
        print(f"  交易次数:     {result['total_trades']:>12d}  (卖出: {result['sell_trades']})")
        print(f"  胜率:         {result['win_rate']:>11.1f}%")
        print(f"  平均收益:     {result['avg_return']:>+11.2f}%")
        print("=" * 70)

        trades = result.get("trades", [])
        if trades:
            print(f"\n  最近20笔交易:")
            print(f"  {'日期':<12} {'代码':<12} {'动作':<6} {'价格':>10} {'收益':>10} {'原因'}")
            print("  " + "-" * 66)
            for t in trades[-20:]:
                print(f"  {t['date']:<12} {t['ts_code']:<12} {t['action']:<6} "
                      f"{t['price']:>10.2f} {t['pnl_pct']:>+9.1f}% {t['reason']}")


def main():
    db_path = DB_PATH
    if not db_path.exists():
        print(f"数据库不存在: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    strategy = MultifactorStrategy(
        top_n=10,
        rebalance_days=20,
        momentum_days=20,
        stop_loss=0.10,
    )

    data = strategy._load_data(conn)
    conn.close()

    daily = data.get("daily", pd.DataFrame())
    valuation = data.get("valuation", pd.DataFrame())
    fundamental = data.get("fundamental", pd.DataFrame())

    print(f"数据加载情况:")
    print(f"  日线数据: {len(daily)} 条, {daily['ts_code'].nunique() if not daily.empty else 0} 只股票")
    print(f"  估值数据: {len(valuation)} 条")
    print(f"  财务数据: {len(fundamental)} 条")

    if daily.empty:
        print("日线数据为空，无法运行回测")
        return

    result = strategy.run_multifactor_backtest(data)
    strategy.print_summary(result)


if __name__ == "__main__":
    main()
