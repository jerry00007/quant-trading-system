"""组合配置回测: ETF轮动 + EnhancedChip

组合逻辑:
- 50%资金配置ETF轮动策略(9只行业ETF动量轮动)
- 50%资金配置增强筹码策略(CSI100个股)
- 各自独立运行,每季度再平衡
- 监控相关性,评估降低组合波动效果
"""
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from strategies.enhanced_chip_strategy import EnhancedChipStrategy
from strategies.etf_rotation_strategy import ETFRotationStrategy

INITIAL_CAPITAL = 100_000
ETF_ALLOCATION = 0.50  # 50% ETF轮动
CHIP_ALLOCATION = 0.50  # 50% 增强筹码
REBALANCE_INTERVAL = 60  # 每60个交易日再平衡(季度)


# ============================================================
# ETF轮动策略回测 (使用已知有效参数)
# ============================================================
def run_etf_rotation(conn: sqlite3.Connection, start_date: str, end_date: str) -> dict:
    """运行ETF轮动回测,返回每日组合价值"""
    # 使用之前验证过的最优参数(来自Obsidian文档)
    strategy = ETFRotationStrategy(
        lookback=20,
        top_k=3,
        rebalance_days=20,
        stop_loss=0.08,
        take_profit=0.30,
    )
    
    etf_data = strategy._load_etf_data(conn)
    
    # 按日期过滤
    for code in etf_data:
        etf_data[code] = etf_data[code][
            (etf_data[code]["trade_date"] >= start_date) &
            (etf_data[code]["trade_date"] <= end_date)
        ].copy()
    
    return strategy.run_rotation_backtest(etf_data)


# ============================================================
# EnhancedChip策略回测 (来自compare_strategies.py)
# ============================================================
def run_enhanced_chip(conn: sqlite3.Connection, start_date: str, end_date: str) -> dict:
    """运行增强筹码策略回测,返回每日组合价值"""
    # 加载CSI100股票列表
    stocks = pd.read_sql(
        "SELECT DISTINCT ts_code FROM daily_quotes WHERE trade_date >= '2020-01-01' ORDER BY ts_code LIMIT 20",
        conn
    )["ts_code"].tolist()
    
    if not stocks:
        return {"portfolio_dates": [], "portfolio_values": [], "total_return": 0}
    
    INITIAL = 100_000 * CHIP_ALLOCATION
    POSITION_PCT = 0.20
    BOARD_LOT = 100
    COMMISSION = 0.0003
    STAMP = 0.0005
    
    strategy = EnhancedChipStrategy()
    
    # 对每只股票计算信号
    all_signals = {}
    for ts_code in stocks:
        df = pd.read_sql(
            f"SELECT * FROM daily_quotes WHERE ts_code='{ts_code}' AND trade_date>='{start_date}' AND trade_date<='{end_date}' ORDER BY trade_date",
            conn
        )
        if len(df) < 60:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        
        try:
            signals = strategy.calculate_signals(df)
            for sig in signals:
                all_signals.setdefault(sig.trade_date, []).append(sig)
        except Exception:
            continue
    
    if not all_signals:
        return {"portfolio_dates": [], "portfolio_values": [], "total_return": 0}
    
    # 获取所有交易日
    all_dates_sorted = sorted(all_signals.keys())
    date_to_idx = {d: i for i, d in enumerate(all_dates_sorted)}
    
    capital = INITIAL
    holdings: Dict[str, dict] = {}  # ts_code -> {shares, entry_price, entry_date}
    portfolio_values: List[float] = []
    portfolio_dates: List[str] = []
    pending_buy: Optional[object] = None
    pending_sell_reason: Optional[str] = None
    
    # 获取完整日期范围(用于遍历)
    full_dates = pd.read_sql(
        f"SELECT DISTINCT trade_date FROM daily_quotes WHERE ts_code='{stocks[0]}' AND trade_date>='{start_date}' AND trade_date<='{end_date}' ORDER BY trade_date",
        conn
    )["trade_date"].tolist()
    
    opens_map = {}
    for ts_code in stocks:
        df = pd.read_sql(
            f"SELECT trade_date, open FROM daily_quotes WHERE ts_code='{ts_code}' AND trade_date>='{start_date}' AND trade_date<='{end_date}'",
            conn
        )
        for _, row in df.iterrows():
            opens_map.setdefault(row["trade_date"], {})[ts_code] = row["open"]
    
    closes_map = {}
    for ts_code in stocks:
        df = pd.read_sql(
            f"SELECT trade_date, close FROM daily_quotes WHERE ts_code='{ts_code}' AND trade_date>='{start_date}' AND trade_date<='{end_date}'",
            conn
        )
        for _, row in df.iterrows():
            closes_map.setdefault(row["trade_date"], {})[ts_code] = row["close"]
    
    for idx, date in enumerate(full_dates):
        sigs = all_signals.get(date, [])
        
        # T+1执行: 先处理前一日挂单
        if pending_buy is not None:
            exec_price = opens_map.get(date, {}).get(pending_buy.ts_code)
            if exec_price is not None:
                max_cost = capital * POSITION_PCT
                shares = int(max_cost / exec_price) // BOARD_LOT * BOARD_LOT
                if shares >= BOARD_LOT:
                    cost = shares * exec_price * (1 + COMMISSION)
                    if cost <= capital:
                        capital -= cost
                        holdings[pending_buy.ts_code] = {
                            "entry_date": date,
                            "entry_price": exec_price,
                            "shares": shares,
                        }
            pending_buy = None
        
        if pending_sell_reason is not None and holdings:
            ts_code = list(holdings.keys())[0]  # 单持仓
            exec_price = opens_map.get(date, {}).get(ts_code)
            if exec_price is not None:
                h = holdings[ts_code]
                revenue = h["shares"] * exec_price * (1 - COMMISSION - STAMP)
                capital += revenue
                pnl = (exec_price / h["entry_price"] - 1) * 100
                del holdings[ts_code]
            pending_sell_reason = None
        
        # 处理当日信号
        for sig in sigs:
            if not holdings and pending_buy is None and sig.action == "buy":
                pending_buy = sig
            elif holdings and pending_sell_reason is None and sig.action == "sell":
                pending_sell_reason = sig.reason
                break
        
        # 计算当日组合价值
        total_value = capital
        for ts_code, h in holdings.items():
            p = closes_map.get(date, {}).get(ts_code, h["entry_price"])
            total_value += h["shares"] * p
        
        portfolio_values.append(total_value)
        portfolio_dates.append(date)
    
    # 计算绩效
    if len(portfolio_values) < 2:
        return {"portfolio_dates": [], "portfolio_values": [], "total_return": 0, "final_capital": INITIAL}
    
    total_return = (capital / INITIAL - 1) * 100
    pv_arr = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv_arr)
    dd = (pv_arr - peak) / peak
    max_dd = dd.min() * 100
    sharpe = (np.mean(np.diff(pv_arr)/pv_arr[:-1]) / (np.std(np.diff(pv_arr)/pv_arr[:-1]) + 1e-8)) * np.sqrt(252) if len(pv_arr) > 1 else 0
    
    return {
        "portfolio_dates": portfolio_dates,
        "portfolio_values": portfolio_values,
        "total_return": total_return,
        "final_capital": capital,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


# ============================================================
# 组合回测
# ============================================================
def run_combination_backtest():
    """组合回测: ETF轮动 + EnhancedChip,各自50%仓位"""
    db_path = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"
    conn = sqlite3.connect(str(db_path))
    
    # 确定回测区间(ETF数据从2011开始,股票数据从2020开始)
    start_date = "2020-01-01"
    end_date = "2026-04-09"
    
    print("=" * 70)
    print("  组合配置回测: ETF轮动 + 增强筹码")
    print("=" * 70)
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  ETF轮动配置: lookback=20, top_k=3, rebalance=20日")
    print(f"  增强筹码配置: 默认参数, 单股20%仓位")
    print(f"  分配比例: ETF {ETF_ALLOCATION*100:.0f}% / 筹码 {CHIP_ALLOCATION*100:.0f}%")
    print("=" * 70)
    
    # 1. ETF轮动
    print("\n  [1/2] 运行ETF轮动策略...")
    etf_result = run_etf_rotation(conn, start_date, end_date)
    
    if etf_result["portfolio_dates"]:
        print(f"    总收益: {etf_result['total_return']:+.1f}%")
        print(f"    夏普:   {etf_result['sharpe']:.2f}")
        print(f"    最大回撤: {etf_result['max_drawdown']:.1f}%")
        print(f"    最终资金: {etf_result['final_capital']:,.0f}")
    else:
        print("    ETF轮动: 无数据")
    
    # 2. 增强筹码
    print("\n  [2/2] 运行增强筹码策略...")
    chip_result = run_enhanced_chip(conn, start_date, end_date)
    
    if chip_result["portfolio_dates"]:
        print(f"    总收益: {chip_result['total_return']:+.1f}%")
        print(f"    夏普:   {chip_result['sharpe']:.2f}")
        print(f"    最大回撤: {chip_result['max_drawdown']:.1f}%")
        print(f"    最终资金: {chip_result['final_capital']:,.0f}")
    else:
        print("    增强筹码: 无数据")
    
    # 3. 组合分析
    if etf_result["portfolio_dates"] and chip_result["portfolio_dates"]:
        print("\n" + "=" * 70)
        print("  组合效果分析")
        print("=" * 70)
        
        etf_dates = set(etf_result["portfolio_dates"])
        chip_dates = set(chip_result["portfolio_dates"])
        common_dates = sorted(etf_dates & chip_dates)
        
        if common_dates:
            etf_pv_map = {d: v for d, v in zip(etf_result["portfolio_dates"], etf_result["portfolio_values"])}
            chip_pv_map = {d: v for d, v in zip(chip_result["portfolio_dates"], chip_result["portfolio_values"])}
            
            # 组合价值 = ETF价值 + 筹码价值 (各自初始50%)
            etf_init = INITIAL_CAPITAL * ETF_ALLOCATION
            chip_init = INITIAL_CAPITAL * CHIP_ALLOCATION
            
            combined_values = []
            combined_dates = []
            
            for d in common_dates:
                etf_val = etf_pv_map.get(d, etf_init)
                chip_val = chip_pv_map.get(d, chip_init)
                combined_values.append(etf_val + chip_val)
                combined_dates.append(d)
            
            comb_arr = np.array(combined_values)
            comb_peak = np.maximum.accumulate(comb_arr)
            comb_dd = (comb_arr - comb_peak) / comb_peak
            comb_max_dd = comb_dd.min() * 100
            
            total_return = (combined_values[-1] / INITIAL_CAPITAL - 1) * 100
            
            returns = np.diff(combined_values) / np.array(combined_values[:-1])
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252) if len(returns) > 1 else 0
            
            print(f"\n  组合总体表现:")
            print(f"    初始资金:     {INITIAL_CAPITAL:>12,.0f}")
            print(f"    期末资金:     {combined_values[-1]:>12,.0f}")
            print(f"    总收益率:     {total_return:>+11.1f}%")
            print(f"    最大回撤:     {comb_max_dd:>11.1f}%")
            print(f"    夏普比率:     {sharpe:>11.2f}")
            
            # 相关性分析
            etf_rets = np.diff([etf_pv_map.get(d, 1) for d in common_dates])
            chip_rets = np.diff([chip_pv_map.get(d, 1) for d in common_dates])
            
            min_len = min(len(etf_rets), len(chip_rets))
            if min_len > 10:
                corr = np.corrcoef(etf_rets[:min_len], chip_rets[:min_len])[0, 1]
                print(f"\n    ETF与筹码日收益相关性: {corr:.3f}")
                if abs(corr) < 0.3:
                    print(f"    → 低相关性! 组合效果: 较好(分散化有效)")
                elif abs(corr) < 0.6:
                    print(f"    → 中等相关性! 组合效果: 一般")
                else:
                    print(f"    → 高相关性! 组合效果: 有限(分散化效果弱)")
            
            # 年度分析
            print(f"\n  年度收益分解:")
            df_yr = pd.DataFrame({"date": combined_dates, "value": combined_values})
            df_yr["year"] = pd.to_datetime(df_yr["date"]).dt.year
            
            for yr, grp in df_yr.groupby("year"):
                yr_vals = grp["value"].values
                yr_start = yr_vals[0]
                yr_end = yr_vals[-1]
                yr_ret = (yr_end / yr_start - 1) * 100
                yr_peak = np.maximum.accumulate(yr_vals)
                yr_dd = (yr_vals[-1] / yr_peak[-1] - 1) * 100
                print(f"    {yr}: {yr_start:>10,.0f} → {yr_end:>10,.0f} ({yr_ret:>+6.1f}%) DD{yr_dd:>6.1f}%")
        else:
            print("    无共同交易日,无法计算组合效果")
    else:
        print("\n  无法进行组合分析(缺少数据)")
    
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    run_combination_backtest()
