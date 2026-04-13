"""实盘交易 Dashboard — FastAPI Backend

提供每日信号、持仓管理、收益追踪等API。
数据流: akshare → SQLite → 策略信号 → REST API → 前端页面
"""
import sqlite3
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"
PORTFOLIO_DB = PROJECT_ROOT / "data" / "sqlite" / "portfolio.db"
WEB_DIR = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT))

from strategies.etf_rotation_strategy import ETFRotationStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="量化交易系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ETF Pool Config ──────────────────────────────────────────────

ETF_POOL = [
    {"code": "sh510500", "name": "500ETF"},
    {"code": "sh511010", "name": "国债ETF"},
    {"code": "sh513100", "name": "纳指ETF"},
    {"code": "sh513500", "name": "标普500"},
    {"code": "sh518880", "name": "黄金ETF"},
    {"code": "sz159915", "name": "创业板ETF"},
    {"code": "sz159919", "name": "沪深300ETF"},
    {"code": "sz159920", "name": "恒生ETF"},
    {"code": "sz159925", "name": "深证100"},
    {"code": "sz159949", "name": "创业板50"},
    {"code": "sz159966", "name": "科技ETF"},
    {"code": "sz159996", "name": "芯片ETF"},
    {"code": "sz159997", "name": "游戏ETF"},
    {"code": "sz159998", "name": "医药ETF"},
]

ETF_CODE_MAP = {e["code"]: e["name"] for e in ETF_POOL}

PORTFOLIO_CONFIG = {
    "initial_capital": 1_000_000,
    "etf_allocation": 0.50,
    "stock_allocation": 0.30,
    "cash_allocation": 0.20,
    "etf_params": {"lookback": 20, "top_k": 3, "rebalance_days": 20, "stop_loss": 0.08, "take_profit": 0.30},
}


# ── Portfolio DB Init ────────────────────────────────────────────

def _init_portfolio_db():
    PORTFOLIO_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PORTFOLIO_DB))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS holdings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            shares INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            target_weight REAL DEFAULT 0,
            strategy TEXT DEFAULT 'manual',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            action TEXT NOT NULL CHECK(action IN ('buy','sell')),
            shares INTEGER NOT NULL,
            price REAL NOT NULL,
            date TEXT NOT NULL,
            reason TEXT DEFAULT '',
            strategy TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date TEXT NOT NULL,
            execute_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            action TEXT NOT NULL CHECK(action IN ('buy','sell','hold')),
            shares INTEGER DEFAULT 0,
            reference_price REAL DEFAULT 0,
            reason TEXT DEFAULT '',
            strategy TEXT DEFAULT '',
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending','executed','skipped','expired')),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS cash_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL NOT NULL,
            operation TEXT NOT NULL,
            date TEXT NOT NULL,
            reason TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

_init_portfolio_db()


def _get_portfolio_conn():
    return sqlite3.connect(str(PORTFOLIO_DB))


def _get_db_conn():
    return sqlite3.connect(str(DB_PATH))


# ── Signal Generation ────────────────────────────────────────────

def _get_latest_trading_date(conn) -> Optional[str]:
    row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
    return row[0] if row and row[0] else None


def _generate_etf_signals(conn, as_of_date: str = None) -> list:
    strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
    etf_data = strategy._load_etf_data(conn)
    if not etf_data:
        return []

    all_dates = sorted(set(d for df in etf_data.values() for d in df["trade_date"].tolist()))
    if not as_of_date:
        as_of_date = all_dates[-1]

    if as_of_date not in all_dates:
        closest = [d for d in all_dates if d <= as_of_date]
        if not closest:
            return []
        as_of_date = closest[-1]

    momentum_df = strategy._compute_momentum(etf_data, as_of_date)
    if momentum_df.empty:
        return []

    top_etfs = momentum_df.head(strategy.top_k)["etf_code"].tolist()

    portfolio_conn = _get_portfolio_conn()
    current_holdings_rows = portfolio_conn.execute(
        "SELECT symbol, shares FROM holdings WHERE strategy='etf_rotation'"
    ).fetchall()
    portfolio_conn.close()

    current_holdings = {row[0]: row[1] for row in current_holdings_rows}

    # Calculate available cash for ETF allocation
    initial = PORTFOLIO_CONFIG["initial_capital"]
    etf_budget = initial * PORTFOLIO_CONFIG["etf_allocation"]
    existing_etf_value = 0.0
    for sym, shrs in current_holdings.items():
        price_row = conn.execute(
            "SELECT close FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
            (sym, as_of_date)
        ).fetchone()
        if price_row:
            existing_etf_value += shrs * price_row[0]
    available_etf_cash = etf_budget - existing_etf_value

    next_date_idx = all_dates.index(as_of_date) + 1 if as_of_date in all_dates else len(all_dates)
    execute_date = all_dates[next_date_idx] if next_date_idx < len(all_dates) else as_of_date

    # Count new buy signals to evenly distribute budget
    new_buys = [code for code in top_etfs if code not in current_holdings]
    per_buy_budget = (available_etf_cash / len(new_buys)) if new_buys else 0

    signals = []

    for code in list(current_holdings.keys()):
        if code not in top_etfs:
            ref_price_row = conn.execute(
                "SELECT open FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
                (code, as_of_date)
            ).fetchone()
            ref_price = ref_price_row[0] if ref_price_row else 0
            signals.append({
                "signal_date": as_of_date,
                "execute_date": execute_date,
                "symbol": code,
                "name": ETF_CODE_MAP.get(code, code),
                "action": "sell",
                "shares": current_holdings[code],
                "reference_price": round(ref_price, 4),
                "reason": f"调仓卖出，排名跌出Top{strategy.top_k}",
                "strategy": "etf_rotation",
                "status": "pending",
            })

    for code in top_etfs:
        if code not in current_holdings:
            rank_row = momentum_df.loc[momentum_df["etf_code"] == code]
            rank_val = int(rank_row["rank"].values[0]) if len(rank_row) > 0 else 0
            ref_price_row = conn.execute(
                "SELECT open FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
                (code, as_of_date)
            ).fetchone()
            ref_price = ref_price_row[0] if ref_price_row else 0
            # Calculate suggested shares (100-share lots, ETF minimum)
            suggested_shares = int(per_buy_budget / ref_price / 100) * 100 if ref_price > 0 else 0
            signals.append({
                "signal_date": as_of_date,
                "execute_date": execute_date,
                "symbol": code,
                "name": ETF_CODE_MAP.get(code, code),
                "action": "buy",
                "shares": suggested_shares,
                "reference_price": round(ref_price, 4),
                "reason": f"调仓买入，动量排名{rank_val}",
                "strategy": "etf_rotation",
                "status": "pending",
            })

    # Check for rebalance day (how many trading days since last rebalance)
    signals.sort(key=lambda x: (0 if x["action"] == "sell" else 1))
    return signals


def _compute_portfolio_metrics(conn) -> dict:
    portfolio_conn = _get_portfolio_conn()
    holdings = portfolio_conn.execute("SELECT symbol, name, shares, entry_price, entry_date, strategy FROM holdings").fetchall()
    trades = portfolio_conn.execute("SELECT symbol, action, shares, price, date, reason, strategy FROM trades ORDER BY date DESC").fetchall()

    cash_row = portfolio_conn.execute("SELECT COALESCE(SUM(amount), 0) FROM cash_ledger").fetchone()
    initial = PORTFOLIO_CONFIG["initial_capital"]
    cash = initial + cash_row[0]
    cash = max(cash, 0)

    portfolio_conn.close()

    total_market_value = 0
    holdings_detail = []
    latest_date = _get_latest_trading_date(conn)

    for symbol, name, shares, entry_price, entry_date, strategy in holdings:
        if symbol.startswith("sh") or symbol.startswith("sz"):
            table = "etf_daily_quotes"
            code_col = "etf_code"
        else:
            table = "daily_quotes"
            code_col = "ts_code"

        price_row = conn.execute(
            f"SELECT close FROM {table} WHERE {code_col}=? ORDER BY trade_date DESC LIMIT 1",
            (symbol,)
        ).fetchone()
        current_price = price_row[0] if price_row else entry_price
        market_value = shares * current_price
        total_market_value += market_value
        pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

        holdings_detail.append({
            "symbol": symbol,
            "name": name or ETF_CODE_MAP.get(symbol, symbol),
            "shares": shares,
            "entry_price": round(entry_price, 4),
            "current_price": round(current_price, 4),
            "market_value": round(market_value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "entry_date": entry_date,
            "strategy": strategy or "manual",
        })

    total_assets = cash + total_market_value
    total_pnl = total_assets - initial
    total_pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0

    etf_value = sum(h["market_value"] for h in holdings_detail if h["strategy"] == "etf_rotation")
    stock_value = sum(h["market_value"] for h in holdings_detail if h["strategy"] != "etf_rotation" and h["strategy"] != "manual")
    manual_value = sum(h["market_value"] for h in holdings_detail if h["strategy"] == "manual")

    recent_trades = [
        {
            "symbol": t[0],
            "name": ETF_CODE_MAP.get(t[0], t[0]),
            "action": t[1],
            "shares": t[2],
            "price": round(t[3], 4),
            "date": t[4],
            "reason": t[5],
            "strategy": t[6],
        }
        for t in trades[:20]
    ]

    return {
        "total_assets": round(total_assets, 2),
        "cash": round(cash, 2),
        "market_value": round(total_market_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "initial_capital": initial,
        "allocation": {
            "etf": round(etf_value, 2),
            "etf_pct": round(etf_value / total_assets * 100, 1) if total_assets > 0 else 0,
            "stock": round(stock_value, 2),
            "stock_pct": round(stock_value / total_assets * 100, 1) if total_assets > 0 else 0,
            "cash_pct": round(cash / total_assets * 100, 1) if total_assets > 0 else 0,
            "manual": round(manual_value, 2),
        },
        "holdings": holdings_detail,
        "recent_trades": recent_trades,
        "data_date": latest_date,
    }


# ── API Models ───────────────────────────────────────────────────

class TradeConfirmRequest(BaseModel):
    symbol: str
    action: str
    shares: int
    price: float
    date: str
    reason: str = ""
    strategy: str = "manual"
    name: str = ""


class SignalActionRequest(BaseModel):
    signal_id: int
    action: str  # "executed", "skipped"
    actual_price: Optional[float] = None
    actual_shares: Optional[int] = None


class CashOperationRequest(BaseModel):
    amount: float
    operation: str  # "deposit", "withdraw"
    reason: str = ""


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard():
    conn = _get_db_conn()
    try:
        portfolio = _compute_portfolio_metrics(conn)
        latest_date = portfolio["data_date"]

        signals = _generate_etf_signals(conn, latest_date)

        portfolio_conn = _get_portfolio_conn()
        pending = portfolio_conn.execute(
            "SELECT id, signal_date, execute_date, symbol, action, shares, reference_price, reason, strategy, status FROM signals WHERE status='pending' ORDER BY execute_date DESC"
        ).fetchall()
        portfolio_conn.close()

        pending_signals = [
            {
                "id": row[0],
                "signal_date": row[1],
                "execute_date": row[2],
                "symbol": row[3],
                "name": ETF_CODE_MAP.get(row[3], row[3]),
                "action": row[4],
                "shares": row[5],
                "reference_price": row[6],
                "reason": row[7],
                "strategy": row[8],
                "status": row[9],
            }
            for row in pending
        ]

        return {
            "portfolio": portfolio,
            "pending_signals": pending_signals,
            "fresh_signals": signals,
            "config": PORTFOLIO_CONFIG,
            "etf_pool": ETF_POOL,
        }
    finally:
        conn.close()


@app.get("/api/signals/generate")
def generate_signals(date: Optional[str] = None):
    conn = _get_db_conn()
    try:
        signals = _generate_etf_signals(conn, date)
        portfolio_conn = _get_portfolio_conn()
        for s in signals:
            portfolio_conn.execute(
                "INSERT INTO signals (signal_date, execute_date, symbol, name, action, shares, reference_price, reason, strategy, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (s["signal_date"], s["execute_date"], s["symbol"], s["name"], s["action"], s["shares"], s["reference_price"], s["reason"], s["strategy"], s["status"])
            )
        portfolio_conn.commit()
        portfolio_conn.close()
        return {"signals": signals, "count": len(signals)}
    finally:
        conn.close()


@app.post("/api/signals/confirm")
def confirm_signal(req: SignalActionRequest):
    portfolio_conn = _get_portfolio_conn()
    row = portfolio_conn.execute("SELECT symbol, action, shares, reference_price, reason, strategy, name FROM signals WHERE id=?", (req.signal_id,)).fetchone()
    if not row:
        portfolio_conn.close()
        raise HTTPException(404, "Signal not found")

    symbol, action, shares, ref_price, reason, strategy, name = row
    actual_price = req.actual_price or ref_price
    actual_shares = req.actual_shares or shares

    if req.action == "executed":
        portfolio_conn.execute("UPDATE signals SET status='executed' WHERE id=?", (req.signal_id,))

        if action == "buy":
            portfolio_conn.execute(
                "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,date('now'),?,?)",
                (symbol, name or ETF_CODE_MAP.get(symbol, symbol), "buy", actual_shares, actual_price, reason, strategy)
            )
            portfolio_conn.execute(
                "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'withdraw',date('now'),?)",
                (-(actual_shares * actual_price), f"买入 {symbol}")
            )
            existing = portfolio_conn.execute("SELECT shares, entry_price FROM holdings WHERE symbol=? AND strategy=?", (symbol, strategy)).fetchone()
            if existing:
                old_shares, old_price = existing
                new_shares = old_shares + actual_shares
                new_entry = (old_shares * old_price + actual_shares * actual_price) / new_shares
                portfolio_conn.execute("UPDATE holdings SET shares=?, entry_price=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?", (new_shares, new_entry, symbol, strategy))
            else:
                portfolio_conn.execute(
                    "INSERT INTO holdings (symbol, name, shares, entry_price, entry_date, strategy) VALUES (?,?,?,?,date('now'),?)",
                    (symbol, name or ETF_CODE_MAP.get(symbol, symbol), actual_shares, actual_price, strategy)
                )

        elif action == "sell":
            portfolio_conn.execute(
                "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,date('now'),?,?)",
                (symbol, name or ETF_CODE_MAP.get(symbol, symbol), "sell", actual_shares, actual_price, reason, strategy)
            )
            portfolio_conn.execute(
                "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'deposit',date('now'),?)",
                (actual_shares * actual_price, f"卖出 {symbol}")
            )
            existing = portfolio_conn.execute("SELECT shares FROM holdings WHERE symbol=? AND strategy=?", (symbol, strategy)).fetchone()
            if existing:
                remaining = existing[0] - actual_shares
                if remaining <= 0:
                    portfolio_conn.execute("DELETE FROM holdings WHERE symbol=? AND strategy=?", (symbol, strategy))
                else:
                    portfolio_conn.execute("UPDATE holdings SET shares=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?", (remaining, symbol, strategy))

    elif req.action == "skipped":
        portfolio_conn.execute("UPDATE signals SET status='skipped' WHERE id=?", (req.signal_id,))

    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


@app.post("/api/trades")
def manual_trade(req: TradeConfirmRequest):
    portfolio_conn = _get_portfolio_conn()
    name = req.name or ETF_CODE_MAP.get(req.symbol, req.symbol)

    portfolio_conn.execute(
        "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,?,?,?)",
        (req.symbol, name, req.action, req.shares, req.price, req.date, req.reason, req.strategy)
    )

    if req.action == "buy":
        portfolio_conn.execute(
            "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'withdraw',?,?)",
            (-(req.shares * req.price), req.date, f"手动买入 {req.symbol}")
        )
        existing = portfolio_conn.execute("SELECT shares, entry_price FROM holdings WHERE symbol=? AND strategy=?", (req.symbol, req.strategy)).fetchone()
        if existing:
            old_shares, old_price = existing
            new_shares = old_shares + req.shares
            new_entry = (old_shares * old_price + req.shares * req.price) / new_shares
            portfolio_conn.execute("UPDATE holdings SET shares=?, entry_price=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?", (new_shares, new_entry, req.symbol, req.strategy))
        else:
            portfolio_conn.execute(
                "INSERT INTO holdings (symbol, name, shares, entry_price, entry_date, strategy) VALUES (?,?,?,?,?,?)",
                (req.symbol, name, req.shares, req.price, req.date, req.strategy)
            )
    elif req.action == "sell":
        portfolio_conn.execute(
            "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'deposit',?,?)",
            (req.shares * req.price, req.date, f"手动卖出 {req.symbol}")
        )
        existing = portfolio_conn.execute("SELECT shares FROM holdings WHERE symbol=? AND strategy=?", (req.symbol, req.strategy)).fetchone()
        if existing:
            remaining = existing[0] - req.shares
            if remaining <= 0:
                portfolio_conn.execute("DELETE FROM holdings WHERE symbol=? AND strategy=?", (req.symbol, req.strategy))
            else:
                portfolio_conn.execute("UPDATE holdings SET shares=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?", (remaining, req.symbol, req.strategy))

    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


@app.get("/api/history")
def get_trade_history(limit: int = Query(50, ge=1, le=200)):
    portfolio_conn = _get_portfolio_conn()
    trades = portfolio_conn.execute(
        "SELECT symbol, name, action, shares, price, date, reason, strategy FROM trades ORDER BY date DESC, id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    portfolio_conn.close()
    return {
        "trades": [
            {
                "symbol": t[0], "name": t[1], "action": t[2],
                "shares": t[3], "price": round(t[4], 4),
                "date": t[5], "reason": t[6], "strategy": t[7],
            }
            for t in trades
        ]
    }


@app.get("/api/performance")
def get_performance():
    conn = _get_db_conn()
    try:
        portfolio_conn = _get_portfolio_conn()
        trades = portfolio_conn.execute(
            "SELECT symbol, action, shares, price, date, strategy FROM trades ORDER BY date"
        ).fetchall()
        portfolio_conn.close()

        if not trades:
            return {"dates": [], "values": [], "benchmark": []}

        initial = PORTFOLIO_CONFIG["initial_capital"]
        cash = initial
        holdings_map = {}
        daily_values = {}

        for symbol, action, shares, price, date, strategy in trades:
            if action == "buy":
                cost = shares * price * 1.0003
                cash -= cost
                if symbol in holdings_map:
                    old_s, old_p = holdings_map[symbol]
                    new_s = old_s + shares
                    new_p = (old_s * old_p + shares * price) / new_s
                    holdings_map[symbol] = (new_s, new_p)
                else:
                    holdings_map[symbol] = (shares, price)
            elif action == "sell":
                revenue = shares * price * (1 - 0.0003 - 0.0005)
                cash += revenue
                if symbol in holdings_map:
                    old_s, old_p = holdings_map[symbol]
                    remaining = old_s - shares
                    if remaining <= 0:
                        del holdings_map[symbol]
                    else:
                        holdings_map[symbol] = (remaining, old_p)

            mv = sum(s * p for s, p in holdings_map.values())
            daily_values[date] = cash + mv

        if not daily_values:
            return {"dates": [], "values": [], "benchmark": []}

        sorted_dates = sorted(daily_values.keys())
        values = [round(daily_values[d], 2) for d in sorted_dates]
        benchmark_start = values[0] if values else initial
        benchmark = [round(benchmark_start, 2)]

        for i in range(1, len(sorted_dates)):
            prev_d = sorted_dates[i - 1]
            curr_d = sorted_dates[i]
            idx_row = conn.execute(
                "SELECT close FROM etf_daily_quotes WHERE etf_code='sz159919' AND trade_date BETWEEN ? AND ? ORDER BY trade_date DESC LIMIT 1",
                (prev_d, curr_d)
            ).fetchone()
            if idx_row:
                ret = idx_row[0]
                prev_idx_row = conn.execute(
                    "SELECT close FROM etf_daily_quotes WHERE etf_code='sz159919' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
                    (prev_d,)
                ).fetchone()
                if prev_idx_row:
                    bench_ret = ret / prev_idx_row[0]
                    benchmark.append(round(benchmark[-1] * bench_ret, 2))
                else:
                    benchmark.append(benchmark[-1])
            else:
                benchmark.append(benchmark[-1])

        return {
            "dates": sorted_dates,
            "values": values,
            "benchmark": benchmark,
            "total_return": round((values[-1] / initial - 1) * 100, 2) if values else 0,
            "benchmark_return": round((benchmark[-1] / benchmark[0] - 1) * 100, 2) if len(benchmark) > 1 else 0,
        }
    finally:
        conn.close()


@app.post("/api/data/update")
def update_market_data():
    try:
        from data.downloader import AKShareDataSource
        import akshare as ak

        downloader = AKShareDataSource()
        conn = _get_db_conn()
        updated = []

        for etf in ETF_POOL:
            code = etf["code"]
            try:
                last_row = conn.execute(
                    "SELECT MAX(trade_date) FROM etf_daily_quotes WHERE etf_code=?", (code,)
                ).fetchone()
                start = last_row[0] if last_row and last_row[0] else "20200101"
                start_dt = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)
                start_str = start_dt.strftime("%Y%m%d")

                if start_dt > datetime.now():
                    continue

                symbol = code[2:]
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_str, adjust="qfq")
                if df is None or df.empty:
                    continue

                col_map = {"日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df["etf_code"] = code

                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR REPLACE INTO etf_daily_quotes (etf_code, trade_date, open, high, low, close, volume, amount) VALUES (?,?,?,?,?,?,?,?)",
                        (code, str(row.get("trade_date", "")), float(row.get("open", 0)), float(row.get("high", 0)), float(row.get("low", 0)), float(row.get("close", 0)), float(row.get("volume", 0)), float(row.get("amount", 0)))
                    )

                updated.append(f"{etf['name']}({code}): +{len(df)}条")
            except Exception as e:
                updated.append(f"{etf['name']}({code}): 失败 - {str(e)[:50]}")

        conn.commit()
        conn.close()
        return {"status": "ok", "updated": updated}
    except Exception as e:
        raise HTTPException(500, f"数据更新失败: {str(e)}")


@app.get("/api/etf/prices")
def get_etf_prices():
    conn = _get_db_conn()
    results = []
    for etf in ETF_POOL:
        row = conn.execute(
            "SELECT close, trade_date FROM etf_daily_quotes WHERE etf_code=? ORDER BY trade_date DESC LIMIT 1",
            (etf["code"],)
        ).fetchone()
        if row:
            prev_row = conn.execute(
                "SELECT close FROM etf_daily_quotes WHERE etf_code=? AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
                (etf["code"], row[1])
            ).fetchone()
            prev_close = prev_row[0] if prev_row else row[0]
            change_pct = round((row[0] / prev_close - 1) * 100, 2)
            results.append({
                "code": etf["code"],
                "name": etf["name"],
                "price": round(row[0], 4),
                "date": row[1],
                "change_pct": change_pct,
            })
    conn.close()
    return {"prices": results}


@app.delete("/api/holdings/{symbol}")
def remove_holding(symbol: str, strategy: str = Query("etf_rotation")):
    portfolio_conn = _get_portfolio_conn()
    portfolio_conn.execute("DELETE FROM holdings WHERE symbol=? AND strategy=?", (symbol, strategy))
    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


# ── Backtest Engine ──────────────────────────────────────────────

def _run_etf_backtest(conn, lookback=20, top_k=3, rebalance=20, start_date=None, end_date=None):
    strategy = ETFRotationStrategy(lookback=lookback, top_k=top_k, rebalance_days=rebalance)
    etf_data = strategy._load_etf_data(conn)
    if not etf_data:
        return None

    if start_date or end_date:
        filtered = {}
        for code, df in etf_data.items():
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= df["trade_date"] >= start_date
            if end_date:
                mask &= df["trade_date"] <= end_date
            sub = df.loc[mask].reset_index(drop=True)
            if len(sub) > 50:
                filtered[code] = sub
        etf_data = filtered

    return strategy.run_rotation_backtest(etf_data)


def _run_stock_backtest(conn, ts_code, strategy_name, start_date=None, end_date=None):
    df = pd.read_sql(
        f"SELECT * FROM daily_quotes WHERE ts_code='{ts_code}' ORDER BY trade_date",
        conn
    )
    if start_date:
        df = df[df["trade_date"] >= start_date]
    if end_date:
        df = df[df["trade_date"] <= end_date]
    if len(df) < 60:
        return None

    df["ts_code"] = ts_code

    if strategy_name == "enhanced_chip":
        from strategies.enhanced_chip_strategy import EnhancedChipStrategy
        s = EnhancedChipStrategy()
        signals = s.calculate_signals(df)
        return _signals_to_backtest_result(signals, df, ts_code)
    elif strategy_name == "dual_ma":
        from strategies.benchmark_strategies import DualMAStrategy
        s = DualMAStrategy()
        signals = s.calculate_signals(df)
        return _signals_to_backtest_result(signals, df, ts_code)
    elif strategy_name == "bollinger":
        from strategies.benchmark_strategies import BollingerBandStrategy
        s = BollingerBandStrategy()
        signals = s.calculate_signals(df)
        return _signals_to_backtest_result(signals, df, ts_code)
    elif strategy_name == "rsi":
        from strategies.benchmark_strategies import RSIStrategy
        s = RSIStrategy()
        signals = s.calculate_signals(df)
        return _signals_to_backtest_result(signals, df, ts_code)
    elif strategy_name == "ml_stock":
        from strategies.ml_stock_strategy import run_ml_backtest_for_stock
        return run_ml_backtest_for_stock(conn, ts_code, start_date, end_date)
    return None


def _signals_to_backtest_result(signals, df, ts_code):
    if not signals:
        return None

    capital = 100_000.0
    shares = 0
    entry_price = 0
    portfolio_values = []
    portfolio_dates = []
    trades = []

    date_index = {str(row["trade_date"]): i for i, row in df.iterrows()}

    buy_dates = sorted(set(
        str(s.trade_date) for s in signals if s.action == "buy"
    ))
    sell_dates = sorted(set(
        str(s.trade_date) for s in signals if s.action == "sell"
    ))

    signal_map = {}
    for s in signals:
        d = str(s.trade_date)
        if d not in signal_map:
            signal_map[d] = []
        signal_map[d].append(s)

    for _, row in df.iterrows():
        date = str(row["trade_date"])
        price = float(row["close"])

        if date in signal_map:
            for s in signal_map[date]:
                if s.action == "buy" and shares == 0:
                    exec_p = float(s.price) if s.price else price
                    new_shares = int(capital * 0.95 / exec_p) // 100 * 100
                    if new_shares >= 100:
                        cost = new_shares * exec_p * 1.0003
                        if cost <= capital:
                            capital -= cost
                            shares = new_shares
                            entry_price = exec_p
                            trades.append({"date": date, "action": "buy", "price": exec_p, "shares": shares, "reason": s.reason})

                elif s.action == "sell" and shares > 0:
                    exec_p = float(s.price) if s.price else price
                    revenue = shares * exec_p * (1 - 0.0003 - 0.0005)
                    pnl = (exec_p / entry_price - 1) * 100
                    capital += revenue
                    trades.append({"date": date, "action": "sell", "price": exec_p, "shares": shares, "pnl_pct": round(pnl, 2), "reason": s.reason})
                    shares = 0
                    entry_price = 0

        val = capital + shares * price
        portfolio_values.append(round(val, 2))
        portfolio_dates.append(date)

    if not portfolio_values:
        return None

    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    dd_arr = (pv - peak) / peak
    max_dd = float(dd_arr.min() * 100) if len(dd_arr) > 0 else 0
    total_return = float((pv[-1] / 100_000 - 1) * 100)
    returns = np.diff(pv) / pv[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0

    sell_trades = [t for t in trades if t["action"] == "sell"]
    wins = sum(1 for t in sell_trades if t.get("pnl_pct", 0) > 0)

    name = _get_stock_name_safe(ts_code)
    return {
        "ts_code": ts_code, "name": name,
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "total_trades": len(trades),
        "sell_trades": len(sell_trades),
        "win_rate": round(wins / len(sell_trades) * 100, 1) if sell_trades else 0,
        "final_capital": round(float(pv[-1]), 2),
        "portfolio_dates": portfolio_dates,
        "portfolio_values": [round(float(v), 2) for v in pv],
        "trades": trades[-30:],
    }


def _get_stock_name_safe(ts_code: str) -> str:
    try:
        conn = _get_db_conn()
        row = conn.execute("SELECT name FROM stock_list WHERE ts_code=? LIMIT 1", (ts_code,)).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return ts_code


@app.get("/api/backtest/run")
def run_backtest(
    strategy: str = Query(..., description="Strategy: etf_rotation, enhanced_chip, dual_ma, bollinger, rsi, ml_stock"),
    symbol: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    lookback: int = Query(20),
    top_k: int = Query(3),
    rebalance: int = Query(20),
):
    conn = _get_db_conn()
    try:
        if strategy == "etf_rotation":
            result = _run_etf_backtest(conn, lookback=lookback, top_k=top_k, rebalance=rebalance, start_date=start_date, end_date=end_date)
            if result:
                return {
                    "strategy": strategy,
                    "total_return": result["total_return"],
                    "sharpe": result["sharpe"],
                    "max_drawdown": result["max_drawdown"],
                    "win_rate": result["win_rate"],
                    "total_trades": result["total_trades"],
                    "cagr": result.get("cagr", 0),
                    "portfolio_dates": result.get("portfolio_dates", [])[-500:],
                    "portfolio_values": result.get("portfolio_values", [])[-500:],
                }
            return {"error": "No result"}

        if not symbol:
            raise HTTPException(400, "symbol required for stock strategies")

        result = _run_stock_backtest(conn, symbol, strategy, start_date, end_date)
        if result:
            return {"strategy": strategy, **result}
        return {"error": "Insufficient data or no signals"}
    finally:
        conn.close()


@app.get("/api/backtest/compare")
def compare_strategies(
    symbols: str = Query(..., description="Comma-separated stock codes"),
    start_date: Optional[str] = Query("20220101"),
    end_date: Optional[str] = Query(None),
):
    conn = _get_db_conn()
    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        strategy_names = ["enhanced_chip", "dual_ma", "rsi", "bollinger"]
        results = []

        for sym in symbol_list[:5]:
            for strat in strategy_names:
                r = _run_stock_backtest(conn, sym, strat, start_date, end_date)
                if r:
                    results.append({
                        "symbol": sym,
                        "name": r.get("name", sym),
                        "strategy": strat,
                        "total_return": r["total_return"],
                        "sharpe": r["sharpe"],
                        "max_drawdown": r["max_drawdown"],
                        "win_rate": r.get("win_rate", 0),
                        "total_trades": r.get("total_trades", 0),
                    })

        etf_r = _run_etf_backtest(conn, start_date=start_date, end_date=end_date)
        if etf_r:
            results.append({
                "symbol": "ETF_POOL",
                "name": "ETF轮动组合",
                "strategy": "etf_rotation",
                "total_return": etf_r["total_return"],
                "sharpe": etf_r["sharpe"],
                "max_drawdown": etf_r["max_drawdown"],
                "win_rate": etf_r.get("win_rate", 0),
                "total_trades": etf_r.get("total_trades", 0),
            })

        results.sort(key=lambda x: x["total_return"], reverse=True)
        return {"results": results, "count": len(results)}
    finally:
        conn.close()


@app.get("/api/strategies/picks")
def get_strategy_picks(date: Optional[str] = None):
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
            date = row[0] if row and row[0] else None

        etf_picks = []
        if date:
            strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
            etf_data = strategy._load_etf_data(conn)
            momentum_df = strategy._compute_momentum(etf_data, date) if etf_data else pd.DataFrame()
            if not momentum_df.empty:
                for _, row in momentum_df.iterrows():
                    etf_picks.append({
                        "symbol": row["etf_code"],
                        "name": ETF_CODE_MAP.get(row["etf_code"], row["etf_code"]),
                        "momentum": round(float(row["momentum"]) * 100, 2),
                        "rank": int(row["rank"]),
                        "close": round(float(row["close"]), 4),
                        "in_top_k": int(row["rank"]) <= strategy.top_k,
                    })

        stocks = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code LIMIT 30"
        ).fetchall()]

        chip_buys = []
        chip_sells = []
        for ts_code in stocks:
            df = pd.read_sql(
                f"SELECT * FROM daily_quotes WHERE ts_code='{ts_code}' AND trade_date <= '{date}' ORDER BY trade_date DESC LIMIT 100",
                conn
            )
            if len(df) < 60:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["ts_code"] = ts_code
            try:
                from strategies.enhanced_chip_strategy import EnhancedChipStrategy
                s = EnhancedChipStrategy()
                signals = s.calculate_signals(df)
                if signals:
                    last = signals[-1]
                    if str(last.trade_date) >= (date or "99999999"):
                        entry = {
                            "symbol": ts_code,
                            "name": _get_stock_name_safe(ts_code),
                            "action": last.action,
                            "price": round(float(last.price), 2),
                            "date": str(last.trade_date),
                            "reason": last.reason,
                        }
                        if last.action == "buy":
                            chip_buys.append(entry)
                        elif last.action == "sell":
                            chip_sells.append(entry)
            except Exception:
                pass

        ml_picks = []
        try:
            from strategies.ml_stock_strategy import generate_daily_stock_picks
            ml_picks = generate_daily_stock_picks(conn, date, top_n=10)
        except Exception:
            pass

        return {
            "date": date,
            "etf_rotation": {
                "description": f"ETF轮动 (L{PORTFOLIO_CONFIG['etf_params']['lookback']} K{PORTFOLIO_CONFIG['etf_params']['top_k']} R{PORTFOLIO_CONFIG['etf_params']['rebalance_days']})",
                "picks": etf_picks,
            },
            "enhanced_chip": {
                "description": "增强筹码策略",
                "buy_signals": chip_buys,
                "sell_signals": chip_sells,
            },
            "ml_stock": {
                "description": "ML选股 (HistGradientBoosting)",
                "picks": ml_picks,
            },
        }
    finally:
        conn.close()


@app.get("/api/strategies/etf-picks")
def get_etf_picks(date: Optional[str] = None):
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
            date = row[0] if row and row[0] else None
        etf_picks = []
        if date:
            strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
            etf_data = strategy._load_etf_data(conn)
            momentum_df = strategy._compute_momentum(etf_data, date) if etf_data else pd.DataFrame()
            if not momentum_df.empty:
                for _, row in momentum_df.iterrows():
                    etf_picks.append({
                        "symbol": row["etf_code"],
                        "name": ETF_CODE_MAP.get(row["etf_code"], row["etf_code"]),
                        "momentum": round(float(row["momentum"]) * 100, 2),
                        "rank": int(row["rank"]),
                        "close": round(float(row["close"]), 4),
                        "in_top_k": int(row["rank"]) <= strategy.top_k,
                    })
        return {"date": date, "etf_rotation": {"description": f"ETF轮动 (L{PORTFOLIO_CONFIG['etf_params']['lookback']} K{PORTFOLIO_CONFIG['etf_params']['top_k']} R{PORTFOLIO_CONFIG['etf_params']['rebalance_days']})", "picks": etf_picks}}
    finally:
        conn.close()


@app.get("/api/strategies/chip-picks")
def get_chip_picks(date: Optional[str] = None, refresh: bool = False):
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
            date = row[0] if row and row[0] else None

        if not refresh:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            cached = conn.execute(
                "SELECT data FROM api_cache WHERE cache_key=? AND created_at > datetime('now', '-1 day')",
                (f"chip_picks_{date}",)
            ).fetchone()
            if cached:
                import json as _json
                return _json.loads(cached[0])

        stocks = [r[0] for r in conn.execute(
            "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
        ).fetchall()]

        chip_buys, chip_sells = [], []
        for ts_code in stocks:
            df = pd.read_sql(
                f"SELECT * FROM daily_quotes WHERE ts_code='{ts_code}' AND trade_date <= '{date}' ORDER BY trade_date DESC LIMIT 100",
                conn
            )
            if len(df) < 60:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["ts_code"] = ts_code
            try:
                from strategies.enhanced_chip_strategy import EnhancedChipStrategy
                s = EnhancedChipStrategy()
                signals = s.calculate_signals(df)
                if signals:
                    last = signals[-1]
                    last_date_str = str(last.trade_date)
                    recent_threshold = date[:8] if date else "99999999"
                    if last_date_str >= recent_threshold:
                        entry = {
                            "symbol": ts_code,
                            "name": _get_stock_name_safe(ts_code),
                            "action": last.action,
                            "price": round(float(last.price), 2),
                            "date": last_date_str,
                            "reason": last.reason,
                        }
                        if last.action == "buy":
                            chip_buys.append(entry)
                        elif last.action == "sell":
                            chip_sells.append(entry)
            except Exception:
                pass

        result = {"enhanced_chip": {"description": "增强筹码策略", "buy_signals": chip_buys, "sell_signals": chip_sells}}
        import json as _json
        conn.execute("INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)", (f"chip_picks_{date}", _json.dumps(result, ensure_ascii=False)))
        conn.commit()
        return result
    finally:
        conn.close()


@app.get("/api/strategies/ml-picks")
def get_ml_picks(date: Optional[str] = None, refresh: bool = False):
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
            date = row[0] if row and row[0] else None

        if not refresh:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            cached = conn.execute(
                "SELECT data FROM api_cache WHERE cache_key=? AND created_at > datetime('now', '-1 day')",
                (f"ml_picks_{date}",)
            ).fetchone()
            if cached:
                import json as _json
                return _json.loads(cached[0])

        ml_picks = []
        try:
            from strategies.ml_stock_strategy import generate_daily_stock_picks
            ml_picks = generate_daily_stock_picks(conn, date, top_n=10)
        except Exception:
            pass

        result = {"ml_stock": {"description": "ML选股 (HistGradientBoosting)", "picks": ml_picks}}

        import json as _json
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (f"ml_picks_{date}", _json.dumps(result, ensure_ascii=False))
        )
        conn.commit()

        return result
    finally:
        conn.close()


@app.get("/api/stocks/list")
def list_available_stocks():
    conn = _get_db_conn()
    stocks = conn.execute(
        "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
    ).fetchall()
    conn.close()
    return {"stocks": [{"code": s[0], "name": _get_stock_name_safe(s[0])} for s in stocks]}


# ── Market Regime ────────────────────────────────────────────────

@app.get("/api/regime")
def get_market_regime():
    conn = _get_db_conn()
    try:
        from strategies.market_regime import get_regime_with_params
        regime = get_regime_with_params(conn)
        return {
            "regime": regime.regime,
            "confidence": regime.confidence,
            "adx": regime.adx,
            "volatility_rank": regime.volatility_rank,
            "ma_alignment": regime.ma_alignment,
            "description": regime.description,
            "recommended_params": regime.recommended_params,
        }
    finally:
        conn.close()


# ── Notification ────────────────────────────────────────────────

class NotificationConfigRequest(BaseModel):
    enabled: bool = False
    email_enabled: bool = False
    email_smtp: str = ""
    email_port: int = 465
    email_user: str = ""
    email_pass: str = ""
    email_to: str = ""
    browser_enabled: bool = True
    notify_time: str = "09:25"
    notify_on_signal: bool = True
    notify_on_regime_change: bool = True


@app.get("/api/notifications/config")
def get_notif_config():
    from web.notification import get_notification_config
    return get_notification_config()


@app.post("/api/notifications/config")
def save_notif_config(config: NotificationConfigRequest):
    from web.notification import save_notification_config
    save_notification_config(config.dict())
    return {"status": "ok"}


@app.post("/api/notifications/test")
def test_notification():
    from web.notification import send_email_notification, format_signal_email, get_notification_config
    config = get_notification_config()
    if not config.get("email_enabled"):
        raise HTTPException(400, "Email not enabled")

    html = format_signal_email(
        signals=[{"action": "buy", "name": "测试ETF", "symbol": "test", "reference_price": 1.0, "reason": "测试通知"}],
        portfolio={"total_assets": 1000000, "total_pnl": 0, "total_pnl_pct": 0},
        regime={"regime": "neutral", "description": "测试"},
    )
    ok = send_email_notification("【测试】量化交易系统通知", html, config)
    return {"sent": ok}


@app.post("/api/notifications/send")
def send_daily_notification():
    from web.notification import send_email_notification, format_signal_email, get_notification_config
    config = get_notification_config()
    conn = _get_db_conn()
    try:
        signals = _generate_etf_signals(conn)
        portfolio = _compute_portfolio_metrics(conn)
        from strategies.market_regime import get_regime_with_params
        regime = get_regime_with_params(conn)
        regime_dict = {"regime": regime.regime, "description": regime.description}
    finally:
        conn.close()

    html = format_signal_email(signals, portfolio, regime_dict)
    today = datetime.now().strftime("%Y-%m-%d")
    subject = f"【量化交易】{today} 操作信号"
    ok = send_email_notification(subject, html, config)
    return {"sent": ok, "signals_count": len(signals)}


# ── Data Expansion ──────────────────────────────────────────────

@app.post("/api/data/expand")
def expand_stock_pool(market: str = Query("csi300", description="csi300, csi500, csi800")):
    import akshare as ak
    conn = _get_db_conn()
    added = []
    errors = []

    try:
        if market == "csi300":
            df_index = ak.index_stock_cons_csindex(symbol="000300")
        elif market == "csi500":
            df_index = ak.index_stock_cons_csindex(symbol="000905")
        elif market == "csi800":
            df_index = ak.index_stock_cons_csindex(symbol="000906")
        else:
            raise HTTPException(400, f"Unknown market: {market}")

        codes = df_index["品种代码"].tolist() if "品种代码" in df_index.columns else []
        existing = set(r[0] for r in conn.execute("SELECT DISTINCT ts_code FROM daily_quotes").fetchall())

        new_codes = []
        for code in codes:
            if len(str(code)) == 6:
                if str(code).startswith('6'):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
                if ts_code not in existing:
                    new_codes.append(ts_code)

        new_codes = new_codes[:50]

        for ts_code in new_codes[:20]:
            symbol = ts_code.split('.')[0]
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20200101", adjust="qfq")
                if df is None or len(df) < 50:
                    continue

                col_map = {"日期": "trade_date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                           "成交量": "vol", "成交额": "amount", "涨跌幅": "pct_chg"}
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df["ts_code"] = ts_code

                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR REPLACE INTO daily_quotes (ts_code, trade_date, open, high, low, close, vol, amount, pct_chg) VALUES (?,?,?,?,?,?,?,?,?)",
                        (ts_code, str(row.get("trade_date", "")), float(row.get("open", 0)),
                         float(row.get("high", 0)), float(row.get("low", 0)), float(row.get("close", 0)),
                         float(row.get("vol", 0)), float(row.get("amount", 0)), float(row.get("pct_chg", 0)))
                    )
                added.append(ts_code)
            except Exception as e:
                errors.append(f"{ts_code}: {str(e)[:40]}")

        conn.commit()
        total = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_quotes").fetchone()[0]
        conn.close()
        return {"added": added, "errors": errors, "total_stocks": total}
    except Exception as e:
        conn.close()
        raise HTTPException(500, f"Expansion failed: {str(e)}")


@app.get("/api/data/stats")
def get_data_stats():
    conn = _get_db_conn()
    stock_count = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_quotes").fetchone()[0]
    stock_rows = conn.execute("SELECT COUNT(*) FROM daily_quotes").fetchone()[0]
    stock_range = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes").fetchone()
    etf_count = conn.execute("SELECT COUNT(DISTINCT etf_code) FROM etf_daily_quotes").fetchone()[0]
    etf_rows = conn.execute("SELECT COUNT(*) FROM etf_daily_quotes").fetchone()[0]
    etf_range = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM etf_daily_quotes").fetchone()
    conn.close()
    return {
        "stocks": {"count": stock_count, "rows": stock_rows, "date_range": list(stock_range)},
        "etfs": {"count": etf_count, "rows": etf_rows, "date_range": list(etf_range)},
    }


# ── Static Files ─────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(WEB_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
