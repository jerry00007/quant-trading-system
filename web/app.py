"""实盘交易 Dashboard — FastAPI Backend

提供每日信号、持仓管理、收益追踪等API。
数据流: akshare → SQLite → 策略信号 → REST API → 前端页面
"""

import sqlite3
import sys
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
import io
import csv
import json
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"
PORTFOLIO_DB = PROJECT_ROOT / "data" / "sqlite" / "portfolio.db"
WEB_DIR = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT))

from strategies.etf_rotation_strategy import ETFRotationStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_scheduler = None


def _run_etf_update():
    """定时更新ETF数据（由APScheduler调用）"""
    try:
        from config.scheduler import get_scheduler

        scheduler = get_scheduler()
        results = scheduler.update_etf_data()
        logger.info(f"定时ETF更新完成: {results}")
    except Exception as e:
        logger.exception(f"定时ETF更新失败: {e}")


def _run_stock_update():
    """定时更新股票数据（由APScheduler调用）"""
    try:
        from config.scheduler import get_scheduler

        scheduler = get_scheduler()
        results = scheduler.update_stock_data()
        logger.info(f"定时股票更新完成: {results}")
    except Exception as e:
        logger.exception(f"定时股票更新失败: {e}")


def _run_cache_cleanup():
    try:
        conn = _get_db_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "DELETE FROM api_cache WHERE created_at < datetime('now', '-7 day')"
        )
        conn.commit()
        conn.close()
        logger.info("缓存清理完成")
    except Exception as e:
        logger.exception(f"缓存清理失败: {e}")


def _run_picks_precompute():
    try:
        logger.info("开始预计算策略选股信号...")
        conn = _get_db_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            row = conn.execute("SELECT MAX(trade_date) FROM daily_quotes").fetchone()
            date = row[0] if row and row[0] else None
            if not date:
                logger.warning("无可用的交易日期，跳过预计算")
                return

            feat_df = _compute_stock_features(conn, date)

            NEW_STRATEGIES = [
                "vol_breakout",
                "dragon_first_yin",
                "trend_ma",
                "top_bottom",
                "bollinger_break",
                "rsi_momentum",
                "macd_cross",
            ]
            for strat in NEW_STRATEGIES:
                try:
                    picks = _apply_strategy_filter(strat, feat_df, date, conn)
                    formatted = [
                        {
                            "ts_code": p.get("ts_code", ""),
                            "name": _get_stock_name_safe(p.get("ts_code", "")),
                            "reason": p.get("reason", ""),
                            "strategy": strat,
                        }
                        for p in (picks if isinstance(picks, list) else [])
                    ]
                    cache_key = f"strategy_picks_{strat}_{date}"
                    conn.execute(
                        "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                        (cache_key, json.dumps(formatted, ensure_ascii=False)),
                    )
                    logger.info(f"预计算完成: {strat} ({len(formatted)} picks)")
                except Exception as e:
                    logger.warning(f"预计算策略 {strat} 失败: {e}")

            try:
                stock_signals = _compute_dashboard_stock_signals(conn, feat_df, date)
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                    (
                        f"dashboard_stocks_{date}",
                        json.dumps(stock_signals, ensure_ascii=False),
                    ),
                )
                logger.info(f"预计算dashboard信号完成 ({len(stock_signals)} signals)")
            except Exception as e:
                logger.warning(f"预计算dashboard信号失败: {e}")

            try:
                portfolio_conn = _get_portfolio_conn()
                try:
                    holdings = portfolio_conn.execute(
                        "SELECT symbol, name, shares, entry_price, strategy FROM holdings"
                    ).fetchall()
                finally:
                    portfolio_conn.close()

                stock_holdings = [
                    (s, n, sh, ep, st)
                    for s, n, sh, ep, st in holdings
                    if not s.startswith("sh") and not s.startswith("sz")
                ]
                signals = []
                if feat_df is not None and stock_holdings:
                    day = feat_df[feat_df["trade_date"] == date]
                    for sym, name, shares, entry_price, strategy in stock_holdings:
                        stock_row = day[day["ts_code"] == sym]
                        if stock_row.empty:
                            continue
                        r = stock_row.iloc[0]
                        current_price = float(r["close"])
                        pnl_pct = (
                            (current_price / entry_price - 1) * 100
                            if entry_price > 0
                            else 0
                        )
                        reasons = []
                        rsi_val = r.get("rsi_14")
                        if pd.notna(rsi_val) and float(rsi_val) > 75:
                            reasons.append(
                                f"RSI超买({float(rsi_val):.0f})，注意回调风险"
                            )
                        ma20_val = r.get("ma20")
                        ret20 = r.get("returns_20d")
                        if (
                            pd.notna(ma20_val)
                            and pd.notna(ret20)
                            and float(current_price) < float(ma20_val)
                            and float(ret20) < -0.1
                        ):
                            reasons.append("跌破20日均线，中期趋势转弱")
                        zlcmq_val = r.get("zlcmq")
                        if pd.notna(zlcmq_val) and float(zlcmq_val) > 92:
                            reasons.append(
                                f"筹码极度集中({float(zlcmq_val):.0f})，高位风险"
                            )
                        if pnl_pct < -5:
                            reasons.append(f"浮亏{pnl_pct:.1f}%，建议关注止损")
                        if reasons:
                            signals.append(
                                {
                                    "symbol": sym,
                                    "name": name or _get_stock_name_safe(sym),
                                    "reasons": reasons,
                                    "strategy": strategy,
                                    "current_price": round(current_price, 2),
                                    "pnl_pct": round(pnl_pct, 2),
                                    "entry_price": round(entry_price, 2),
                                }
                            )
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                    (
                        f"sell_signals_{date}",
                        json.dumps({"signals": signals}, ensure_ascii=False),
                    ),
                )
                logger.info(f"预计算卖出信号完成 ({len(signals)} signals)")
            except Exception as e:
                logger.warning(f"预计算卖出信号失败: {e}")

            conn.commit()
            logger.info(f"策略预计算全部完成 (date={date})")
        finally:
            conn.close()
    except Exception as e:
        logger.exception(f"预计算任务失败: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from config.scheduler import get_scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.start()
    logger.info("定时调度器已启动")

    # 注册每日定时任务：16:30更新ETF数据，17:00更新股票数据
    _scheduler.add_job(
        _run_etf_update,
        CronTrigger(hour=16, minute=30),
        id="daily_etf_update",
        name="每日ETF数据更新",
        replace_existing=True,
        misfire_grace_time=3600,  # 允许1小时延迟（如果服务未运行）
    )
    _scheduler.add_job(
        _run_stock_update,
        CronTrigger(hour=17, minute=0),
        id="daily_stock_update",
        name="每日股票数据更新",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_picks_precompute,
        CronTrigger(hour=17, minute=30),
        id="daily_picks_precompute",
        name="每日策略信号预计算",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_cache_cleanup,
        CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="weekly_cache_cleanup",
        name="每周缓存清理",
        replace_existing=True,
    )
    logger.info(f"已注册定时任务: {[job.name for job in _scheduler.get_jobs()]}")

    yield

    if _scheduler:
        _scheduler.shutdown()
        logger.info("定时调度器已关闭")


app = FastAPI(title="量化交易系统", version="1.0.0", lifespan=lifespan)

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
    "etf_params": {
        "lookback": 20,
        "top_k": 3,
        "rebalance_days": 20,
        "stop_loss": 0.08,
        "take_profit": 0.30,
    },
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


def _get_cash_balance(portfolio_conn) -> float:
    """Get actual cash balance from ledger."""
    row = portfolio_conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM cash_ledger"
    ).fetchone()
    return PORTFOLIO_CONFIG["initial_capital"] + (row[0] if row else 0)


# ── Signal Generation ────────────────────────────────────────────


def _get_latest_trading_date(conn) -> Optional[str]:
    row = conn.execute("SELECT MAX(trade_date) FROM etf_daily_quotes").fetchone()
    return row[0] if row and row[0] else None


def _generate_etf_signals(conn, as_of_date: str = None) -> list:
    strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
    etf_data = strategy._load_etf_data(conn)
    if not etf_data:
        return []

    all_dates = sorted(
        set(d for df in etf_data.values() for d in df["trade_date"].tolist())
    )
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
    try:
        current_holdings_rows = portfolio_conn.execute(
            "SELECT symbol, shares FROM holdings WHERE strategy='etf_rotation'"
        ).fetchall()
    finally:
        portfolio_conn.close()

    current_holdings = {row[0]: row[1] for row in current_holdings_rows}

    # Calculate available cash for ETF allocation
    _pc = _get_portfolio_conn()
    try:
        actual_cash = _get_cash_balance(_pc)
    finally:
        _pc.close()
    etf_budget = min(
        PORTFOLIO_CONFIG["initial_capital"] * PORTFOLIO_CONFIG["etf_allocation"],
        actual_cash * 0.8,
    )
    etf_budget = max(etf_budget, 0)
    existing_etf_value = 0.0
    for sym, shrs in current_holdings.items():
        price_row = conn.execute(
            "SELECT close FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
            (sym, as_of_date),
        ).fetchone()
        if price_row:
            existing_etf_value += shrs * price_row[0]
    available_etf_cash = etf_budget - existing_etf_value

    next_date_idx = (
        all_dates.index(as_of_date) + 1 if as_of_date in all_dates else len(all_dates)
    )
    execute_date = (
        all_dates[next_date_idx] if next_date_idx < len(all_dates) else as_of_date
    )

    # Count new buy signals to evenly distribute budget
    new_buys = [code for code in top_etfs if code not in current_holdings]
    per_buy_budget = (available_etf_cash / len(new_buys)) if new_buys else 0

    signals = []

    for code in list(current_holdings.keys()):
        if code not in top_etfs:
            ref_price_row = conn.execute(
                "SELECT open FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
                (code, as_of_date),
            ).fetchone()
            ref_price = ref_price_row[0] if ref_price_row else 0
            signals.append(
                {
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
                }
            )

    for code in top_etfs:
        if code not in current_holdings:
            rank_row = momentum_df.loc[momentum_df["etf_code"] == code]
            rank_val = int(rank_row["rank"].values[0]) if len(rank_row) > 0 else 0
            ref_price_row = conn.execute(
                "SELECT open FROM etf_daily_quotes WHERE etf_code=? AND trade_date=?",
                (code, as_of_date),
            ).fetchone()
            ref_price = ref_price_row[0] if ref_price_row else 0
            # Calculate suggested shares (100-share lots, ETF minimum)
            suggested_shares = (
                int(per_buy_budget / ref_price / 100) * 100 if ref_price > 0 else 0
            )
            signals.append(
                {
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
                }
            )

    # Check for rebalance day (how many trading days since last rebalance)
    signals.sort(key=lambda x: 0 if x["action"] == "sell" else 1)
    return signals


def _compute_portfolio_metrics(conn) -> dict:
    portfolio_conn = _get_portfolio_conn()
    try:
        holdings = portfolio_conn.execute(
            "SELECT symbol, name, shares, entry_price, entry_date, strategy FROM holdings"
        ).fetchall()
        trades = portfolio_conn.execute(
            "SELECT symbol, action, shares, price, date, reason, strategy FROM trades ORDER BY date DESC"
        ).fetchall()

        cash_row = portfolio_conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM cash_ledger"
        ).fetchone()
        initial = PORTFOLIO_CONFIG["initial_capital"]
        cash = initial + cash_row[0]
    finally:
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
            (symbol,),
        ).fetchone()
        current_price = price_row[0] if price_row else entry_price
        market_value = shares * current_price
        total_market_value += market_value
        pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

        holdings_detail.append(
            {
                "symbol": symbol,
                "name": name or ETF_CODE_MAP.get(symbol, symbol),
                "shares": shares,
                "entry_price": round(entry_price, 4),
                "current_price": round(current_price, 4),
                "market_value": round(market_value, 2),
                "pnl_pct": round(pnl_pct, 2),
                "entry_date": entry_date,
                "strategy": strategy or "manual",
            }
        )

    total_assets = cash + total_market_value
    total_pnl = total_assets - initial
    total_pnl_pct = (total_pnl / initial) * 100 if initial > 0 else 0

    etf_value = sum(
        h["market_value"] for h in holdings_detail if h["strategy"] == "etf_rotation"
    )
    stock_value = sum(
        h["market_value"]
        for h in holdings_detail
        if h["strategy"] != "etf_rotation" and h["strategy"] != "manual"
    )
    manual_value = sum(
        h["market_value"] for h in holdings_detail if h["strategy"] == "manual"
    )

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
            "etf_pct": round(etf_value / total_assets * 100, 1)
            if total_assets > 0
            else 0,
            "stock": round(stock_value, 2),
            "stock_pct": round(stock_value / total_assets * 100, 1)
            if total_assets > 0
            else 0,
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
    action: Literal["buy", "sell"]
    shares: int = Field(..., gt=0, description="Must be positive")
    price: float = Field(..., gt=0, description="Must be positive")
    date: str
    reason: str = ""
    strategy: str = "manual"
    name: str = ""


class SignalActionRequest(BaseModel):
    signal_id: int
    action: Literal["executed", "skipped"]
    actual_price: Optional[float] = Field(None, gt=0)
    actual_shares: Optional[int] = Field(None, gt=0)


class CashOperationRequest(BaseModel):
    amount: float
    operation: str  # "deposit", "withdraw"
    reason: str = ""


# ── API Endpoints ────────────────────────────────────────────────


@app.get("/api/dashboard")
def get_dashboard(refresh: bool = False):
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
                "action": row[4],
                "shares": row[5],
                "reference_price": row[6],
                "reason": row[7],
                "strategy": row[8],
                "status": row[9],
            }
            for row in pending
        ]

        stock_signals = []
        try:
            stock_date_row = conn.execute(
                "SELECT MAX(trade_date) FROM daily_quotes"
            ).fetchone()
            stock_date = (
                stock_date_row[0]
                if stock_date_row and stock_date_row[0]
                else latest_date
            )
            if stock_date:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        cache_key TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TEXT DEFAULT (datetime('now'))
                    )
                """)
                cache_key = f"dashboard_stocks_{stock_date}"
                if not refresh:
                    cached = conn.execute(
                        "SELECT data FROM api_cache WHERE cache_key=? AND created_at > datetime('now', '-1 day')",
                        (cache_key,),
                    ).fetchone()
                    if cached:
                        stock_signals = json.loads(cached[0])
                        return {
                            "portfolio": portfolio,
                            "pending_signals": pending_signals,
                            "fresh_signals": signals,
                            "stock_signals": stock_signals,
                            "config": PORTFOLIO_CONFIG,
                            "etf_pool": ETF_POOL,
                        }

                feat_df = _compute_stock_features(conn, stock_date)
                stock_signals = _compute_dashboard_stock_signals(
                    conn, feat_df, stock_date
                )
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                    (cache_key, json.dumps(stock_signals, ensure_ascii=False)),
                )
                conn.commit()
        except Exception as e:
            logger.exception(f"Dashboard股票信号计算失败: {e}")

        return {
            "portfolio": portfolio,
            "pending_signals": pending_signals,
            "fresh_signals": signals,
            "stock_signals": stock_signals[:10],
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
            existing = portfolio_conn.execute(
                "SELECT id FROM signals WHERE signal_date=? AND symbol=? AND strategy=? AND status='pending'",
                (s["signal_date"], s["symbol"], s["strategy"]),
            ).fetchone()
            if not existing:
                portfolio_conn.execute(
                    "INSERT INTO signals (signal_date, execute_date, symbol, name, action, shares, reference_price, reason, strategy, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        s["signal_date"],
                        s["execute_date"],
                        s["symbol"],
                        s["name"],
                        s["action"],
                        s["shares"],
                        s["reference_price"],
                        s["reason"],
                        s["strategy"],
                        s["status"],
                    ),
                )
        portfolio_conn.commit()
        portfolio_conn.close()
        return {"signals": signals, "count": len(signals)}
    finally:
        conn.close()


@app.post("/api/signals/confirm")
def confirm_signal(req: SignalActionRequest):
    portfolio_conn = _get_portfolio_conn()
    row = portfolio_conn.execute(
        "SELECT symbol, action, shares, reference_price, reason, strategy, name FROM signals WHERE id=?",
        (req.signal_id,),
    ).fetchone()
    if not row:
        portfolio_conn.close()
        raise HTTPException(404, "Signal not found")

    symbol, action, shares, ref_price, reason, strategy, name = row
    actual_price = req.actual_price or ref_price
    actual_shares = req.actual_shares or shares

    if req.action == "executed":
        cursor = portfolio_conn.execute(
            "UPDATE signals SET status='executed' WHERE id=? AND status='pending'",
            (req.signal_id,),
        )
        if cursor.rowcount == 0:
            portfolio_conn.close()
            raise HTTPException(409, "Signal already processed or not found")

        if action == "buy":
            cost = actual_shares * actual_price
            current_cash = _get_cash_balance(portfolio_conn)
            if current_cash < cost:
                portfolio_conn.close()
                raise HTTPException(
                    400, f"Insufficient cash: need {cost:.2f}, have {current_cash:.2f}"
                )
            portfolio_conn.execute(
                "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'withdraw',date('now'),?)",
                (-(actual_shares * actual_price), f"买入 {symbol}"),
            )
            existing = portfolio_conn.execute(
                "SELECT shares, entry_price FROM holdings WHERE symbol=? AND strategy=?",
                (symbol, strategy),
            ).fetchone()
            if existing:
                old_shares, old_price = existing
                new_shares = old_shares + actual_shares
                new_entry = (
                    old_shares * old_price + actual_shares * actual_price
                ) / new_shares
                portfolio_conn.execute(
                    "UPDATE holdings SET shares=?, entry_price=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?",
                    (new_shares, new_entry, symbol, strategy),
                )
            else:
                portfolio_conn.execute(
                    "INSERT INTO holdings (symbol, name, shares, entry_price, entry_date, strategy) VALUES (?,?,?,?,date('now'),?)",
                    (
                        symbol,
                        name or ETF_CODE_MAP.get(symbol, symbol),
                        actual_shares,
                        actual_price,
                        strategy,
                    ),
                )

        elif action == "sell":
            portfolio_conn.execute(
                "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,date('now'),?,?)",
                (
                    symbol,
                    name or ETF_CODE_MAP.get(symbol, symbol),
                    "sell",
                    actual_shares,
                    actual_price,
                    reason,
                    strategy,
                ),
            )
            portfolio_conn.execute(
                "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'deposit',date('now'),?)",
                (actual_shares * actual_price, f"卖出 {symbol}"),
            )
            existing = portfolio_conn.execute(
                "SELECT shares FROM holdings WHERE symbol=? AND strategy=?",
                (symbol, strategy),
            ).fetchone()
            if not existing:
                portfolio_conn.close()
                raise HTTPException(400, f"No holdings found for {symbol}")
            if actual_shares > existing[0]:
                actual_shares = existing[0]
            remaining = existing[0] - actual_shares
            if remaining <= 0:
                portfolio_conn.execute(
                    "DELETE FROM holdings WHERE symbol=? AND strategy=?",
                    (symbol, strategy),
                )
            else:
                portfolio_conn.execute(
                    "UPDATE holdings SET shares=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?",
                    (remaining, symbol, strategy),
                )

    elif req.action == "skipped":
        portfolio_conn.execute(
            "UPDATE signals SET status='skipped' WHERE id=?", (req.signal_id,)
        )

    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


@app.post("/api/trades")
def manual_trade(req: TradeConfirmRequest):
    portfolio_conn = _get_portfolio_conn()
    name = req.name or ETF_CODE_MAP.get(req.symbol, req.symbol)

    portfolio_conn.execute(
        "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,?,?,?)",
        (
            req.symbol,
            name,
            req.action,
            req.shares,
            req.price,
            req.date,
            req.reason,
            req.strategy,
        ),
    )

    if req.action == "buy":
        cost = req.shares * req.price
        current_cash = _get_cash_balance(portfolio_conn)
        if current_cash < cost:
            portfolio_conn.close()
            raise HTTPException(
                400, f"Insufficient cash: need {cost:.2f}, have {current_cash:.2f}"
            )
        portfolio_conn.execute(
            "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'withdraw',?,?)",
            (-(req.shares * req.price), req.date, f"手动买入 {req.symbol}"),
        )
        existing = portfolio_conn.execute(
            "SELECT shares, entry_price FROM holdings WHERE symbol=? AND strategy=?",
            (req.symbol, req.strategy),
        ).fetchone()
        if existing:
            old_shares, old_price = existing
            new_shares = old_shares + req.shares
            new_entry = (old_shares * old_price + req.shares * req.price) / new_shares
            portfolio_conn.execute(
                "UPDATE holdings SET shares=?, entry_price=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?",
                (new_shares, new_entry, req.symbol, req.strategy),
            )
        else:
            portfolio_conn.execute(
                "INSERT INTO holdings (symbol, name, shares, entry_price, entry_date, strategy) VALUES (?,?,?,?,?,?)",
                (req.symbol, name, req.shares, req.price, req.date, req.strategy),
            )
    elif req.action == "sell":
        portfolio_conn.execute(
            "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'deposit',?,?)",
            (req.shares * req.price, req.date, f"手动卖出 {req.symbol}"),
        )
        existing = portfolio_conn.execute(
            "SELECT shares FROM holdings WHERE symbol=? AND strategy=?",
            (req.symbol, req.strategy),
        ).fetchone()
        if not existing:
            portfolio_conn.close()
            raise HTTPException(400, f"No holdings found for {req.symbol}")
        if req.shares > existing[0]:
            req.shares = existing[0]
        remaining = existing[0] - req.shares
        if remaining <= 0:
            portfolio_conn.execute(
                "DELETE FROM holdings WHERE symbol=? AND strategy=?",
                (req.symbol, req.strategy),
            )
        else:
            portfolio_conn.execute(
                "UPDATE holdings SET shares=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?",
                (remaining, req.symbol, req.strategy),
            )

    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


class PickBuyRequest(BaseModel):
    symbol: str
    name: str = ""
    price: float = Field(..., gt=0)
    shares: int = Field(default=100, gt=0)
    strategy: str = ""
    reason: str = ""


@app.post("/api/strategies/picks/buy")
def buy_from_picks(req: PickBuyRequest):
    portfolio_conn = _get_portfolio_conn()
    name = req.name or _get_stock_name_safe(req.symbol)
    today_str = datetime.now().strftime("%Y%m%d")

    portfolio_conn.execute(
        "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,?,?,?,?)",
        (
            req.symbol,
            name,
            "buy",
            req.shares,
            req.price,
            today_str,
            req.reason,
            req.strategy,
        ),
    )

    cost = req.shares * req.price
    current_cash = _get_cash_balance(portfolio_conn)
    if current_cash < cost:
        portfolio_conn.close()
        raise HTTPException(
            400, f"Insufficient cash: need {cost:.2f}, have {current_cash:.2f}"
        )
    portfolio_conn.execute(
        "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'withdraw',?,?)",
        (-(req.shares * req.price), today_str, f"策略买入 {req.symbol}"),
    )
    existing = portfolio_conn.execute(
        "SELECT shares, entry_price FROM holdings WHERE symbol=? AND strategy=?",
        (req.symbol, req.strategy),
    ).fetchone()
    if existing:
        old_shares, old_price = existing
        new_shares = old_shares + req.shares
        new_entry = (old_shares * old_price + req.shares * req.price) / new_shares
        portfolio_conn.execute(
            "UPDATE holdings SET shares=?, entry_price=?, updated_at=datetime('now') WHERE symbol=? AND strategy=?",
            (new_shares, new_entry, req.symbol, req.strategy),
        )
    else:
        portfolio_conn.execute(
            "INSERT INTO holdings (symbol, name, shares, entry_price, entry_date, strategy) VALUES (?,?,?,?,?,?)",
            (req.symbol, name, req.shares, req.price, today_str, req.strategy),
        )

    portfolio_conn.commit()
    portfolio_conn.close()
    return {"status": "ok"}


@app.get("/api/history")
def get_trade_history(limit: int = Query(50, ge=1, le=200)):
    portfolio_conn = _get_portfolio_conn()
    trades = portfolio_conn.execute(
        "SELECT symbol, name, action, shares, price, date, reason, strategy FROM trades ORDER BY date DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    portfolio_conn.close()
    return {
        "trades": [
            {
                "symbol": t[0],
                "name": t[1],
                "action": t[2],
                "shares": t[3],
                "price": round(t[4], 4),
                "date": t[5],
                "reason": t[6],
                "strategy": t[7],
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
                (prev_d, curr_d),
            ).fetchone()
            if idx_row:
                ret = idx_row[0]
                prev_idx_row = conn.execute(
                    "SELECT close FROM etf_daily_quotes WHERE etf_code='sz159919' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
                    (prev_d,),
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
            "benchmark_return": round((benchmark[-1] / benchmark[0] - 1) * 100, 2)
            if len(benchmark) > 1
            else 0,
        }
    finally:
        conn.close()


@app.post("/api/data/update")
def update_market_data(
    update_stocks: bool = Query(False, description="是否同时更新股票数据"),
):
    try:
        from config.scheduler import get_scheduler

        scheduler = get_scheduler()
        etf_results = scheduler.update_etf_data()

        result = {"etf_updated": etf_results}

        if update_stocks:
            stock_results = scheduler.update_stock_data()
            result["stock_updated"] = stock_results

        return result
    except Exception as e:
        logger.exception(f"数据更新失败: {e}")
        raise HTTPException(500, f"数据更新失败: {str(e)}")


@app.post("/api/data/update-full")
def update_full_data():
    try:
        from config.scheduler import get_scheduler

        scheduler = get_scheduler()
        etf_results = scheduler.update_etf_data()
        stock_results = scheduler.update_stock_data()

        return {
            "status": "ok",
            "etf_updated": etf_results,
            "stock_updated": stock_results,
        }
    except Exception as e:
        logger.exception(f"全量数据更新失败: {e}")
        raise HTTPException(500, f"全量数据更新失败: {str(e)}")


@app.get("/api/etf/prices")
def get_etf_prices():
    conn = _get_db_conn()
    try:
        results = []
        for etf in ETF_POOL:
            row = conn.execute(
                "SELECT close, trade_date FROM etf_daily_quotes WHERE etf_code=? ORDER BY trade_date DESC LIMIT 1",
                (etf["code"],),
            ).fetchone()
            if row:
                prev_row = conn.execute(
                    "SELECT close FROM etf_daily_quotes WHERE etf_code=? AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
                    (etf["code"], row[1]),
                ).fetchone()
                prev_close = prev_row[0] if prev_row else row[0]
                change_pct = round((row[0] / prev_close - 1) * 100, 2)
                results.append(
                    {
                        "code": etf["code"],
                        "name": etf["name"],
                        "price": round(row[0], 4),
                        "date": row[1],
                        "change_pct": change_pct,
                    }
                )
        return {"prices": results}
    finally:
        conn.close()


@app.delete("/api/holdings/{symbol}")
def remove_holding(symbol: str, strategy: str = Query("etf_rotation")):
    portfolio_conn = _get_portfolio_conn()
    try:
        holding = portfolio_conn.execute(
            "SELECT shares, entry_price, name FROM holdings WHERE symbol=? AND strategy=?",
            (symbol, strategy),
        ).fetchone()
        if not holding:
            raise HTTPException(404, f"Holding not found: {symbol}")

        shares, entry_price, name = holding
        market_conn = _get_db_conn()
        try:
            if symbol.startswith("sh") or symbol.startswith("sz"):
                price_row = market_conn.execute(
                    "SELECT close FROM etf_daily_quotes WHERE etf_code=? ORDER BY trade_date DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
            else:
                price_row = market_conn.execute(
                    "SELECT close FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
        finally:
            market_conn.close()

        sell_price = price_row[0] if price_row else entry_price

        portfolio_conn.execute(
            "INSERT INTO trades (symbol, name, action, shares, price, date, reason, strategy) VALUES (?,?,?,?,date('now'),?,?,?)",
            (
                symbol,
                name or ETF_CODE_MAP.get(symbol, symbol),
                "sell",
                shares,
                sell_price,
                f"清仓卖出",
                strategy,
            ),
        )
        portfolio_conn.execute(
            "INSERT INTO cash_ledger (amount, operation, date, reason) VALUES (?,'deposit',date('now'),?)",
            (shares * sell_price, f"清仓卖出 {symbol}"),
        )
        portfolio_conn.execute(
            "DELETE FROM holdings WHERE symbol=? AND strategy=?", (symbol, strategy)
        )
        portfolio_conn.commit()
        return {"status": "ok", "sold_shares": shares, "sold_price": sell_price}
    finally:
        portfolio_conn.close()


@app.get("/api/holdings/sell-signals")
def get_holdings_sell_signals(refresh: bool = False):
    portfolio_conn = _get_portfolio_conn()
    try:
        holdings = portfolio_conn.execute(
            "SELECT symbol, name, shares, entry_price, strategy FROM holdings"
        ).fetchall()
    finally:
        portfolio_conn.close()

    if not holdings:
        return {"signals": []}

    stock_holdings = [
        (sym, name, shares, entry_price, strategy)
        for sym, name, shares, entry_price, strategy in holdings
        if not sym.startswith("sh") and not sym.startswith("sz")
    ]

    if not stock_holdings:
        return {"signals": []}

    conn = _get_db_conn()
    try:
        row = conn.execute("SELECT MAX(trade_date) FROM daily_quotes").fetchone()
        latest_date = row[0] if row and row[0] else None
        if not latest_date:
            return {"signals": []}

        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        sell_cache_key = f"sell_signals_{latest_date}"
        if not refresh:
            cached = conn.execute(
                "SELECT data FROM api_cache WHERE cache_key=? AND created_at > datetime('now', '-1 day')",
                (sell_cache_key,),
            ).fetchone()
            if cached:
                return json.loads(cached[0])

        feat_df = _compute_stock_features(conn, latest_date)
        if feat_df is None:
            return {"signals": []}

        day = feat_df[feat_df["trade_date"] == latest_date]

        signals = []
        for sym, name, shares, entry_price, strategy in stock_holdings:
            stock_row = day[day["ts_code"] == sym]
            if stock_row.empty:
                continue

            r = stock_row.iloc[0]
            current_price = float(r["close"])
            pnl_pct = (current_price / entry_price - 1) * 100 if entry_price > 0 else 0

            reasons = []

            rsi_val = r.get("rsi_14")
            if pd.notna(rsi_val) and float(rsi_val) > 75:
                reasons.append(f"RSI超买({float(rsi_val):.0f})，注意回调风险")

            ma20_val = r.get("ma20")
            ret20 = r.get("returns_20d")
            if (
                pd.notna(ma20_val)
                and pd.notna(ret20)
                and float(current_price) < float(ma20_val)
                and float(ret20) < -0.1
            ):
                reasons.append("跌破20日均线，中期趋势转弱")

            zlcmq_val = r.get("zlcmq")
            if pd.notna(zlcmq_val) and float(zlcmq_val) > 92:
                reasons.append(f"筹码极度集中({float(zlcmq_val):.0f})，高位风险")

            if pnl_pct < -5:
                reasons.append(f"浮亏{pnl_pct:.1f}%，建议关注止损")

            if reasons:
                signals.append(
                    {
                        "symbol": sym,
                        "name": name or _get_stock_name_safe(sym),
                        "reasons": reasons,
                        "strategy": strategy,
                        "current_price": round(current_price, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "entry_price": round(entry_price, 2),
                    }
                )

        result = {"signals": signals}
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (sell_cache_key, json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return result
    except Exception as e:
        logger.exception(f"获取卖出信号失败: {e}")
        raise HTTPException(500, f"获取卖出信号失败: {str(e)}")
    finally:
        conn.close()


def _compute_dashboard_stock_signals(conn, feat_df, stock_date):
    """Compute stock signals for dashboard (chip + ML).

    Extracted so it can be called from both the dashboard endpoint and the
    daily precompute job.  Returns a list of signal dicts (max 10).
    """
    stock_signals = []
    if feat_df is None or not stock_date:
        return stock_signals

    day = feat_df[feat_df["trade_date"] == stock_date]
    day = day[day["cumcount"] >= 74]

    # ── Chip signals ──
    cur_z = day["zlcmq"].values
    valid = (~np.isnan(cur_z)) & (cur_z >= 60) & (cur_z <= 92)
    chip_day = day[valid]
    if not chip_day.empty:
        z5_max = chip_day["zlcmq_max5"].values
        z_prev = chip_day["zlcmq_prev1"].values
        cur_close = chip_day["close"].values
        cur_open = chip_day["open"].values
        prev_close = np.where(
            chip_day["returns_1d"].isna(),
            cur_close,
            cur_close / (1.0 + chip_day["returns_1d"].fillna(0).values),
        )
        was_high = z5_max >= 85
        declining = cur_z[valid] < z_prev
        not_fast = (z5_max - cur_z[valid]) <= 35
        stable = (cur_close >= cur_open) | (cur_close > prev_close)
        chip_passed = chip_day[was_high & declining & not_fast & stable].copy()
        if not chip_passed.empty:
            chip_passed = chip_passed.sort_values("zlcmq", ascending=False)
            seen = set()
            for _, r in chip_passed.head(5).iterrows():
                ts = r["ts_code"]
                if ts not in seen:
                    seen.add(ts)
                    stock_signals.append(
                        {
                            "symbol": ts,
                            "name": _get_stock_name_safe(ts),
                            "action": "buy",
                            "price": round(float(r["close"]), 2),
                            "reason": f"筹码高位回落企稳 ZLCMQ={round(float(r['zlcmq']), 1)}",
                            "strategy": "chip",
                            "source": "筹码策略",
                        }
                    )

    # ── ML signals ──
    FEATURE_COLS = [
        "returns_20d",
        "volatility_20d",
        "returns_5d",
        "volatility_5d",
        "price_to_ma20",
        "ma5_to_ma20",
        "vol_to_vol_ma",
        "price_to_high20",
        "volume_ratio",
        "rsi_14",
    ]
    pool = feat_df[feat_df["cumcount"] >= 49].dropna(subset=FEATURE_COLS)
    if pool.shape[0] >= 100:
        pool["future_close_5d"] = pool.groupby("ts_code")["close"].transform(
            lambda s: s.shift(-5)
        )
        pool["label_5d_up"] = (pool["future_close_5d"] > pool["close"]).astype(
            np.float64
        )
        labels = pool["label_5d_up"].values
        has_label = ~np.isnan(labels)
        X_all = pool[FEATURE_COLS].values
        y_all = labels
        X_train = X_all[has_label]
        y_train = np.nan_to_num(y_all[has_label], nan=0.0)
        if len(X_train) >= 50 and len(np.unique(y_train)) >= 2:
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier

                model = HistGradientBoostingClassifier(
                    max_iter=50,
                    max_depth=3,
                    learning_rate=0.1,
                    min_samples_leaf=10,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                ml_day = day.dropna(subset=FEATURE_COLS)
                if not ml_day.empty:
                    X_score = ml_day[FEATURE_COLS].values
                    proba = model.predict_proba(X_score)
                    prob_up = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    ml_passed = ml_day[prob_up > 0.45].copy()
                    if not ml_passed.empty:
                        ml_passed["prob_up"] = prob_up[prob_up > 0.45]
                        ml_passed = ml_passed.sort_values("prob_up", ascending=False)
                        seen_codes = {s["symbol"] for s in stock_signals}
                        for _, r in ml_passed.head(5).iterrows():
                            ts = r["ts_code"]
                            if ts not in seen_codes:
                                seen_codes.add(ts)
                                stock_signals.append(
                                    {
                                        "symbol": ts,
                                        "name": _get_stock_name_safe(ts),
                                        "action": "buy",
                                        "price": round(float(r["close"]), 2),
                                        "reason": f"ML预测上涨概率{round(float(r['prob_up']) * 100)}%",
                                        "strategy": "ml",
                                        "source": "ML策略",
                                    }
                                )
            except Exception:
                pass

    return stock_signals[:10]


def _get_stock_name_safe(ts_code: str) -> str:
    try:
        conn = _get_db_conn()
        row = conn.execute(
            "SELECT name FROM stock_list WHERE ts_code=? LIMIT 1", (ts_code,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return ts_code


@app.get("/api/strategies/picks")
def get_strategy_picks(date: Optional[str] = None):
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None

        etf_picks = []
        if date:
            strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
            etf_data = strategy._load_etf_data(conn)
            momentum_df = (
                strategy._compute_momentum(etf_data, date)
                if etf_data
                else pd.DataFrame()
            )
            if not momentum_df.empty:
                for _, row in momentum_df.iterrows():
                    etf_picks.append(
                        {
                            "symbol": row["etf_code"],
                            "name": ETF_CODE_MAP.get(row["etf_code"], row["etf_code"]),
                            "momentum": round(float(row["momentum"]) * 100, 2),
                            "rank": int(row["rank"]),
                            "close": round(float(row["close"]), 4),
                            "in_top_k": int(row["rank"]) <= strategy.top_k,
                        }
                    )

        stocks = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
            ).fetchall()
        ]

        chip_buys = []
        chip_sells = []
        for ts_code in stocks:
            df = pd.read_sql(
                "SELECT * FROM daily_quotes WHERE ts_code=? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 100",
                conn,
                params=[ts_code, date],
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
            except Exception as e:
                logger.exception(f"Strategy failed: {e}")

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
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None
        etf_picks = []
        if date:
            strategy = ETFRotationStrategy(**PORTFOLIO_CONFIG["etf_params"])
            etf_data = strategy._load_etf_data(conn)
            momentum_df = (
                strategy._compute_momentum(etf_data, date)
                if etf_data
                else pd.DataFrame()
            )
            if not momentum_df.empty:
                for _, row in momentum_df.iterrows():
                    etf_picks.append(
                        {
                            "symbol": row["etf_code"],
                            "name": ETF_CODE_MAP.get(row["etf_code"], row["etf_code"]),
                            "momentum": round(float(row["momentum"]) * 100, 2),
                            "rank": int(row["rank"]),
                            "close": round(float(row["close"]), 4),
                            "in_top_k": int(row["rank"]) <= strategy.top_k,
                        }
                    )
        return {
            "date": date,
            "etf_rotation": {
                "description": f"ETF轮动 (L{PORTFOLIO_CONFIG['etf_params']['lookback']} K{PORTFOLIO_CONFIG['etf_params']['top_k']} R{PORTFOLIO_CONFIG['etf_params']['rebalance_days']})",
                "picks": etf_picks,
            },
        }
    finally:
        conn.close()


def _tdx_sma_chip(x, n, m):
    """TDX-style SMA used for ZLCMQ calculation."""
    y = np.empty(len(x))
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (m * x[i] + (n - m) * y[i - 1]) / n
    return y


def _compute_zlcmq_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ZLCMQ + zlcmq_max5 + zlcmq_prev1 for all stocks in df.

    df must be sorted by (ts_code, trade_date) and have columns:
    ts_code, trade_date, close, high, low
    """
    g = df.groupby("ts_code", sort=False)
    df["low_75"] = g["low"].transform(lambda s: s.rolling(75, min_periods=1).min())
    df["high_75"] = g["high"].transform(lambda s: s.rolling(75, min_periods=1).max())

    var7_raw = np.where(
        (df["high_75"] - df["low_75"]) / 100.0 > 1e-10,
        (df["close"] - df["low_75"]) / ((df["high_75"] - df["low_75"]) / 100.0),
        0.0,
    )
    var7_raw = np.nan_to_num(var7_raw, nan=0.0)

    codes = df["ts_code"].values
    boundaries = np.where(codes[:-1] != codes[1:])[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(codes)]])
    zlcmq_arr = np.empty(len(df), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        seg = var7_raw[start:end]
        v8 = _tdx_sma_chip(seg, 20, 1)
        v8s = _tdx_sma_chip(v8, 15, 1)
        vara = 3.0 * v8 - 2.0 * v8s
        zlcmq_arr[start:end] = 100.0 - vara

    df["zlcmq"] = zlcmq_arr
    df["zlcmq_max5"] = g["zlcmq"].transform(lambda s: s.rolling(5, min_periods=1).max())
    df["zlcmq_prev1"] = g["zlcmq"].transform(lambda s: s.shift(1))
    return df


def _compute_stock_features(conn, date):
    """Compute ALL stock features needed by ALL strategies in one vectorized pass.

    Loads 280 trading days of data (enough for top_bottom which needs 250+).
    Returns a DataFrame with all features, or None if insufficient data.
    """
    trading_dates_rows = conn.execute(
        "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 280",
        (date,),
    ).fetchall()
    if len(trading_dates_rows) < 30:
        return None

    min_date = trading_dates_rows[-1][0]

    df = pd.read_sql(
        "SELECT ts_code, trade_date, open, high, low, close, vol, pct_chg FROM daily_quotes WHERE trade_date >= ? AND trade_date <= ? ORDER BY ts_code, trade_date",
        conn,
        params=[min_date, date],
    )
    if df.empty:
        return None

    for col in ("close", "open", "high", "low", "vol"):
        df[col] = df[col].astype(np.float64)

    g = df.groupby("ts_code", sort=False)

    # Returns
    df["returns_1d"] = g["close"].transform(lambda s: s.pct_change(1))
    df["returns_5d"] = g["close"].transform(lambda s: s.pct_change(5))
    df["returns_20d"] = g["close"].transform(lambda s: s.pct_change(20))

    # Volatility
    df["volatility_5d"] = g["returns_1d"].transform(
        lambda s: s.rolling(5, min_periods=1).std()
    )
    df["volatility_20d"] = g["returns_1d"].transform(
        lambda s: s.rolling(20, min_periods=1).std()
    )

    # Moving averages
    df["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["ma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["ma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=1).mean())

    # Previous MAs (for trend_ma first-alignment detection)
    df["prev_ma5"] = g["ma5"].transform(lambda s: s.shift(1))
    df["prev_ma10"] = g["ma10"].transform(lambda s: s.shift(1))
    df["prev_ma20"] = g["ma20"].transform(lambda s: s.shift(1))
    df["prev_ma60"] = g["ma60"].transform(lambda s: s.shift(1))

    # Volume features
    df["vol_ma20"] = g["vol"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["volume_ratio"] = np.where(df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"], 1.0)

    # High / price ratios
    df["high20"] = g["high"].transform(
        lambda s: s.rolling(20, min_periods=1).max().shift(1)
    )
    df["price_to_high20"] = np.where(
        df["high20"] > 0, df["close"] / df["high20"] - 1.0, 0.0
    )
    df["price_to_ma20"] = np.where(df["ma20"] > 0, df["close"] / df["ma20"] - 1.0, 0.0)
    df["ma5_to_ma20"] = np.where(df["ma20"] > 0, df["ma5"] / df["ma20"] - 1.0, 0.0)
    df["vol_to_vol_ma"] = np.where(
        df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"] - 1.0, 0.0
    )

    # RSI 14
    delta = g["close"].transform(lambda s: s.diff())
    gain = delta.groupby(df["ts_code"]).transform(
        lambda s: s.clip(lower=0).rolling(14, min_periods=1).mean()
    )
    loss = delta.groupby(df["ts_code"]).transform(
        lambda s: (-s.clip(upper=0)).rolling(14, min_periods=1).mean()
    )
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))

    df["cumcount"] = g.cumcount()

    _compute_zlcmq_vectorized(df)

    # MACD features (params: 15/26/13 matching backtest engine)
    ema_fast = g["close"].transform(lambda s: s.ewm(span=15, adjust=False).mean())
    ema_slow = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["dif"] = ema_fast - ema_slow
    df["dea"] = g["dif"].transform(lambda s: s.ewm(span=13, adjust=False).mean())
    df["macd_hist"] = 2.0 * (df["dif"] - df["dea"])
    df["prev_dif"] = g["dif"].shift(1)
    df["prev_dea"] = g["dea"].shift(1)

    # Bollinger features (params: 25-period, 2.5x matching backtest engine)
    df["bb_mid"] = g["close"].transform(lambda s: s.rolling(25, min_periods=25).mean())
    df["bb_std"] = g["close"].transform(lambda s: s.rolling(25, min_periods=25).std())
    df["bb_upper"] = df["bb_mid"] + 2.5 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2.5 * df["bb_std"]
    df["prev_close"] = g["close"].shift(1)
    df["prev_lower"] = g["bb_lower"].shift(1)
    df["dist_to_lower"] = np.where(
        df["bb_lower"] != 0,
        (df["close"] - df["bb_lower"]) / df["bb_lower"],
        0.0,
    )

    # Top/Bottom features (通达信顶底图 indicator)
    df["ma13"] = g["close"].transform(lambda s: s.rolling(13, min_periods=13).mean())
    df["var4_tb"] = np.where(
        df["ma13"] > 0,
        100.0 - np.abs((df["close"] - df["ma13"]) / df["ma13"] * 100.0),
        50.0,
    )
    try:
        from scipy.stats import percentileofscore

        df["varb_tb"] = (
            g["close"]
            .transform(
                lambda s: s.rolling(250, min_periods=20).apply(
                    lambda x: percentileofscore(x, x[-1]), raw=True
                )
            )
            .fillna(50.0)
        )
    except Exception:
        df["varb_tb"] = 50.0
    df["score_tb"] = df["var4_tb"] - df["varb_tb"]
    df["score_max5_prev_tb"] = g["score_tb"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).max()
    )

    return df


@app.get("/api/strategies/chip-picks")
def get_chip_picks(date: Optional[str] = None, refresh: bool = False):
    """Chip strategy picks using relaxed backtest-engine conditions.

    Conditions (from backtest/dynamic_engine.py _chip_strategy):
    - ZLCMQ in [60, 92]
    - zlcmq_max5 >= 85 (was_high_zone)
    - zlcmq < zlcmq_prev1 (currently declining)
    - (zlcmq_max5 - zlcmq) <= 35 (not falling too fast)
    - Price stable: (close >= open) OR (close > prev_close)
    """
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None

        if not date:
            return {
                "enhanced_chip": {
                    "description": "筹码策略(回测引擎条件)",
                    "buy_signals": [],
                    "sell_signals": [],
                }
            }

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
                (f"chip_picks_{date}",),
            ).fetchone()
            if cached:
                return json.loads(cached[0])

        trading_dates_rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 100",
            (date,),
        ).fetchall()
        if not trading_dates_rows:
            return {
                "enhanced_chip": {
                    "description": "筹码策略(回测引擎条件)",
                    "buy_signals": [],
                    "sell_signals": [],
                }
            }

        min_date = trading_dates_rows[-1][0]

        df = pd.read_sql(
            "SELECT ts_code, trade_date, open, high, low, close, vol FROM daily_quotes WHERE trade_date >= ? AND trade_date <= ? ORDER BY ts_code, trade_date",
            conn,
            params=[min_date, date],
        )
        if df.empty:
            return {
                "enhanced_chip": {
                    "description": "筹码策略(回测引擎条件)",
                    "buy_signals": [],
                    "sell_signals": [],
                }
            }

        for col in ("close", "open", "high", "low", "vol"):
            df[col] = df[col].astype(np.float64)

        g = df.groupby("ts_code", sort=False)
        df["returns_1d"] = g["close"].transform(lambda s: s.pct_change(1))
        df["cumcount"] = g.cumcount()

        df = _compute_zlcmq_vectorized(df)

        day = df[df["trade_date"] == date].copy()

        day = day[day["cumcount"] >= 74]
        if day.empty:
            result = {
                "enhanced_chip": {
                    "description": "筹码策略(回测引擎条件)",
                    "buy_signals": [],
                    "sell_signals": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"chip_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        cur_z = day["zlcmq"].values
        z5_max = day["zlcmq_max5"].values
        z_prev = day["zlcmq_prev1"].values
        cur_close = day["close"].values
        cur_open = day["open"].values
        prev_close = np.where(
            day["returns_1d"].isna(),
            cur_close,
            cur_close / (1.0 + day["returns_1d"].fillna(0).values),
        )

        valid = (~np.isnan(cur_z)) & (cur_z >= 60) & (cur_z <= 92)
        day = day[valid]
        if day.empty:
            result = {
                "enhanced_chip": {
                    "description": "筹码策略(回测引擎条件)",
                    "buy_signals": [],
                    "sell_signals": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"chip_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        cur_z = day["zlcmq"].values
        z5_max = day["zlcmq_max5"].values
        z_prev = day["zlcmq_prev1"].values
        cur_close = day["close"].values
        cur_open = day["open"].values
        prev_close = np.where(
            day["returns_1d"].isna(),
            cur_close,
            cur_close / (1.0 + day["returns_1d"].fillna(0).values),
        )

        was_high_zone = z5_max >= 85
        is_declining = cur_z < z_prev
        not_too_fast = (z5_max - cur_z) <= 35
        is_stable = (cur_close >= cur_open) | (cur_close > prev_close)

        passed = day[was_high_zone & is_declining & not_too_fast & is_stable].copy()
        if not passed.empty:
            passed = passed.sort_values("zlcmq", ascending=False).head(5)

        chip_buys = []
        for _, row in passed.iterrows():
            chip_buys.append(
                {
                    "symbol": row["ts_code"],
                    "name": _get_stock_name_safe(row["ts_code"]),
                    "action": "buy",
                    "price": round(float(row["close"]), 2),
                    "date": date,
                    "zlcmq": round(float(row["zlcmq"]), 1),
                    "zlcmq_max5": round(float(row["zlcmq_max5"]), 1),
                    "reason": f"筹码高位回落企稳 ZLCMQ={round(float(row['zlcmq']), 1)} 峰值={round(float(row['zlcmq_max5']), 1)}",
                }
            )

        result = {
            "enhanced_chip": {
                "description": "筹码策略(回测引擎条件)",
                "buy_signals": chip_buys,
                "sell_signals": [],
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (f"chip_picks_{date}", json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return result
    finally:
        conn.close()


@app.get("/api/strategies/ml-picks")
def get_ml_picks(date: Optional[str] = None, refresh: bool = False):
    """ML picks using pooled HistGradientBoosting (same as backtest engine).

    Trains ONE model on ALL stocks pooled together instead of per-stock training.
    """
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None

        if not date:
            return {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }

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
                (f"ml_picks_{date}",),
            ).fetchone()
            if cached:
                return json.loads(cached[0])

        FEATURE_COLS = [
            "returns_20d",
            "volatility_20d",
            "returns_5d",
            "volatility_5d",
            "price_to_ma20",
            "ma5_to_ma20",
            "vol_to_vol_ma",
            "price_to_high20",
            "volume_ratio",
            "rsi_14",
        ]

        trading_dates_rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 176",
            (date,),
        ).fetchall()
        if len(trading_dates_rows) < 30:
            result = {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        min_date = trading_dates_rows[-1][0]

        df = pd.read_sql(
            "SELECT ts_code, trade_date, open, high, low, close, vol FROM daily_quotes WHERE trade_date >= ? AND trade_date <= ? ORDER BY ts_code, trade_date",
            conn,
            params=[min_date, date],
        )
        if df.empty:
            result = {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        for col in ("close", "open", "high", "low", "vol"):
            df[col] = df[col].astype(np.float64)

        g = df.groupby("ts_code", sort=False)
        df["returns_1d"] = g["close"].transform(lambda s: s.pct_change(1))
        df["returns_5d"] = g["close"].transform(lambda s: s.pct_change(5))
        df["returns_20d"] = g["close"].transform(lambda s: s.pct_change(20))
        df["volatility_5d"] = g["returns_1d"].transform(
            lambda s: s.rolling(5, min_periods=1).std()
        )
        df["volatility_20d"] = g["returns_1d"].transform(
            lambda s: s.rolling(20, min_periods=1).std()
        )
        df["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        df["vol_ma20"] = g["vol"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
        df["volume_ratio"] = np.where(
            df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"], 1.0
        )
        df["high20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=1).max())
        df["price_to_high20"] = np.where(
            df["high20"] > 0, df["close"] / df["high20"] - 1.0, 0.0
        )
        df["price_to_ma20"] = np.where(
            df["ma20"] > 0, df["close"] / df["ma20"] - 1.0, 0.0
        )
        df["ma5_to_ma20"] = np.where(df["ma20"] > 0, df["ma5"] / df["ma20"] - 1.0, 0.0)
        df["vol_to_vol_ma"] = np.where(
            df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"] - 1.0, 0.0
        )

        delta = g["close"].transform(lambda s: s.diff())
        gain = delta.groupby(df["ts_code"]).transform(
            lambda s: s.clip(lower=0).rolling(14, min_periods=1).mean()
        )
        loss = delta.groupby(df["ts_code"]).transform(
            lambda s: (-s.clip(upper=0)).rolling(14, min_periods=1).mean()
        )
        df["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))

        df["cumcount"] = g.cumcount()
        df["future_close_5d"] = g["close"].transform(lambda s: s.shift(-5))
        df["label_5d_up"] = (df["future_close_5d"] > df["close"]).astype(np.float64)

        pool = df[df["cumcount"] >= 49].dropna(subset=FEATURE_COLS)
        if pool.shape[0] < 100:
            result = {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        labels = pool["label_5d_up"].values
        has_label = ~np.isnan(labels)
        X_all = pool[FEATURE_COLS].values
        y_all = labels
        X_train = X_all[has_label]
        y_train = np.nan_to_num(y_all[has_label], nan=0.0)

        if len(X_train) < 50 or len(np.unique(y_train)) < 2:
            result = {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)

        day = df[df["trade_date"] == date].dropna(subset=FEATURE_COLS)
        if day.empty:
            result = {
                "ml_stock": {
                    "description": "ML选股 (HistGradientBoosting 池化)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        X_score = day[FEATURE_COLS].values
        proba = model.predict_proba(X_score)
        prob_up = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

        passed = day[prob_up > 0.45].copy()
        if not passed.empty:
            passed["prob_up"] = prob_up[prob_up > 0.45]
            passed = passed.sort_values("prob_up", ascending=False)

        ml_picks = []
        for rank, (_, row) in enumerate(passed.head(5).iterrows(), 1):
            p = round(float(row["prob_up"]) * 100)
            momentum_str = (
                f"{row['returns_20d'] * 100:.1f}"
                if pd.notna(row.get("returns_20d"))
                else "N/A"
            )
            ml_picks.append(
                {
                    "symbol": row["ts_code"],
                    "name": _get_stock_name_safe(row["ts_code"]),
                    "prediction": 1,
                    "prob_up": round(float(row["prob_up"]), 4),
                    "date": date,
                    "close": round(float(row["close"]), 2),
                    "reason": f"ML预测上涨概率{p}%，动量{momentum_str}%",
                }
            )

        result = {
            "ml_stock": {
                "description": "ML选股 (HistGradientBoosting 池化)",
                "picks": ml_picks,
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (f"ml_picks_{date}", json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return result
    finally:
        conn.close()


@app.get("/api/strategies/multifactor-picks")
def get_multifactor_picks(date: Optional[str] = None, refresh: bool = False):
    """Three-factor scoring: score = 0.4*(-close) + 0.3*returns_20d + 0.3*(1/volatility_20d).

    Stocks with cumcount >= 29, sorted by score descending, top 10.
    """
    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None

        if not date:
            return {
                "multifactor": {
                    "description": "三因子选股 (市值+动量+质量)",
                    "picks": [],
                }
            }

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
                (f"multifactor_picks_{date}",),
            ).fetchone()
            if cached:
                return json.loads(cached[0])

        trading_dates_rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 31",
            (date,),
        ).fetchall()
        if len(trading_dates_rows) < 5:
            result = {
                "multifactor": {
                    "description": "三因子选股 (市值+动量+质量)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"multifactor_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        min_date = trading_dates_rows[-1][0]

        df = pd.read_sql(
            "SELECT ts_code, trade_date, open, high, low, close, vol FROM daily_quotes WHERE trade_date >= ? AND trade_date <= ? ORDER BY ts_code, trade_date",
            conn,
            params=[min_date, date],
        )
        if df.empty:
            result = {
                "multifactor": {
                    "description": "三因子选股 (市值+动量+质量)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"multifactor_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        for col in ("close", "open", "high", "low", "vol"):
            df[col] = df[col].astype(np.float64)

        g = df.groupby("ts_code", sort=False)
        df["returns_1d"] = g["close"].transform(lambda s: s.pct_change(1))
        df["returns_20d"] = g["close"].transform(lambda s: s.pct_change(20))
        df["volatility_20d"] = g["returns_1d"].transform(
            lambda s: s.rolling(20, min_periods=1).std()
        )
        df["cumcount"] = g.cumcount()

        day = df[df["trade_date"] == date].copy()
        day = day[day["cumcount"] >= 29]
        if day.empty:
            result = {
                "multifactor": {
                    "description": "三因子选股 (市值+动量+质量)",
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (f"multifactor_picks_{date}", json.dumps(result, ensure_ascii=False)),
            )
            conn.commit()
            return result

        momentum = day["returns_20d"].values
        volatility = day["volatility_20d"].values
        quality = np.where(
            (volatility == 0) | np.isnan(volatility),
            0.0,
            1.0 / (volatility + 1e-6),
        )
        market_cap_proxy = day["close"].values
        scores = (
            0.4 * (-market_cap_proxy)
            + 0.3 * np.nan_to_num(momentum, nan=0.0)
            + 0.3 * quality
        )
        day["score"] = scores
        day = day.sort_values("score", ascending=False)

        picks = []
        for rank, (_, row) in enumerate(day.head(5).iterrows(), 1):
            momentum_str = (
                f"{row['returns_20d'] * 100:.1f}"
                if pd.notna(row.get("returns_20d"))
                else "N/A"
            )
            picks.append(
                {
                    "symbol": row["ts_code"],
                    "name": _get_stock_name_safe(row["ts_code"]),
                    "close": round(float(row["close"]), 2),
                    "score": round(float(row["score"]), 2),
                    "momentum_20d": round(float(row["returns_20d"]), 4)
                    if pd.notna(row["returns_20d"])
                    else 0.0,
                    "volatility_20d": round(float(row["volatility_20d"]), 4)
                    if pd.notna(row["volatility_20d"])
                    else 0.0,
                    "rank": rank,
                    "reason": f"三因子评分#{rank} 动量{momentum_str}% 低波动",
                }
            )

        result = {
            "multifactor": {
                "description": "三因子选股 (市值+动量+质量)",
                "picks": picks,
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (f"multifactor_picks_{date}", json.dumps(result, ensure_ascii=False)),
        )
        conn.commit()
        return result
    finally:
        conn.close()


_STRATEGY_DESCRIPTIONS = {
    "vol_breakout": "爆量突破策略 — 低位横盘后爆量突破",
    "dragon_first_yin": "龙头首阴策略 — 连续涨停后首阴低吸",
    "trend_ma": "均线趋势策略 — 多头排列首次形成",
    "top_bottom": "顶底图策略 — 通达信顶底图底部信号",
    "bollinger_break": "布林带突破策略 — 触及下轨超卖反弹",
    "rsi_momentum": "RSI动量策略 — RSI超卖反弹",
    "macd_cross": "MACD金叉策略 — MACD金叉确认",
}


def _apply_strategy_filter(strategy_name: str, feat_df, date, conn):
    """Apply strategy filter logic on the feature DataFrame for a given date.

    Returns a list of dicts with keys: ts_code, reason, sort_key.
    """
    day = feat_df[feat_df["trade_date"] == date].copy()
    if day.empty:
        return []

    if strategy_name == "vol_breakout":
        day = day[day["cumcount"] >= 25]
        if day.empty:
            return []
        signals = day[
            (day["volatility_5d"] < 0.15)
            & (day["vol"] > day["vol_ma20"] * 2.0)
            & (day["close"] > day["high20"])
            & (day["close"] > day["ma20"])
        ].copy()
        if signals.empty:
            return []
        signals = signals.sort_values("volume_ratio", ascending=False)
        results = []
        for _, row in signals.head(5).iterrows():
            vr = float(row["volume_ratio"]) if pd.notna(row["volume_ratio"]) else 0
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": f"低位横盘后爆量突破20日高点，量比{vr:.1f}倍",
                }
            )
        return results

    elif strategy_name == "dragon_first_yin":
        day = day[day["cumcount"] >= 40]
        if day.empty:
            return []
        trading_dates_sorted = sorted(feat_df["trade_date"].unique())
        date_idx = (
            trading_dates_sorted.index(date) if date in trading_dates_sorted else -1
        )
        if date_idx < 1:
            return []
        prev_date = trading_dates_sorted[date_idx - 1]
        yesterday = feat_df[feat_df["trade_date"] == prev_date][
            ["ts_code", "pct_chg", "open"]
        ].copy()
        yesterday = yesterday.rename(
            columns={"pct_chg": "prev_pct_chg", "open": "prev_open"}
        )
        today_df = feat_df[feat_df["trade_date"] == date][
            ["ts_code", "open", "close"]
        ].copy()
        merged = today_df.merge(yesterday, on="ts_code", how="inner")
        if merged.empty:
            return []
        signals = merged[
            (merged["prev_pct_chg"] > 9.5)
            & (merged["close"] < merged["open"])
            & (merged["close"] > merged["prev_open"])
        ].copy()
        if signals.empty:
            return []
        valid_codes = set(signals["ts_code"].values)
        day_filtered = day[day["ts_code"].isin(valid_codes)]
        signals = signals[signals["ts_code"].isin(day_filtered["ts_code"].values)]
        signals = signals.sort_values("prev_pct_chg", ascending=False)
        results = []
        for _, row in signals.head(5).iterrows():
            pct = float(row["prev_pct_chg"]) if pd.notna(row["prev_pct_chg"]) else 0
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": f"昨日涨停{pct:.1f}%后首阴回踩，支撑位守住",
                }
            )
        return results

    elif strategy_name == "trend_ma":
        day = day[day["cumcount"] >= 35].dropna(
            subset=["ma10", "prev_ma5", "prev_ma10", "prev_ma20", "prev_ma60"]
        )
        if day.empty:
            return []
        bull_today = (
            (day["ma5"] > day["ma10"])
            & (day["ma10"] > day["ma20"])
            & (day["ma20"] > day["ma60"])
        )
        bull_yesterday = (
            (day["prev_ma5"] > day["prev_ma10"])
            & (day["prev_ma10"] > day["prev_ma20"])
            & (day["prev_ma20"] > day["prev_ma60"])
        )
        signals = day[bull_today & ~bull_yesterday].copy()
        if signals.empty:
            return []
        signals = signals.sort_values("ma5_to_ma20", ascending=False)
        results = []
        for _, row in signals.head(5).iterrows():
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": "均线系统首次多头排列，趋势启动信号",
                }
            )
        return results

    elif strategy_name == "top_bottom":
        day = day[day["cumcount"] >= 250].dropna(subset=["score_tb"])
        if day.empty:
            return []
        signals = day[
            (day["score_tb"] > 10) & (day["score_tb"] > day["score_max5_prev_tb"])
        ].copy()
        if signals.empty:
            return []
        signals = signals.sort_values("score_tb", ascending=False)
        results = []
        for _, row in signals.head(5).iterrows():
            sc = float(row["score_tb"]) if pd.notna(row["score_tb"]) else 0
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": f"顶底图指标得分{sc:.1f}，底部反弹信号",
                }
            )
        return results

    elif strategy_name == "bollinger_break":
        day = day.dropna(subset=["bb_lower", "prev_close", "prev_lower"])
        if day.empty:
            return []
        day = day[day["cumcount"] >= 30]
        if day.empty:
            return []
        signals = day[
            (day["prev_close"] > day["prev_lower"]) & (day["close"] <= day["bb_lower"])
        ].copy()
        if signals.empty:
            return []
        signals = signals.sort_values("dist_to_lower", ascending=True)
        results = []
        for _, row in signals.head(5).iterrows():
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": "触及布林带下轨，超卖反弹机会",
                }
            )
        return results

    elif strategy_name == "rsi_momentum":
        day = day[day["cumcount"] >= 20]
        if day.empty:
            return []
        signals = day[day["rsi_14"] < 25].copy()
        if signals.empty:
            return []
        signals = signals.sort_values("rsi_14", ascending=True)
        results = []
        for _, row in signals.head(5).iterrows():
            rsi = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else 0
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": f"RSI={rsi:.0f}严重超卖，技术反弹概率大",
                }
            )
        return results

    elif strategy_name == "macd_cross":
        day = day.dropna(subset=["prev_dif", "prev_dea", "dif", "dea"])
        if day.empty:
            return []
        day = day[day["cumcount"] >= 40]
        if day.empty:
            return []
        signals = day[
            (day["prev_dif"] <= day["prev_dea"]) & (day["dif"] > day["dea"])
        ].copy()
        if signals.empty:
            return []
        signals["macd_abs"] = np.abs(signals["macd_hist"].values)
        signals = signals.sort_values("macd_abs", ascending=False)
        results = []
        for _, row in signals.head(5).iterrows():
            results.append(
                {
                    "ts_code": row["ts_code"],
                    "reason": "MACD金叉确认，多头力量增强",
                }
            )
        return results

    return []


@app.get("/api/strategies/{strategy_name}/picks")
def get_strategy_picks_by_name(
    strategy_name: str,
    date: Optional[str] = None,
    refresh: bool = False,
):
    if strategy_name not in _STRATEGY_DESCRIPTIONS:
        raise HTTPException(404, f"策略 {strategy_name} 不支持选股API")

    conn = _get_db_conn()
    try:
        if not date:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM etf_daily_quotes"
            ).fetchone()
            date = row[0] if row and row[0] else None

        if not date:
            return {
                strategy_name: {
                    "description": _STRATEGY_DESCRIPTIONS[strategy_name],
                    "picks": [],
                }
            }

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
                (f"strategy_picks_{strategy_name}_{date}",),
            ).fetchone()
            if cached:
                return json.loads(cached[0])

        feat_df = _compute_stock_features(conn, date)
        if feat_df is None:
            result = {
                strategy_name: {
                    "description": _STRATEGY_DESCRIPTIONS[strategy_name],
                    "picks": [],
                }
            }
            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
                (
                    f"strategy_picks_{strategy_name}_{date}",
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            conn.commit()
            return result

        filtered = _apply_strategy_filter(strategy_name, feat_df, date, conn)

        picks = []
        for rank, item in enumerate(filtered, 1):
            ts_code = item["ts_code"]
            price_row = feat_df[
                (feat_df["ts_code"] == ts_code) & (feat_df["trade_date"] == date)
            ]
            price = float(price_row["close"].iloc[0]) if not price_row.empty else 0.0
            picks.append(
                {
                    "symbol": ts_code,
                    "name": _get_stock_name_safe(ts_code),
                    "action": "buy",
                    "price": round(price, 2),
                    "date": date,
                    "reason": item["reason"],
                    "strategy": strategy_name,
                    "rank": rank,
                }
            )

        result = {
            strategy_name: {
                "description": _STRATEGY_DESCRIPTIONS[strategy_name],
                "picks": picks,
            }
        }
        conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, ?)",
            (
                f"strategy_picks_{strategy_name}_{date}",
                json.dumps(result, ensure_ascii=False),
            ),
        )
        conn.commit()
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"策略选股失败 {strategy_name}: {e}")
        raise HTTPException(500, f"策略选股失败: {str(e)}")
    finally:
        conn.close()


@app.get("/api/stocks/list")
def list_available_stocks():
    conn = _get_db_conn()
    stocks = conn.execute(
        "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
    ).fetchall()
    conn.close()
    return {
        "stocks": [{"code": s[0], "name": _get_stock_name_safe(s[0])} for s in stocks]
    }


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
    from web.notification import (
        send_email_notification,
        format_signal_email,
        get_notification_config,
    )

    config = get_notification_config()
    if not config.get("email_enabled"):
        raise HTTPException(400, "Email not enabled")

    html = format_signal_email(
        signals=[
            {
                "action": "buy",
                "name": "测试ETF",
                "symbol": "test",
                "reference_price": 1.0,
                "reason": "测试通知",
            }
        ],
        portfolio={"total_assets": 1000000, "total_pnl": 0, "total_pnl_pct": 0},
        regime={"regime": "neutral", "description": "测试"},
    )
    ok = send_email_notification("【测试】量化交易系统通知", html, config)
    return {"sent": ok}


@app.post("/api/notifications/send")
def send_daily_notification():
    from web.notification import (
        send_email_notification,
        format_signal_email,
        get_notification_config,
    )

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
def expand_stock_pool(
    market: str = Query("csi300", description="csi300, csi500, csi800"),
):
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
        existing = set(
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT ts_code FROM daily_quotes"
            ).fetchall()
        )

        new_codes = []
        for code in codes:
            if len(str(code)) == 6:
                if str(code).startswith("6"):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
                if ts_code not in existing:
                    new_codes.append(ts_code)

        new_codes = new_codes[:50]

        for ts_code in new_codes[:20]:
            symbol = ts_code.split(".")[0]
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily", start_date="20200101", adjust="qfq"
                )
                if df is None or len(df) < 50:
                    continue

                col_map = {
                    "日期": "trade_date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "vol",
                    "成交额": "amount",
                    "涨跌幅": "pct_chg",
                }
                df = df.rename(
                    columns={k: v for k, v in col_map.items() if k in df.columns}
                )
                df["ts_code"] = ts_code

                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR REPLACE INTO daily_quotes (ts_code, trade_date, open, high, low, close, vol, amount, pct_chg) VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            ts_code,
                            str(row.get("trade_date", "")),
                            float(row.get("open", 0)),
                            float(row.get("high", 0)),
                            float(row.get("low", 0)),
                            float(row.get("close", 0)),
                            float(row.get("vol", 0)),
                            float(row.get("amount", 0)),
                            float(row.get("pct_chg", 0)),
                        ),
                    )
                added.append(ts_code)
            except Exception as e:
                errors.append(f"{ts_code}: {str(e)[:40]}")

        conn.commit()
        total = conn.execute(
            "SELECT COUNT(DISTINCT ts_code) FROM daily_quotes"
        ).fetchone()[0]
        conn.close()
        return {"added": added, "errors": errors, "total_stocks": total}
    except Exception as e:
        conn.close()
        raise HTTPException(500, f"Expansion failed: {str(e)}")


@app.get("/api/data/stats")
def get_data_stats():
    conn = _get_db_conn()
    stock_count = conn.execute(
        "SELECT COUNT(DISTINCT ts_code) FROM daily_quotes"
    ).fetchone()[0]
    stock_rows = conn.execute("SELECT COUNT(*) FROM daily_quotes").fetchone()[0]
    stock_range = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes"
    ).fetchone()
    etf_count = conn.execute(
        "SELECT COUNT(DISTINCT etf_code) FROM etf_daily_quotes"
    ).fetchone()[0]
    etf_rows = conn.execute("SELECT COUNT(*) FROM etf_daily_quotes").fetchone()[0]
    etf_range = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM etf_daily_quotes"
    ).fetchone()
    conn.close()
    return {
        "stocks": {
            "count": stock_count,
            "rows": stock_rows,
            "date_range": list(stock_range),
        },
        "etfs": {"count": etf_count, "rows": etf_rows, "date_range": list(etf_range)},
    }


@app.get("/api/strategies")
def get_strategies():
    from strategies import list_strategies

    return {"strategies": list_strategies()}


@app.get("/api/strategies/{name}/config")
def get_strategy_config(name: str):
    from strategies import get_strategy

    info = get_strategy(name)
    if not info:
        raise HTTPException(404, f"策略 {name} 不存在")
    return {
        "name": info.name,
        "description": info.description,
        "supported_features": info.supported_features,
        "config_schema": info.config_schema.schema()
        if hasattr(info.config_schema, "schema")
        else str(info.config_schema),
    }


@app.post("/api/strategies/{name}/validate")
def validate_strategy(name: str, config: dict):
    from strategies import validate_strategy_config

    try:
        validated = validate_strategy_config(name, config)
        return {"valid": True, "config": validated.dict()}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.get("/api/scheduler/status")
def get_scheduler_status():
    if _scheduler is None:
        return {"running": False, "jobs": []}
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
            }
        )
    return {"running": _scheduler.running, "jobs": jobs}


@app.post("/api/scheduler/trigger/{job_name}")
def trigger_job(job_name: str):
    if _scheduler is None:
        raise HTTPException(500, "调度器未运行")
    try:
        job = _scheduler.get_job(job_name)
        if job:
            job.modify(next_run_time=datetime.now())
            return {"status": "ok", "message": f"任务 {job_name} 已触发"}
        raise HTTPException(404, f"任务 {job_name} 不存在")
    except Exception as e:
        raise HTTPException(500, f"触发失败: {str(e)}")


# ── Static Files ─────────────────────────────────────────────────


@app.get("/")
def serve_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/stocks/{code}/detail")
def get_stock_detail(code: str):
    conn = _get_db_conn()
    try:
        info = conn.execute(
            "SELECT * FROM stock_list WHERE ts_code=?", (code,)
        ).fetchone()
        if not info:
            raise HTTPException(404, f"股票 {code} 不存在")

        latest = conn.execute(
            "SELECT * FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1",
            (code,),
        ).fetchone()

        columns = [desc[0] for desc in conn.description]
        info_dict = dict(zip(columns, info)) if info else {}

        return {
            "basic": {
                "ts_code": info_dict.get("ts_code"),
                "symbol": info_dict.get("symbol"),
                "name": info_dict.get("name"),
                "area": info_dict.get("area"),
                "industry": info_dict.get("industry"),
                "market": info_dict.get("market"),
                "list_date": info_dict.get("list_date"),
            },
            "latest_quote": dict(zip(columns, latest)) if latest else None,
        }
    finally:
        conn.close()


@app.get("/api/stocks/{code}/indicators")
def get_stock_indicators(
    code: str,
    period: str = Query("daily", enum=["daily", "weekly", "monthly"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    conn = _get_db_conn()
    try:
        df = pd.read_sql(
            "SELECT * FROM daily_quotes WHERE ts_code=? ORDER BY trade_date",
            conn,
            params=[code],
        )
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]

        if len(df) < 20:
            raise HTTPException(400, "数据不足，无法计算指标")

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["vol"].astype(float)

        df["ma5"] = close.rolling(5, min_periods=1).mean()
        df["ma10"] = close.rolling(10, min_periods=1).mean()
        df["ma20"] = close.rolling(20, min_periods=1).mean()
        df["ma60"] = close.rolling(60, min_periods=1).mean()

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["dif"] = exp1 - exp2
        df["dea"] = (2 * exp1 - 2 * exp2).ewm(span=9, adjust=False).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 0.0001)
        df["rsi"] = 100 - (100 / (1 + rs))

        sma = (high + low + close) / 3
        mad = (
            (sma - sma.rolling(20, min_periods=1).mean())
            .abs()
            .rolling(20, min_periods=1)
            .mean()
        )
        df["cci"] = (sma - sma.rolling(20, min_periods=1).mean()) / (
            0.015 * mad + 0.001
        )

        df["boll_mid"] = close.rolling(20, min_periods=1).mean()
        df["boll_std"] = close.rolling(20, min_periods=1).std()
        df["boll_upper"] = df["boll_mid"] + 2 * df["boll_std"]
        df["boll_lower"] = df["boll_mid"] - 2 * df["boll_std"]

        lowest_low = low.rolling(9, min_periods=1).min()
        highest_high = high.rolling(9, min_periods=1).max()
        df["kdj_k"] = 100 * (close - lowest_low) / (highest_high - lowest_low + 0.0001)
        df["kdj_d"] = df["kdj_k"].rolling(3, min_periods=1).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        result_cols = [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "ma5",
            "ma10",
            "ma20",
            "ma60",
            "macd",
            "dif",
            "dea",
            "rsi",
            "cci",
            "boll_upper",
            "boll_mid",
            "boll_lower",
            "kdj_k",
            "kdj_d",
            "kdj_j",
        ]

        result = df[[c for c in result_cols if c in df.columns]].tail(100)
        return {"code": code, "indicators": result.to_dict(orient="records")}
    finally:
        conn.close()


@app.get("/api/stocks/{code}/charts")
def get_stock_charts(
    code: str,
    period: str = Query("daily", enum=["daily", "weekly", "monthly"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    adjust: str = Query("qfq", enum=["qfq", "hfq", ""]),
):
    conn = _get_db_conn()
    try:
        df = pd.read_sql(
            "SELECT trade_date, open, high, low, close, vol, amount, pct_chg FROM daily_quotes WHERE ts_code=? ORDER BY trade_date",
            conn,
            params=[code],
        )
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]

        if df.empty:
            raise HTTPException(404, f"股票 {code} 无数据")

        return {
            "code": code,
            "period": period,
            "data": df.tail(500).to_dict(orient="records"),
            "date_range": {
                "start": str(df["trade_date"].iloc[0]) if len(df) > 0 else None,
                "end": str(df["trade_date"].iloc[-1]) if len(df) > 0 else None,
            },
        }
    finally:
        conn.close()


@app.get("/api/stocks/list")
def get_stocks_list(
    market: Optional[str] = None,
    industry: Optional[str] = None,
    limit: int = Query(100, le=500),
):
    conn = _get_db_conn()
    try:
        query = "SELECT * FROM stock_list WHERE 1=1"
        params = []
        if market:
            query += " AND market=?"
            params.append(market)
        if industry:
            query += " AND industry=?"
            params.append(industry)
        query += f" LIMIT {limit}"

        rows = conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in conn.description]
        return {"stocks": [dict(zip(columns, row)) for row in rows]}
    finally:
        conn.close()


@app.get("/api/backtest/dynamic")
def run_dynamic_backtest(
    strategies: str = Query(..., description="策略列表，逗号分隔，如 'ma,chip,ml'"),
    stock_pool: str = Query("hs300", enum=["hs300", "zz500", "zz800", "all"]),
    start_date: str = Query(..., description="开始日期，如 20250101"),
    end_date: str = Query(..., description="结束日期，如 20260413"),
    initial_capital: float = Query(1_000_000, description="初始资金"),
    rebalance_days: int = Query(20, ge=5, le=60, description="调仓周期"),
    max_positions: int = Query(10, ge=1, le=50, description="最大持仓数"),
    max_position_pct: float = Query(
        0.2, ge=0.05, le=0.5, description="单只最大仓位占比"
    ),
):
    try:
        from backtest.dynamic_engine import (
            DynamicBacktestEngine,
            BacktestConfig,
            init_backtest_db,
        )

        strategies_list = [s.strip() for s in strategies.split(",") if s.strip()]
        init_backtest_db()

        config = BacktestConfig(
            strategies=strategies_list,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            rebalance_days=rebalance_days,
            max_positions=max_positions,
            max_position_pct=max_position_pct,
        )

        engine = DynamicBacktestEngine(config)
        run_id = engine.run()
        engine.close()

        return {
            "run_id": run_id,
            "status": "completed",
            "message": f"回测完成，策略: {', '.join(strategies_list)}",
        }
    except Exception as e:
        logger.exception(f"动态回测失败: {e}")
        raise HTTPException(500, f"回测失败: {str(e)}")


@app.get("/api/backtest/{run_id}/result")
def get_backtest_result(run_id: str):
    from backtest.dynamic_engine import get_backtest_result

    result = get_backtest_result(run_id)
    if not result:
        raise HTTPException(404, f"回测结果不存在: {run_id}")

    return result


@app.get("/api/backtest/{run_id}/delivery")
def get_backtest_delivery(
    run_id: str,
    strategy: Optional[str] = Query(None, description="策略名称，不填则返回所有"),
    fmt: Optional[str] = Query("json", enum=["json", "csv"], description="返回格式"),
):
    from backtest.dynamic_engine import get_delivery

    deliveries = get_delivery(run_id, strategy or None)

    if strategy:
        strategy_names = [strategy]
    else:
        conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "backtest.db"))
        try:
            rows = conn.execute(
                "SELECT DISTINCT strategy_name FROM backtest_trades WHERE run_id=?",
                (run_id,),
            ).fetchall()
            strategy_names = [r[0] for r in rows]
        finally:
            conn.close()

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "trade_date",
                "symbol",
                "action",
                "price",
                "shares",
                "amount",
                "reason",
                "strategy",
            ]
        )
        for d in deliveries:
            writer.writerow(
                [
                    d.get("trade_date", ""),
                    d.get("symbol", ""),
                    d.get("action", ""),
                    d.get("price", ""),
                    d.get("shares", ""),
                    d.get("amount", ""),
                    d.get("reason", ""),
                    d.get("strategy_name", ""),
                ]
            )
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=delivery_{run_id}.csv"
            },
        )

    return {
        "run_id": run_id,
        "strategies": strategy_names,
        "deliveries": deliveries,
    }


@app.get("/api/backtest/history")
def get_backtest_history(limit: int = Query(20, le=100)):
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "sqlite" / "backtest.db"))
    try:
        rows = conn.execute(
            "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = [desc[0] for desc in conn.description]
        runs = []
        for row in rows:
            r = dict(zip(cols, row))
            results = conn.execute(
                "SELECT strategy_name, total_return, sharpe, max_drawdown FROM backtest_results WHERE run_id=?",
                (r["id"],),
            ).fetchall()
            r["results"] = [
                dict(
                    zip(
                        ["strategy_name", "total_return", "sharpe", "max_drawdown"], res
                    )
                )
                for res in results
            ]
            runs.append(r)
        return {"runs": runs}
    finally:
        conn.close()


@app.get("/api/backtest/strategies")
def get_backtest_available_strategies():
    return {
        "strategies": [
            {
                "id": "ma",
                "name": "均线策略",
                "description": "MA5上穿MA20买入，下穿卖出",
            },
            {"id": "chip", "name": "筹码策略", "description": "基于ZLCMQ指标选股"},
            {"id": "ml", "name": "ML策略", "description": "机器学习预测涨跌"},
            {
                "id": "multifactor",
                "name": "多因子策略",
                "description": "市值+质量+动量三因子",
            },
            {"id": "etf_rotation", "name": "ETF轮动", "description": "动量排名轮动"},
            {
                "id": "vol_breakout",
                "name": "爆量突破",
                "description": "低位横盘后爆量突破",
            },
            {
                "id": "dragon_first_yin",
                "name": "龙头首阴",
                "description": "连续涨停后首阴低吸",
            },
            {"id": "trend_ma", "name": "均线趋势", "description": "多头排列首次形成"},
            {"id": "top_bottom", "name": "顶底图", "description": "顶底图指标底部信号"},
            {
                "id": "bollinger_break",
                "name": "布林带突破",
                "description": "触及下轨超卖反弹",
            },
            {"id": "rsi_momentum", "name": "RSI动量", "description": "RSI超卖反弹"},
            {"id": "macd_cross", "name": "MACD金叉", "description": "MACD金叉确认"},
        ],
        "stock_pools": [
            {"id": "hs300", "name": "沪深300"},
            {"id": "zz500", "name": "中证500"},
            {"id": "zz800", "name": "中证800"},
            {"id": "all", "name": "全市场"},
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8787)
