"""数据库初始化脚本"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from config.settings import config

def init_database():
    """初始化数据库表结构"""
    engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=config.DB_ECHO)

    sql_statements = [
        """
        CREATE TABLE IF NOT EXISTS stock_list (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            area TEXT,
            industry TEXT,
            market TEXT,
            list_date TEXT,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS daily_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            change REAL,
            pct_chg REAL,
            vol REAL,
            amount REAL,
            UNIQUE(ts_code, trade_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS minute_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            UNIQUE(ts_code, datetime)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            ma5 REAL,
            ma10 REAL,
            ma20 REAL,
            ma60 REAL,
            macd REAL,
            dif REAL,
            dea REAL,
            rsi REAL,
            kdj_k REAL,
            kdj_d REAL,
            kdj_j REAL,
            boll_upper REAL,
            boll_mid REAL,
            boll_lower REAL,
            UNIQUE(ts_code, trade_date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS financial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            ann_date TEXT NOT NULL,
            end_date TEXT,
            report_type TEXT,
            basic_eps REAL,
            diluted_eps REAL,
            total_revenue REAL,
            revenue REAL,
            total_cogs REAL,
            oper_exp REAL,
            total_profit REAL,
            n_income REAL,
            n_income_attr_p REAL,
            UNIQUE(ts_code, ann_date)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_daily_quotes_ts_code ON daily_quotes(ts_code)",
        "CREATE INDEX IF NOT EXISTS idx_daily_quotes_trade_date ON daily_quotes(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_minute_quotes_ts_code ON minute_quotes(ts_code)",
        "CREATE INDEX IF NOT EXISTS idx_minute_quotes_datetime ON minute_quotes(datetime)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_ts_code ON indicators(ts_code)",
        "CREATE INDEX IF NOT EXISTS idx_indicators_trade_date ON indicators(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_financial_ts_code ON financial(ts_code)",
        "CREATE INDEX IF NOT EXISTS idx_financial_ann_date ON financial(ann_date)",
    ]

    with engine.connect() as conn:
        for sql in sql_statements:
            conn.execute(text(sql))
        conn.commit()

    print(f"数据库初始化完成: {config.DB_PATH}")
    print(f"已创建表: stock_list, daily_quotes, minute_quotes, indicators, financial")


if __name__ == "__main__":
    init_database()
