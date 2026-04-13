"""
Download extended data: ETF daily quotes + stock fundamental/valuation data.
Uses AKShare APIs and stores into the existing SQLite database.
"""

import sqlite3
import time
import logging

import akshare as ak
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/sqlite/stock_data.db"

INDUSTRY_ETFS = {
    "sz159919": "300ETF",
    "sz159915": "创业板ETF",
    "sz159949": "创业板50",
    "sh510500": "500ETF",
    "sz159966": "芯片ETF",
    "sz159996": "家电ETF",
    "sz159998": "计算机ETF",
    "sz159997": "电子ETF",
    "sz512660": "军工ETF",
    "sz512170": "医疗ETF",
    "sz512010": "医药ETF",
    "sz159925": "沪深300ETF",
    "sz510880": "红利ETF",
}


def create_tables(conn: sqlite3.Connection):
    """Create extended data tables if they don't exist."""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS etf_daily_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            etf_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            UNIQUE(etf_code, trade_date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_valuation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            total_market_cap REAL,
            circulating_market_cap REAL,
            UNIQUE(ts_code, trade_date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_fundamental (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            report_date TEXT NOT NULL,
            net_profit REAL,
            net_profit_yoy REAL,
            revenue REAL,
            revenue_yoy REAL,
            eps REAL,
            bvps REAL,
            roe REAL,
            gross_margin REAL,
            net_margin REAL,
            debt_ratio REAL,
            UNIQUE(ts_code, report_date)
        )
    """)
    conn.commit()
    logger.info("Tables created/verified.")


def download_etf_data(conn: sqlite3.Connection, etfs: dict | None = None):
    """Download ETF daily quotes from Sina via AKShare."""
    if etfs is None:
        etfs = INDUSTRY_ETFS

    total_rows = 0
    success_count = 0

    for code, name in etfs.items():
        print(f"Downloading ETF {code} ({name})...", end=" ", flush=True)
        try:
            df = ak.fund_etf_hist_sina(symbol=code)
            if df is None or df.empty:
                print("EMPTY")
                continue

            df = df.rename(columns={"date": "trade_date"})
            df["etf_code"] = code

            cols = ["etf_code", "trade_date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]

            df["trade_date"] = df["trade_date"].astype(str)
            inserted = 0
            for _, row in df.iterrows():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO etf_daily_quotes (etf_code, trade_date, open, high, low, close, volume) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            row["etf_code"],
                            row["trade_date"],
                            float(row["open"]) if pd.notna(row.get("open")) else None,
                            float(row["high"]) if pd.notna(row.get("high")) else None,
                            float(row["low"]) if pd.notna(row.get("low")) else None,
                            float(row["close"]) if pd.notna(row.get("close")) else None,
                            float(row["volume"]) if pd.notna(row.get("volume")) else None,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            total_rows += inserted
            success_count += 1
            print(f"OK, {len(df)} rows")

        except Exception as e:
            print(f"FAILED: {e}")
            logger.error(f"ETF {code} download failed: {e}")

        time.sleep(1)

    print(f"\nETF download summary: {success_count}/{len(etfs)} ETFs, {total_rows} total rows")


def download_valuation_data(conn: sqlite3.Connection, ts_codes: list[str]):
    """Download stock valuation (market cap) data from Baidu via AKShare."""
    total_rows = 0
    success_count = 0

    for i, ts_code in enumerate(ts_codes):
        code = ts_code.split(".")[0]
        print(f"[{i+1}/{len(ts_codes)}] Downloading valuation for {ts_code}...", end=" ", flush=True)
        try:
            df = ak.stock_zh_valuation_baidu(symbol=code, indicator="总市值", period="近一年")
            if df is None or df.empty:
                print("EMPTY")
                time.sleep(1)
                continue

            df = df.rename(columns={"date": "trade_date", "value": "total_market_cap"})
            df["ts_code"] = ts_code
            df["trade_date"] = df["trade_date"].astype(str)

            for _, row in df.iterrows():
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO stock_valuation (ts_code, trade_date, total_market_cap, circulating_market_cap) "
                        "VALUES (?, ?, ?, NULL)",
                        (
                            row["ts_code"],
                            row["trade_date"],
                            float(row["total_market_cap"]) if pd.notna(row.get("total_market_cap")) else None,
                        ),
                    )
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            total_rows += len(df)
            success_count += 1
            print(f"OK, {len(df)} rows")

        except Exception as e:
            print(f"FAILED: {e}")
            logger.error(f"Valuation {ts_code} download failed: {e}")

        time.sleep(1)

    print(f"\nValuation download summary: {success_count}/{len(ts_codes)} stocks, {total_rows} total rows")


def download_fundamental_data(conn: sqlite3.Connection, ts_codes: list[str]):
    """Download stock fundamental financial data from THS via AKShare."""
    total_rows = 0
    success_count = 0

    col_map = {
        "报告期": "report_date",
        "净利润": "net_profit",
        "净利润同比增长率": "net_profit_yoy",
        "营业总收入": "revenue",
        "营业总收入同比增长率": "revenue_yoy",
        "基本每股收益": "eps",
        "每股净资产": "bvps",
        "净资产收益率": "roe",
        "销售净利率": "net_margin",
        "资产负债率": "debt_ratio",
    }

    for i, ts_code in enumerate(ts_codes):
        code = ts_code.split(".")[0]
        print(f"[{i+1}/{len(ts_codes)}] Downloading fundamentals for {ts_code}...", end=" ", flush=True)
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按年度")
            if df is None or df.empty:
                print("EMPTY")
                time.sleep(1)
                continue

            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            df["ts_code"] = ts_code
            if "report_date" in df.columns:
                df["report_date"] = df["report_date"].astype(str)

            schema_cols = [
                "ts_code", "report_date", "net_profit", "net_profit_yoy",
                "revenue", "revenue_yoy", "eps", "bvps", "roe",
                "gross_margin", "net_margin", "debt_ratio",
            ]

            for _, row in df.iterrows():
                values = []
                for col in schema_cols:
                    if col in row.index and pd.notna(row.get(col)):
                        val = row[col]
                        if col in ("ts_code", "report_date"):
                            values.append(str(val))
                            continue
                        if isinstance(val, bool):
                            values.append(None)
                            continue
                        try:
                            if isinstance(val, str):
                                val = val.replace(",", "").replace("亿", "").replace("%", "")
                                if val in ("False", "True", "--", ""):
                                    values.append(None)
                                else:
                                    values.append(float(val))
                            else:
                                values.append(float(val))
                        except (ValueError, TypeError):
                            values.append(None)
                    else:
                        values.append(None)

                if values[1] is None:
                    continue

                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO stock_fundamental "
                        "(ts_code, report_date, net_profit, net_profit_yoy, revenue, revenue_yoy, "
                        "eps, bvps, roe, gross_margin, net_margin, debt_ratio) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        tuple(values),
                    )
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            inserted = conn.execute(
                "SELECT COUNT(*) FROM stock_fundamental WHERE ts_code=?", (ts_code,)
            ).fetchone()[0]
            total_rows += inserted
            success_count += 1
            print(f"OK, {len(df)} rows ({inserted} inserted)")

        except Exception as e:
            print(f"FAILED: {e}")
            logger.error(f"Fundamental {ts_code} download failed: {e}")

        time.sleep(1)

    print(f"\nFundamental download summary: {success_count}/{len(ts_codes)} stocks, {total_rows} total rows")


def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    print("\n===== Downloading ETF Daily Quotes =====")
    download_etf_data(conn)

    ts_codes = [r[0] for r in conn.execute("SELECT DISTINCT ts_code FROM daily_quotes").fetchall()]
    print(f"\nFound {len(ts_codes)} stocks in daily_quotes table")

    if ts_codes:
        print("\n===== Downloading Stock Valuation Data =====")
        download_valuation_data(conn, ts_codes)

        print("\n===== Downloading Stock Fundamental Data =====")
        download_fundamental_data(conn, ts_codes)

    conn.close()
    print("\nAll done!")


if __name__ == "__main__":
    main()
