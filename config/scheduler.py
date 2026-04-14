"""定时任务调度器"""
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)


class DataUpdateScheduler:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def update_etf_data(self) -> List[str]:
        """更新ETF数据"""
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

        conn = self._get_conn()
        updated = []

        for etf in ETF_POOL:
            code = etf["code"]
            try:
                last_row = conn.execute(
                    "SELECT MAX(trade_date) FROM etf_daily_quotes WHERE etf_code=?",
                    (code,)
                ).fetchone()
                start = last_row[0] if last_row and last_row[0] else "20200101"
                start_dt = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)
                start_str = start_dt.strftime("%Y%m%d")

                if start_dt > datetime.now():
                    updated.append(f"{etf['name']}({code}): 无需更新")
                    continue

                symbol = code[2:]
                df = ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start_str, adjust="qfq")
                if df is None or df.empty:
                    updated.append(f"{etf['name']}({code}): 无数据")
                    continue

                col_map = {
                    "日期": "trade_date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df["etf_code"] = code

                rows_to_insert = [
                    (
                        code,
                        str(row.get("trade_date", "")),
                        float(row.get("open", 0)),
                        float(row.get("high", 0)),
                        float(row.get("low", 0)),
                        float(row.get("close", 0)),
                        float(row.get("volume", 0)),
                        float(row.get("amount", 0)),
                    )
                    for _, row in df.iterrows()
                ]

                conn.executemany(
                    "INSERT OR REPLACE INTO etf_daily_quotes (etf_code, trade_date, open, high, low, close, volume, amount) VALUES (?,?,?,?,?,?,?,?)",
                    rows_to_insert,
                )
                conn.commit()
                updated.append(f"{etf['name']}({code}): +{len(df)}条")
            except Exception as e:
                logger.warning(f"更新ETF {code} 失败: {e}")
                updated.append(f"{etf['name']}({code}): 失败 - {str(e)[:30]}")

        return updated

    def update_stock_data(self, batch_size: int = 100) -> dict:
        """批量更新股票数据
        
        Args:
            batch_size: 每批更新的股票数量
            
        Returns:
            更新结果统计
        """
        conn = self._get_conn()

        stocks = [r[0] for r in conn.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchall()]

        if not stocks:
            return {"total": 0, "success": 0, "failed": 0, "errors": ["股票列表为空"]}

        results = {"total": len(stocks), "success": 0, "failed": 0, "errors": []}

        today = datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

        for i, ts_code in enumerate(stocks):
            try:
                last_row = conn.execute(
                    "SELECT MAX(trade_date) FROM daily_quotes WHERE ts_code=?", (ts_code,)
                ).fetchone()
                start = last_row[0] if last_row and last_row[0] else "20200101"

                if start >= yesterday:
                    results["success"] += 1
                    continue

                df = ak.stock_zh_a_hist(symbol=ts_code[:6], period="daily", start_date=start.replace("-", ""), end_date=today, adjust="qfq")
                if df is None or df.empty:
                    results["success"] += 1
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
                    "涨跌额": "change",
                    "换手率": "turnover",
                }
                df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
                df["ts_code"] = ts_code

                if "trade_date" in df.columns:
                    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

                rows_to_insert = [
                    (
                        row.get("ts_code", ts_code),
                        str(row.get("trade_date", "")),
                        float(row.get("open", 0)),
                        float(row.get("high", 0)),
                        float(row.get("low", 0)),
                        float(row.get("close", 0)),
                        float(row.get("pre_close", row.get("close", 0))),
                        float(row.get("change", 0)),
                        float(row.get("pct_chg", 0)),
                        float(row.get("vol", 0)),
                        float(row.get("amount", 0)),
                    )
                    for _, row in df.iterrows()
                ]

                conn.executemany(
                    "INSERT OR REPLACE INTO daily_quotes (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    rows_to_insert,
                )
                conn.commit()
                results["success"] += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"股票数据更新进度: {i+1}/{len(stocks)}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{ts_code}: {str(e)[:30]}")
                logger.warning(f"更新股票 {ts_code} 失败: {e}")

        return results

    def update_realtime_quotes(self) -> dict:
        """获取实时行情快照"""
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return {"success": False, "error": "无数据"}

            conn = self._get_conn()
            cache_key = "realtime_quotes"
            import json

            conn.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, data) VALUES (?, datetime('now'))",
                (cache_key, json.dumps(df.to_dict(), ensure_ascii=False)),
            )
            conn.commit()

            return {"success": True, "count": len(df)}
        except Exception as e:
            logger.warning(f"获取实时行情失败: {e}")
            return {"success": False, "error": str(e)}

    def cleanup_old_cache(self) -> int:
        """清理过期的缓存"""
        conn = self._get_conn()
        conn.execute("DELETE FROM api_cache WHERE created_at < datetime('now', '-7 days')")
        deleted = conn.total_changes
        conn.commit()
        return deleted


_scheduler: Optional[DataUpdateScheduler] = None


def get_scheduler() -> DataUpdateScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = DataUpdateScheduler()
    return _scheduler


def shutdown_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.close()
        _scheduler = None
