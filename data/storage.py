"""数据存储层"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging

from config.settings import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataStorage:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DB_PATH
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=config.DB_ECHO
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )

    @contextmanager
    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise
        finally:
            session.close()

    def save_stock_list(self, df: pd.DataFrame):
        """保存股票列表"""
        if df.empty:
            return

        with self.engine.connect() as conn:
            df.to_sql(
                "stock_list",
                conn,
                if_exists="replace",
                index=False
            )
        logger.info(f"保存股票列表成功，共 {len(df)} 条")

    def save_daily_quotes(self, df: pd.DataFrame):
        if df.empty:
            return

        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text("""
                    INSERT OR REPLACE INTO daily_quotes
                    (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                    VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :pre_close, :change, :pct_chg, :vol, :amount)
                """), row.to_dict())
            conn.commit()
        logger.info(f"保存日线数据成功，共 {len(df)} 条")

    def get_daily_quotes(
        self,
        ts_code: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """查询日线行情"""
        query = "SELECT * FROM daily_quotes WHERE ts_code = :ts_code"
        params = {"ts_code": ts_code}

        if start_date:
            query += " AND trade_date >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND trade_date <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY trade_date ASC"

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                text(query),
                conn,
                params=params
            )
        return df

    def get_latest_trade_date(self, ts_code: str) -> str:
        """获取最新交易日期"""
        query = """
            SELECT MAX(trade_date) as max_date
            FROM daily_quotes
            WHERE ts_code = :ts_code
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"ts_code": ts_code}).fetchone()
            return result[0] if result and result[0] else None

    def exists_daily_quotes(self, ts_code: str) -> bool:
        """检查是否已有日线数据"""
        query = "SELECT COUNT(*) as cnt FROM daily_quotes WHERE ts_code = :ts_code"

        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"ts_code": ts_code}).fetchone()
            return result[0] > 0

    def get_stock_info(self, ts_code: str) -> pd.DataFrame:
        """获取股票基本信息"""
        query = "SELECT * FROM stock_list WHERE ts_code = :ts_code"

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                text(query),
                conn,
                params={"ts_code": ts_code}
            )
        return df
