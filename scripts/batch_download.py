"""批量下载股票数据到数据库"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.downloader import AKShareDownloader
from data.storage import DataStorage
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


STOCKS_TO_DOWNLOAD = [
    "600000",
    "000001",
    "000002",
    "600519",
    "000858",
]


def download_and_save(symbol: str, days: int = 365):
    """下载并保存单只股票数据"""
    downloader = AKShareDownloader()
    storage = DataStorage()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    logger.info(f"开始下载 {symbol} 数据 ({start_date_str} - {end_date_str})")

    try:
        df = downloader.get_daily_quotes(
            symbol=symbol,
            start_date=start_date_str,
            end_date=end_date_str,
            adjust="qfq"
        )

        if df.empty:
            logger.warning(f"股票 {symbol} 无数据")
            return False

        storage.save_daily_quotes(df)
        logger.info(f"✓ {symbol} 保存成功，共 {len(df)} 条")
        return True

    except Exception as e:
        logger.error(f"✗ {symbol} 下载失败: {e}")
        return False


def main():
    """批量下载多只股票"""
    logger.info("=" * 50)
    logger.info("开始批量下载股票数据")
    logger.info("=" * 50)

    success_count = 0
    fail_count = 0

    for symbol in STOCKS_TO_DOWNLOAD:
        if download_and_save(symbol, days=365):
            success_count += 1
        else:
            fail_count += 1

    logger.info("=" * 50)
    logger.info(f"下载完成！成功: {success_count}, 失败: {fail_count}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
