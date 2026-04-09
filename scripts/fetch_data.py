"""数据拉取脚本"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.downloader import AKShareDownloader
from data.storage import DataStorage
from datetime import datetime, timedelta


def fetch_daily_data(symbol: str, days: int = 365):
    """拉取日线数据"""
    downloader = AKShareDownloader()
    storage = DataStorage()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")

    print(f"拉取 {symbol} 日线数据: {start_date_str} - {end_date_str}")

    df = downloader.get_daily_quotes(
        symbol=symbol,
        start_date=start_date_str,
        end_date=end_date_str,
        adjust="qfq"
    )

    if not df.empty:
        print(f"数据列: {df.columns.tolist()}")
        print(f"数据预览:\n{df.head()}")
    else:
        print("未获取到数据")
        return

    print(f"\n数据获取成功，共 {len(df)} 条")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="拉取股票数据")
    parser.add_argument("symbol", help="股票代码，如 600000")
    parser.add_argument("--days", type=int, default=365, help="拉取天数，默认365天")

    args = parser.parse_args()

    fetch_daily_data(args.symbol, args.days)
