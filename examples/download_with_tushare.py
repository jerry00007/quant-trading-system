"""使用 Tushare 下载股票数据"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.source_manager import DataSourceManager, DataSourceConfig
from data.storage import DataStorage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """使用 Tushare 下载股票数据"""
    print("=" * 60)
    print("使用 Tushare 下载股票数据")
    print("=" * 60)

    DATA_SOURCES = [
        DataSourceConfig(
            source_type="tushare",
            token="REDACTED_TUSHARE_TOKEN",
            enabled=True,
            priority=1
        ),
    ]

    manager = DataSourceManager(DATA_SOURCES)

    print(f"当前激活数据源: {manager.get_active_source().get_source_name()}")
    print(f"数据源: Tushare")
    print(f"Token: {DATA_SOURCES[0].token}")

    stocks = [
        "600000.SH",
        "000858.SZ",
        "600519.SH"
    ]

    start_date = "20240101"
    end_date = "20241231"

    print(f"\n准备下载 {len(stocks)} 只股票数据")
    print(f"时间范围: {start_date} - {end_date}")
    print("-" * 60)

    source = manager.get_active_source()
    storage = DataStorage()

    total_downloaded = 0
    success_count = 0
    fail_count = 0

    for symbol in stocks:
        try:
            print(f"\n[{stocks.index(symbol) + 1}/{len(stocks)}] 正在下载: {symbol}")

            df = source.get_daily_quotes(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df.empty:
                print(f"  ✗ 数据为空")
                fail_count += 1
                continue

            print(f"  ✓ 获取成功，共 {len(df)} 条数据")
            print(f"  数据列: {df.columns.tolist()}")
            print(f"   数据日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")

            storage.save_daily_quotes(df)
            print(f"  ✓ 保存到数据库成功")

            total_downloaded += len(df)
            success_count += 1

        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print("下载统计:")
    print("-" * 60)
    print(f"  成功: {success_count} 只股票")
    print(f"  失败: {fail_count} 只股票")
    print(f"  总数据量: {total_downloaded} 条")
    print("=" * 60)

    print("\n✓ 数据已保存到 SQLite 数据库")
    print("  可以使用 python main.py query <股票代码> 查询数据")


if __name__ == "__main__":
    main()
