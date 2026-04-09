"""下载中证100热门股票历史数据（5年）"""
import sys
from pathlib import Path
import time
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.source_manager import DataSourceManager, DataSourceConfig
from data.storage import DataStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("批量下载中证100热门股票数据")
    print("=" * 60)

    CZ100_STOCKS = [
        "600000.SH",
        "600519.SH",
        "000001.SZ",
        "000002.SZ",
        "000858.SZ",
        "600036.SH",
        "601318.SH",
        "600276.SH",
        "000063.SZ",
        "600309.SH",
        "000333.SZ",
        "601888.SH",
        "000725.SZ",
        "600887.SH",
        "000651.SZ",
        "600585.SH",
        "600030.SH",
        "000100.SZ",
        "600031.SH",
        "000568.SZ",
        "600690.SH",
        "000876.SZ",
        "601328.SH",
        "600016.SH",
        "600104.SH",
        "000527.SZ",
    ]

    start_date = "20200101"
    end_date = "20241231"

    DATA_SOURCES = [
        DataSourceConfig(
            source_type="tushare",
            token="REDACTED_TUSHARE_TOKEN",
            enabled=True,
            priority=1
        ),
    ]

    manager = DataSourceManager(DATA_SOURCES)

    print(f"数据源: {manager.get_active_source().get_source_name()}")
    print(f"准备下载 {len(CZ100_STOCKS)} 只股票")
    print(f"时间范围: {start_date} - {end_date}")
    print(f"数据量: 约 5 年历史数据")
    print("-" * 60)

    source = manager.get_active_source()
    storage = DataStorage()

    total_downloaded = 0
    success_count = 0
    fail_count = 0

    start_time = time.time()

    for idx, symbol in enumerate(CZ100_STOCKS, 1):
        try:
            print(f"\n[{idx}/{len(CZ100_STOCKS)}] 正在下载: {symbol}")

            df = source.get_daily_quotes(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df.empty:
                print(f"  ✗ 获取失败，数据为空")
                fail_count += 1
            else:
                print(f"  ✓ 获取成功，共 {len(df)} 条数据")
                print(f"  数据列: {df.columns.tolist()}")

                storage.save_daily_quotes(df)
                print(f"  ✓ 保存到数据库成功")

                total_downloaded += len(df)
                success_count += 1

            elapsed = time.time() - start_time
            print(f"  已用时间: {elapsed:.1f} 秒")

            if idx < len(CZ100_STOCKS) - 1:
                print(f"  等待 2 秒后继续（避免请求过频）...")
                time.sleep(2)

        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            fail_count += 1

            if idx < len(CZ100_STOCKS) - 1:
                print(f"  等待 2 秒后继续（避免请求过频）...")
                time.sleep(2)

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("下载统计:")
    print("-" * 60)
    print(f"  成功: {success_count} 只股票")
    print(f"  失败: {fail_count} 只股票")
    print(f"  总数据量: {total_downloaded} 条")
    print(f"  总用时: {total_time/60:.1f} 分钟")
    print("=" * 60)

    print("\n✓ 数据已保存到 SQLite 数据库")
    print("可以使用以下命令查询数据:")
    print("  python main.py query 600000.SH")
    print("  python main.py query 000001.SZ")
    print("=" * 60)


if __name__ == "__main__":
    main()
