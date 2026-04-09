"""切换到 Tushare 并下载数据"""
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
    """切换到 Tushare 并下载数据"""
    print("=" * 60)
    print("切换到 Tushare 数据源")
    print("=" * 60)

    DATA_SOURCES = [
        DataSourceConfig(
            source_type="tushare",
            token="REDACTED_TUSHARE_TOKEN",
            enabled=True,
            priority=1
        ),
        DataSourceConfig(
            source_type="akshare",
            enabled=True,
            priority=2
        ),
    ]

    manager = DataSourceManager(DATA_SOURCES)

    print(f"\n已配置数据源: {manager.get_available_sources()}")
    print(f"当前激活数据源: {manager.active_source}")

    print("\n" + "-" * 60)
    print("切换到 Tushare...")
    print("-" * 60)

    success = manager.switch_source("tushare")

    if success:
        print(f"✓ 成功切换到: {manager.get_active_source().get_source_name()}")

        source = manager.get_active_source()

        stocks = ["600519", "000858", "600000"]
        start_date = "20240101"
        end_date = "20241231"

        print("\n" + "-" * 60)
        print(f"开始下载 {len(stocks)} 只股票数据...")
        print("-" * 60)

        storage = DataStorage()

        for symbol in stocks:
            try:
                print(f"\n正在下载: {symbol}")

                df = source.get_daily_quotes(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )

                if df.empty:
                    print(f"  ✗ 获取失败，数据为空")
                    continue

                print(f"  ✓ 获取成功，共 {len(df)} 条数据")
                print(f"  数据列: {df.columns.tolist()}")
                print(f"  数据预览:\n{df.head(3).to_string(index=False)}")

                storage.save_daily_quotes(df)
                print(f"  ✓ 保存到数据库成功")

            except Exception as e:
                print(f"  ✗ 下载失败: {e}")

        print("\n" + "=" * 60)
        print("下载完成！")
        print("=" * 60)

    else:
        print("✗ 切换失败")
        print("请检查 Tushare token 是否正确")


if __name__ == "__main__":
    main()
