"""测试网易163数据源"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.source_manager import DataSourceManager, DataSourceConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 直接导入网易163数据源
from data.netease_163_source import Netease163DataSource


def test_netease_source():
    """测试网易163数据源"""
    print("=" * 60)
    print("测试网易163数据源")
    print("=" * 60)

    DATA_SOURCES = [
        DataSourceConfig(
            source_type="tushare",
            enabled=True,
            priority=1
        ),
        DataSourceConfig(
            source_type="netease_163",
            enabled=True,
            priority=3
        ),
    ]

    manager = DataSourceManager(DATA_SOURCES)

    print(f"\n已配置数据源: {manager.get_available_sources()}")
    print(f"当前激活数据源: {manager.active_source}")

    print("\n" + "-" * 60)
    print("测试 1: 检查数据源可用性")
    print("-" * 60)

    available = manager.get_available_sources()
    for source_type in available:
        print(f"  - {source_type}: {'✓ 可用' if manager.sources.get(source_type) else '✗ 不可用'}")

    print("\n" + "-" * 60)
    print("测试 2: 获取 Tushare 数据（默认）")
    print("-" * 60)

    symbol = "600519"
    start_date = "20240101"
    end_date = "20241231"

    try:
        source_tushare = manager.get_active_source()
        df1 = source_tushare.get_daily_quotes(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df1.empty:
            print(f"Tushare 获取失败，数据为空")
        else:
            print(f"Tushare 获取成功，共 {len(df1)} 条数据")
            print(f"数据列: {df1.columns.tolist()}")
            if not df1.empty:
                print(f"\n数据预览:\n{df1.head(3)}")

    except Exception as e:
        print(f"Tushare 测试失败: {e}")

    print("\n" + "-" * 60)
    print("测试 3: 切换到网易163并获取数据")
    print("-" * 60)

    try:
        success = manager.switch_source("netease_163")
        if success:
            source_netease = manager.get_active_source()
            print(f"已切换到数据源: {source_netease.get_source_name()}")

            df2 = source_netease.get_daily_quotes(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df2.empty:
                print(f"网易163 获取失败，数据为空")
            else:
                print(f"网易163 获取成功，共 {len(df2)} 条数据")
                print(f"数据列: {df2.columns.tolist()}")
                if not df2.empty:
                    print(f"\n数据预览:\n{df2.head(3)}")
        else:
            print(f"切换到网易163失败")
    except Exception as e:
        print(f"网易163 测试失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_netease_source()
