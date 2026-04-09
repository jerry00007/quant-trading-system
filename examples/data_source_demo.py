"""数据源配置示例"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.base_source import DataSourceConfig
from data.source_manager import DataSourceManager


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
    DataSourceConfig(
        source_type="netease_163",
        enabled=True,
        priority=3
    ),
]


def main():
    manager = DataSourceManager(DATA_SOURCES)

    print("=" * 60)
    print("数据源配置示例")
    print("=" * 60)
    print(f"\n已配置数据源: {manager.get_available_sources()}")
    print(f"当前激活数据源: {manager.active_source}")
    print(f"获取数据源实例: {manager.get_active_source().get_source_name()}")

    print("\n" + "-" * 60)
    print("功能说明:")
    print("-" * 60)
    print("1. 支持 Tushare、AKShare、网易163 三数据源")
    print("2. 按优先级自动选择数据源（数字越小优先级越高）")
    print("3. 统一的数据格式标准化")
    print("4. 数据源切换功能")
    print("5. 扩展性：可轻松添加新的数据源")
    print("-" * 60)


if __name__ == "__main__":
    main()
