"""数据源管理器"""
from typing import List, Optional
import logging

from .base_source import BaseDataSource, DataSourceConfig
from .tushare_source import TushareDataSource
from .downloader import AKShareDataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceManager:
    """数据源管理器，支持多数据源切换"""

    def __init__(self, configs: List[DataSourceConfig]):
        self.sources = {}

        for config in configs:
            if config.source_type == "akshare":
                self.sources["akshare"] = AKShareDataSource()
                logger.info(f"注册 AKShare 数据源，优先级: {config.priority}")
            elif config.source_type == "tushare":
                self.sources["tushare"] = TushareDataSource(token=config.token)
                logger.info(f"注册 Tushare 数据源，优先级: {config.priority}")

        self.active_source = self._get_highest_priority_source()

    def _get_highest_priority_source(self) -> str:
        if not self.sources:
            return "akshare"

        enabled_sources = [(k, s) for k, s in self.sources.items() if isinstance(s, BaseDataSource)]
        if not enabled_sources:
            return list(self.sources.keys())[0]

        enabled_sources.sort(key=lambda x: getattr(x[1], 'priority', 99))
        source_name = enabled_sources[0][0]
        logger.info(f"激活数据源: {source_name}")
        return source_name

    def get_active_source(self) -> BaseDataSource:
        if self.active_source not in self.sources:
            logger.warning(f"激活的数据源 {self.active_source} 不可用，切换到最高优先级数据源")
            self.active_source = self._get_highest_priority_source()

        return self.sources[self.active_source]

    def switch_source(self, source_type: str) -> bool:
        if source_type not in self.sources:
            logger.error(f"未知的数据源: {source_type}")
            return False

        self.active_source = source_type
        logger.info(f"切换到数据源: {source_type}")
        return True

    def add_source(self, source_type: str, source: BaseDataSource) -> bool:
        if source_type in self.sources:
            logger.warning(f"数据源 {source_type} 已存在")
            return False

        self.sources[source_type] = source
        logger.info(f"添加数据源: {source_type}")
        return True

    def get_available_sources(self) -> List[str]:
        return list(self.sources.keys())
