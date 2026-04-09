"""配置管理模块"""
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_DIR = DATA_DIR / "sqlite"
LOGS_DIR = PROJECT_ROOT / "logs"

SQLITE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = SQLITE_DIR / "stock_data.db"


class Config:
    """配置类"""

    DB_PATH = str(DB_PATH)
    DB_ECHO = False
    AKSHARE_TIMEOUT = 30
    DEFAULT_ADJUST = "qfq"
    BATCH_SIZE = 100
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def load_env():
    """从环境变量加载配置"""
    config = Config()

    if os.getenv("DB_PATH"):
        config.DB_PATH = os.getenv("DB_PATH")
    if os.getenv("DB_ECHO"):
        config.DB_ECHO = os.getenv("DB_ECHO").lower() == "true"
    if os.getenv("LOG_LEVEL"):
        config.LOG_LEVEL = os.getenv("LOG_LEVEL")

    return config


config = load_env()
