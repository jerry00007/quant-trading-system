# A股量化交易系统

基于 Python 的个人量化交易系统，支持数据获取、策略回测、实时监控等功能。

## 项目结构

```
jiaoyixitong/
├── config/               # 配置管理
├── data/                # 数据层
│   ├── base_source.py      # 数据源抽象接口
│   ├── downloader.py         # AKShare 实现
│   ├── tushare_source.py   # Tushare 实现
│   ├── source_manager.py    # 多数据源管理器
│   ├── storage.py           # 数据存储
│   └── __init__.py
├── backtest/             # 回测模块
│   ├── simple_backtest.py  # 简单回测引擎
│   └── __init__.py
├── strategies/            # 策略模块
│   ├── ma_strategy.py       # 双均线交叉策略
│   └── __init__.py
├── examples/              # 示例代码
│   ├── backtest_example.py  # 回测示例
│   └── data_source_demo.py # 数据源示例
├── scripts/               # 工具脚本
│   ├── init_db.py          # 数据库初始化
│   ├── batch_download.py    # 批量下载
│   └── __init__.py
├── tests/                # 测试
├── logs/                 # 日志文件
├── main.py               # CLI 入口
├── requirements.txt        # 依赖
└── README.md             # 文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化数据库

```bash
python scripts/init_db.py
```

### 3. 使用 CLI 工具

```bash
python main.py init                 # 初始化数据库
python main.py daily 600000      # 拉取日线数据
python main.py query 600000       # 查询数据
```

## 功能特性

- [x] 数据获取（AKShare）
- [x] 本地数据存储（SQLite）
- [x] 数据获取
- [x] 双数据源支持（AKShare + Tushare）
- [x] 数据源抽象层（统一接口）
- [x] 数据源管理器（优先级切换）
- [x] 策略框架（双均线交叉）
- [x] 回测引擎（简单回测）
- [ ] 技术指标计算
- [ ] 高级策略
- [ ] 参数优化
- [ ] 实时监控

## 技术栈

- Python 3.10+
- SQLite / SQLAlchemy
- AKShare / Tushare（数据源）
- Click（CLI）
- Rich（终端美化）
- Pandas / NumPy

## 数据源说明

### 支持的数据源

| 数据源 | 类型 | 优先级 | 说明 |
|--------|------|--------|------|
| Tushare | 基础接口 | 1 | 需要积分，稳定 |
| AKShare | 完整接口 | 2 | 免费，功能丰富 |
| 网易163 | CSV接口 | 3 | 备用，稳定 |

### 数据源切换

系统按优先级自动选择数据源（数字越小优先级越高）。

**示例配置**：
```python
from data.base_source import DataSourceConfig
from data.source_manager import DataSourceManager

DATA_SOURCES = [
    DataSourceConfig(
        source_type="netease_163",
        enabled=True,
        priority=3
    ),
]

manager = DataSourceManager(DATA_SOURCES)
source = manager.get_active_source()
```

### 数据源使用

```python
from data.source_manager import DataSourceManager

DATA_SOURCES = [
    DataSourceConfig(
        source_type="tushare",
        token="REDACTED_TUSHARE_TOKEN",
        enabled=True,
        priority=1
    ),
]

manager = DataSourceManager(DATA_SOURCES)
source = manager.get_active_source()

df = source.get_daily_quotes(
    symbol="600000",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"
)
```

## 知识库

详细文档请查看 Obsidian 知识库 `jydoc`。

## 架构设计参考

参考项目：https://github.com/akfamily/akquant
借鉴设计：混合架构、模块化、高性能

## 许可证

MIT
