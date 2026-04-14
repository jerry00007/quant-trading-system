# A股量化交易系统 v2.0

基于 Python 的个人量化交易系统，支持数据获取、策略回测、实时监控、个股分析等功能。

## 项目结构

```
jiaoyixitong/
├── config/               # 配置管理
│   ├── settings.py          # 配置管理
│   └── scheduler.py          # 定时任务调度器
├── data/                   # 数据层
│   ├── base_source.py       # 数据源抽象接口
│   ├── downloader.py         # AKShare 实现
│   ├── tushare_source.py    # Tushare 实现
│   ├── netease_163_source.py # 网易163实现
│   ├── source_manager.py     # 多数据源管理器
│   └── storage.py           # 数据存储
├── strategies/             # 策略模块
│   ├── registry.py          # 策略注册表
│   ├── ma_strategy.py       # 双均线交叉策略
│   ├── etf_rotation_strategy.py # ETF轮动策略
│   ├── enhanced_chip_strategy.py # 增强筹码策略
│   ├── ml_stock_strategy.py # ML选股策略
│   ├── multifactor_strategy.py # 多因子选股策略
│   └── factors.py           # 技术因子库
├── web/                    # Web服务
│   └── app.py              # FastAPI后端
├── scripts/                # 工具脚本
├── main.py                 # CLI入口
└── requirements.txt         # 依赖
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

### 3. 启动Web服务

```bash
bash start.sh
```

服务地址: http://localhost:8787

## 功能特性

### 数据层
- [x] 数据获取（AKShare + Tushare 双数据源）
- [x] 定时任务调度（APScheduler）
- [x] 批量数据插入优化
- [x] 数据缓存机制

### 策略层
- [x] 策略注册机制（统一接口 + Pydantic验证）
- [x] ETF轮动策略
- [x] 增强筹码策略
- [x] ML选股策略（支持模型持久化）
- [x] 多因子选股策略
- [x] 技术因子库

### 分析层
- [x] 个股详情API
- [x] 技术指标API（MA/MACD/RSI/KDJ/CCI/布林带）
- [x] K线数据API
- [x] 市场状态检测

### 回测
- [x] 简单回测引擎
- [x] 多策略对比
- [x] 绩效指标计算

## 技术栈

- Python 3.10+
- SQLite / SQLAlchemy
- FastAPI + Uvicorn
- APScheduler（定时任务）
- AKShare / Tushare（数据源）
- scikit-learn（ML选股）
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
