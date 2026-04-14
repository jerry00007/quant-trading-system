#!/usr/bin/env python3
"""
全市场A股批量下载脚本
- 数据源: EastMoney(股票列表) + Tencent(历史K线)
- 覆盖: 全量A股 (~5800只)
- 数据: 前复权日线，最近3年
- 限速: 0.3秒/请求，守护进程模式
"""
import os
import sys
import sqlite3
import json
import time
import requests
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import config

# 清除代理
for k in list(os.environ.keys()):
    if any(x in k.lower() for x in ['proxy', 'http', 'https']):
        del os.environ[k]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'batch_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 配置 ==========
MAX_WORKERS = 3           # 并发线程数
REQUEST_DELAY = 0.3       # 请求间隔(秒)
BATCH_SIZE = 50           # 每批处理多少只股票后写入数据库
RETRY_TIMES = 3           # 重试次数
HISTORY_YEARS = 3         # 数据年限
DAYS_PER_REQUEST = 640   # 腾讯API每次最多返回约640天

STOCK_DB = config.DB_PATH
PROGRESS_FILE = PROJECT_ROOT / 'data' / 'download_progress.json'

# ========== 数据源 ==========
SINA_LIST_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeDataSimple?page={page}&num=1000&sort=symbol&asc=1&node={node}&symbol=&_s_r_a=page"
TENANT_KLINE_URL = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayhfq&param=%s,day,%s,%s,%d,qfq"
SINA_NODES = [
    ('sh_a', '上证A股'),
    ('sz_a', '深证A股'),
]

# ========== 全局状态 ==========
g_stop_flag = False
g_lock = Lock()
g_progress = {
    'total': 0,
    'done': 0,
    'failed': [],
    'start_time': None,
    'last_update': None
}


def signal_handler(sig, frame):
    global g_stop_flag
    logger.info("收到停止信号，正在保存进度...")
    g_stop_flag = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ========== 工具函数 ==========
def normalize_ts_code(symbol):
    if '.' in symbol:
        return symbol
    s = symbol.lower()
    if s.startswith('sh'):
        return f"{symbol[2:]}.SH"
    elif s.startswith('sz'):
        return f"{symbol[2:]}.SZ"
    if symbol.startswith(('6', '5', '9')):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def normalize_tencent_code(ts_code):
    """将 '000001.SZ' 转换为 'sz000001'"""
    if ts_code.endswith('.SH'):
        return 'sh' + ts_code[:6]
    elif ts_code.endswith('.SZ'):
        return 'sz' + ts_code[:6]
    elif ts_code.startswith('sh') or ts_code.startswith('sz'):
        return ts_code
    else:
        # 假设是6开头为SH，其他为SZ
        if ts_code.startswith(('6', '5', '9')):
            return 'sh' + ts_code[:6]
        else:
            return 'sz' + ts_code[:6]


def save_progress():
    """保存进度到文件"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(g_progress, f, indent=2)


def load_progress():
    """加载进度"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return None


# ========== 数据获取 ==========
def fetch_stock_list_from_sina():
    """从新浪获取全市场股票列表"""
    logger.info("从新浪获取全市场股票列表...")
    
    all_stocks = []
    
    for node, node_name in SINA_NODES:
        page = 1
        while True:
            url = SINA_LIST_URL.format(page=page, node=node)
            try:
                r = requests.get(url, timeout=15)
                r.encoding = 'utf-8'
                data = json.loads(r.text)
                
                if not data:
                    break
                    
                for s in data:
                    symbol = s.get('symbol', '')
                    ts_code = normalize_ts_code(symbol)
                    all_stocks.append({
                        'ts_code': ts_code,
                        'symbol': symbol,
                        'name': s.get('name', ''),
                        'area': '',
                        'industry': '',
                        'market': '',
                        'list_date': '',
                    })
                
                logger.info(f"  {node_name} 第{page}页: +{len(data)} 只, 累计 {len(all_stocks)}")
                
                if len(data) < 1000:
                    break
                page += 1
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"获取{node_name}列表失败(第{page}页): {e}")
                time.sleep(2)
                continue
    
    seen = set()
    unique = []
    for s in all_stocks:
        if s['ts_code'] not in seen:
            seen.add(s['ts_code'])
            unique.append(s)
    
    logger.info(f"新浪返回共 {len(unique)} 只股票")
    return unique


def fetch_history_from_tencent(ts_code, start_date, end_date):
    code = normalize_tencent_code(ts_code)
    url = TENANT_KLINE_URL % (code, start_date, end_date, DAYS_PER_REQUEST)
    
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        
        if not text.startswith('kline_dayhfq='):
            return []
        
        json_str = text[len('kline_dayhfq='):]
        data = json.loads(json_str)
        
        all_stock_data = data.get('data', {})
        stock_data = all_stock_data.get(code, {})
        if not stock_data:
            return []
        
        qfq_data = stock_data.get('qfqday', [])
        return qfq_data
    except Exception as e:
        return []


def fetch_3years_data(ts_code):
    """获取某股票近3年前复权日线数据(分2次请求避免数据截断)"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    # 计算分割点: 约1.5年前
    mid_date = (datetime.now() - timedelta(days=545)).strftime('%Y-%m-%d')
    old_start = (datetime.now() - timedelta(days=HISTORY_YEARS * 365 + 30)).strftime('%Y-%m-%d')
    
    all_rows = []
    
    # 第一次: 较新数据
    rows1 = fetch_history_from_tencent(ts_code, mid_date, end_date)
    all_rows.extend(rows1)
    time.sleep(REQUEST_DELAY)
    
    # 第二次: 较旧数据
    rows2 = fetch_history_from_tencent(ts_code, old_start, mid_date)
    all_rows.extend(rows2)
    time.sleep(REQUEST_DELAY)
    
    # 去重并排序
    seen = set()
    unique_rows = []
    for row in all_rows:
        date_str = row[0]
        if date_str not in seen:
            seen.add(date_str)
            unique_rows.append(row)
    
    unique_rows.sort(key=lambda x: x[0])
    return unique_rows


def parse_kline_row(row, ts_code):
    """解析K线数据行"""
    try:
        trade_date = row[0]
        open_ = float(row[1]) if row[1] else 0
        close = float(row[2]) if row[2] else 0
        high = float(row[3]) if row[3] else 0
        low = float(row[4]) if row[4] else 0
        vol = float(row[5]) if row[5] else 0
        
        return {
            'ts_code': ts_code,
            'trade_date': trade_date,
            'open': open_,
            'close': close,
            'high': high,
            'low': low,
            'vol': vol,
            'amount': 0.0,
        }
    except (ValueError, IndexError):
        return None


# ========== 数据库操作 ==========
def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(STOCK_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS stock_list (
            ts_code TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            area TEXT,
            industry TEXT,
            market TEXT,
            list_date TEXT,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS daily_quotes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT,
            trade_date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL DEFAULT 0,
            change REAL DEFAULT 0,
            pct_chg REAL DEFAULT 0,
            vol REAL,
            amount REAL,
            UNIQUE(ts_code, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_quotes_code_date 
            ON daily_quotes(ts_code, trade_date);
        CREATE INDEX IF NOT EXISTS idx_daily_quotes_date 
            ON daily_quotes(trade_date);
    """)
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")


def save_stock_list(stocks):
    """保存股票列表到数据库"""
    conn = sqlite3.connect(STOCK_DB)
    
    # 使用REPLACE保证主键唯一
    for s in stocks:
        conn.execute("""
            INSERT OR REPLACE INTO stock_list 
            (ts_code, symbol, name, area, industry, market, list_date, update_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (s['ts_code'], s['symbol'], s['name'], s.get('area',''),
              s.get('industry',''), s.get('market',''), s.get('list_date','')))
    
    conn.commit()
    conn.close()
    logger.info(f"保存股票列表 {len(stocks)} 只到数据库")


def save_daily_quotes(rows_data, ts_code):
    """批量保存日线数据"""
    if not rows_data:
        return 0
    
    conn = sqlite3.connect(STOCK_DB)
    cursor = conn.cursor()
    inserted = 0
    
    for row in rows_data:
        parsed = parse_kline_row(row, ts_code)
        if parsed is None:
            continue
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO daily_quotes
                (ts_code, trade_date, open, high, low, close, vol, amount)
                VALUES (:ts_code, :trade_date, :open, :high, :low, :close, :vol, :amount)
            """, parsed)
            inserted += 1
        except Exception as e:
            pass
    
    conn.commit()
    conn.close()
    return inserted


def get_stocks_with_data():
    """获取已有日线数据的股票代码"""
    conn = sqlite3.connect(STOCK_DB)
    codes = [r[0] for r in conn.execute("SELECT DISTINCT ts_code FROM daily_quotes").fetchall()]
    conn.close()
    return set(codes)


def get_last_date(ts_code):
    """获取某股票最新数据日期"""
    conn = sqlite3.connect(STOCK_DB)
    row = conn.execute(
        "SELECT MAX(trade_date) FROM daily_quotes WHERE ts_code=?", (ts_code,)
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


# ========== 批量处理 ==========
def process_single_stock(ts_code):
    """处理单只股票: 获取3年数据并写入数据库"""
    global g_progress, g_stop_flag
    
    if g_stop_flag:
        return None
    
    try:
        # 获取3年数据
        rows = fetch_3years_data(ts_code)
        
        if not rows:
            return {'ts_code': ts_code, 'status': 'no_data', 'count': 0}
        
        # 写入数据库
        inserted = save_daily_quotes(rows, ts_code)
        
        return {'ts_code': ts_code, 'status': 'success', 'count': inserted}
        
    except Exception as e:
        logger.debug(f"处理{ts_code}异常: {e}")
        return {'ts_code': ts_code, 'status': 'error', 'count': 0, 'error': str(e)}


def run_batch_download(stock_list=None, force_update=False):
    """
    批量下载主函数
    stock_list: 股票列表(如果为None则从EastMoney获取)
    force_update: True=强制更新所有股票, False=只更新缺失的
    """
    global g_progress, g_lock
    
    init_db()
    
    if stock_list is None:
        stock_list = fetch_stock_list_from_sina()
    
    if not stock_list:
        logger.error("股票列表为空，退出")
        return
    
    # 2. 保存股票列表
    save_stock_list(stock_list)
    
    # 3. 确定要下载的股票
    existing_codes = get_stocks_with_data()
    all_codes = [s['ts_code'] for s in stock_list]
    
    if force_update:
        target_codes = all_codes
    else:
        target_codes = [c for c in all_codes if c not in existing_codes]
    
    logger.info(f"全量: {len(all_codes)} 只, 已有数据: {len(existing_codes)}, 需下载: {len(target_codes)}")
    
    # 4. 加载进度
    saved_progress = load_progress()
    if saved_progress and not force_update:
        done_set = set(saved_progress.get('done_list', []))
        g_progress = saved_progress
        g_progress['done_list'] = list(done_set)
        target_codes = [c for c in target_codes if c not in done_set]
        logger.info(f"从进度恢复: 已完成 {len(done_set)} 只, 剩余 {len(target_codes)} 只")
    
    g_progress['total'] = len(all_codes)
    g_progress['done'] = len(existing_codes)
    g_progress['start_time'] = datetime.now().isoformat()
    g_progress['failed'] = []
    g_progress['done_list'] = list(existing_codes)
    save_progress()
    
    # 5. 批量下载
    total_target = len(target_codes)
    success_count = 0
    failed_list = []
    batch_buffer = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_code = {
            executor.submit(process_single_stock, code): code 
            for code in target_codes
        }
        
        for future in as_completed(future_to_code):
            if g_stop_flag:
                logger.info("停止下载，保存进度...")
                save_progress()
                break
            
            code = future_to_code[future]
            try:
                result = future.result(timeout=30)
                if result and result['status'] == 'success':
                    success_count += 1
                    
                    with g_lock:
                        g_progress['done'] += 1
                        g_progress['done_list'].append(code)
                        g_progress['last_update'] = datetime.now().isoformat()
                        
                        # 定期保存进度
                        if g_progress['done'] % 100 == 0:
                            save_progress()
                            elapsed = time.time() - start_time
                            rate = g_progress['done'] / elapsed if elapsed > 0 else 0
                            eta = (total_target - g_progress['done']) / rate / 60 if rate > 0 else 0
                            logger.info(f"进度: {g_progress['done']}/{total_target}, "
                                       f"成功率: {success_count}/{g_progress['done']}, "
                                       f"预计剩余: {eta:.1f}分钟")
                else:
                    failed_list.append(code)
                    with g_lock:
                        g_progress['failed'].append(code)
                        
            except Exception as e:
                logger.debug(f"处理{code}超时/异常: {e}")
                failed_list.append(code)
                with g_lock:
                    g_progress['failed'].append(code)
    
    # 6. 最终统计
    save_progress()
    elapsed_total = time.time() - start_time
    
    conn = sqlite3.connect(STOCK_DB)
    r = conn.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT ts_code) FROM daily_quotes").fetchone()
    conn.close()
    
    logger.info("=" * 60)
    logger.info(f"批量下载完成!")
    logger.info(f"总耗时: {elapsed_total/60:.1f} 分钟")
    logger.info(f"成功: {success_count} 只")
    logger.info(f"失败: {len(failed_list)} 只")
    logger.info(f"数据库: {r[2]} 只股票, {r[0]} ~ {r[1]}")
    logger.info("=" * 60)


def show_status():
    """显示当前下载状态"""
    progress = load_progress()
    if not progress:
        logger.info("没有下载进度记录")
        return
    
    conn = sqlite3.connect(STOCK_DB)
    r = conn.execute("SELECT MIN(trade_date), MAX(trade_date), COUNT(DISTINCT ts_code) FROM daily_quotes").fetchone()
    conn.close()
    
    logger.info(f"下载状态:")
    logger.info(f"  股票总数: {progress.get('total', 0)}")
    logger.info(f"  已完成: {progress.get('done', 0)}")
    logger.info(f"  失败: {len(progress.get('failed', []))}")
    logger.info(f"  数据库: {r[2]} 只股票, {r[0]} ~ {r[1]}")
    
    if progress.get('failed'):
        logger.info(f"  失败列表(前20): {progress['failed'][:20]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='全市场A股批量下载')
    parser.add_argument('--force', action='store_true', help='强制重新下载所有股票')
    parser.add_argument('--status', action='store_true', help='仅显示下载状态')
    parser.add_argument('--daemon', action='store_true', help='守护进程模式(循环运行)')
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.daemon:
        logger.info("守护进程模式启动，每60秒检查并更新...")
        while True:
            run_batch_download()
            logger.info(f"等待60秒后重新检查... ({datetime.now()})")
            time.sleep(60)
    else:
        run_batch_download(force_update=args.force)
