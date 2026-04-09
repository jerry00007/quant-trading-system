"""下载中证100全部成分股 5 年历史数据"""
import sys
from pathlib import Path
import time
import json
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import akshare as ak
import tushare as ts
import pandas as pd
from data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

TUSHARE_TOKEN = "REDACTED_TUSHARE_TOKEN"
START_DATE = "20210409"
END_DATE = "20260409"
REQUEST_INTERVAL = 1.5
PROGRESS_FILE = PROJECT_ROOT / "data" / "download_progress.json"
BATCH_SIZE = 45
BATCH_PAUSE = 70


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def get_zz100_stocks():
    """获取中证100成分股列表"""
    logger.info("正在获取中证100成分股列表...")
    df = ak.index_stock_cons(symbol="000903")
    codes = df["品种代码"].tolist()
    names = df["品种名称"].tolist()

    stocks = []
    for code, name in zip(codes, names):
        if code.startswith("6"):
            ts_code = f"{code}.SH"
        else:
            ts_code = f"{code}.SZ"
        stocks.append({"ts_code": ts_code, "code": code, "name": name})

    logger.info(f"获取到中证100成分股 {len(stocks)} 只")
    return stocks


def download_stock(pro, storage, ts_code, name, start_date, end_date):
    """下载单只股票数据"""
    try:
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            logger.warning(f"  {ts_code} ({name}): 无数据")
            return False

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
        storage.save_daily_quotes(df)
        logger.info(f"  ✓ {ts_code} ({name}): {len(df)} 条")
        return True

    except Exception as e:
        logger.error(f"  ✗ {ts_code} ({name}): {e}")
        return False


def main():
    print("=" * 70)
    print("  中证100成分股历史数据批量下载")
    print(f"  数据范围: {START_DATE} ~ {END_DATE}")
    print(f"  数据源: Tushare")
    print(f"  请求间隔: {REQUEST_INTERVAL}s")
    print("=" * 70)

    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    storage = DataStorage()

    stocks = get_zz100_stocks()
    progress = load_progress()

    completed = set(progress["completed"])
    failed_list = progress["failed"]

    pending = [s for s in stocks if s["ts_code"] not in completed]
    logger.info(f"待下载: {len(pending)} 只, 已完成: {len(completed)} 只, 之前失败: {len(failed_list)} 只")

    if not pending:
        logger.info("所有股票已下载完成！")
        return

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for idx, stock in enumerate(pending, 1):
        ts_code = stock["ts_code"]
        name = stock["name"]

        print(f"\n[{len(completed) + idx}/{len(stocks)}] {ts_code} - {name}")

        ok = download_stock(pro, storage, ts_code, name, START_DATE, END_DATE)

        if ok:
            success_count += 1
            completed.add(ts_code)
            progress["completed"] = list(completed)
        else:
            fail_count += 1
            if ts_code not in failed_list:
                failed_list.append(ts_code)
            progress["failed"] = failed_list

        save_progress(progress)

        elapsed = time.time() - start_time
        speed = idx / elapsed * 60 if elapsed > 0 else 0
        remaining = (len(pending) - idx) / speed if speed > 0 else 0
        logger.info(f"  进度: {idx}/{len(pending)} | 速度: {speed:.0f}只/分 | 预计剩余: {remaining:.0f}分钟")

        if idx < len(pending):
            time.sleep(REQUEST_INTERVAL)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("  下载完成")
    print("=" * 70)
    print(f"  本次成功: {success_count} 只")
    print(f"  本次失败: {fail_count} 只")
    print(f"  累计完成: {len(completed)}/{len(stocks)} 只")
    print(f"  用时: {total_time / 60:.1f} 分钟")

    # 统计数据库总量
    db_count = storage.engine.connect().execute(
        __import__("sqlalchemy").text("SELECT COUNT(*) FROM daily_quotes")
    ).fetchone()[0]
    print(f"  数据库总记录数: {db_count} 条")

    if failed_list:
        print(f"\n  失败股票: {failed_list}")
        print("  可重新运行脚本自动重试")

    print("=" * 70)


if __name__ == "__main__":
    main()
