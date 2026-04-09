"""验证数据库中的数据"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.storage import DataStorage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("验证数据库中的数据")
    print("=" * 60)

    storage = DataStorage()

    stocks = ["600000.SH", "000858.SZ", "600519.SH"]

    for symbol in stocks:
        print(f"\n股票: {symbol}")
        print("-" * 60)

        df = storage.get_daily_quotes(symbol)

        if df.empty:
            print("✗ 无数据")
        else:
            print(f"✓ 数据量: {len(df)} 条")
            print(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
            print(f"  最新 5 条数据:")
            print(df.tail(5)[['trade_date', 'close', 'pct_chg']].to_string(index=False))

    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
