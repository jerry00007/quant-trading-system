"""诊断 Tushare 数据源问题"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tushare as ts
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tushare_connection():
    """测试 Tushare 连接和数据获取"""
    print("=" * 60)
    print("Tushare 诊断测试")
    print("=" * 60)

    token = "REDACTED_TUSHARE_TOKEN"

    print(f"\n使用 Token: {token}")
    print("-" * 60)

    ts.set_token(token)
    pro = ts.pro_api()

    print("\n测试 1: 检查积分")
    print("-" * 60)
    try:
        user_info = pro.query('user', 'userInfo')
        print(f"用户信息: {user_info}")

        if '积分' in user_info.columns:
            points = user_info['积分'].values[0] if not user_info['积分'].empty else 0
            print(f"当前积分: {points}")
            print(f"积分状态: {'✓ 可以下载数据' if points >= 100 else '✗ 积分不足'}")

        if '积分' not in user_info.columns:
            print("⚠ 无法获取积分信息")

    except Exception as e:
        print(f"✗ 查询用户信息失败: {e}")

    print("\n" + "-" * 60)
    print("测试 2: 尝试获取日线数据")
    print("-" * 60)

    symbol = "600000.SH"
    start_date = "20240101"
    end_date = "20241231"

    try:
        df = pro.daily(
            ts_code=symbol,
            start_date=start_date,
            end_date=end_date,
            adj="qfq"
        )

        print(f"✓ 接口调用成功")
        print(f"  获取数据行数: {len(df)}")
        print(f"  数据列: {df.columns.tolist()}")

        if not df.empty:
            print(f"\n数据预览（前 5 行）:")
            print(df.head().to_string(index=False))
        else:
            print("⚠ 数据为空")

    except Exception as e:
        print(f"✗ 获取数据失败: {e}")
        print(f"错误类型: {type(e).__name__}")

    print("\n" + "-" * 60)
    print("测试 3: 尝试获取股票列表")
    print("-" * 60)

    try:
        df_list = pro.stock_basic(ts_code="", list_status="L")
        print(f"✓ 获取股票列表成功，共 {len(df_list)} 只股票")
        print(f"  列: {df_list.columns.tolist()}")

        if not df_list.empty:
            print(f"\n股票预览（前 5 只）:")
            print(df_list.head().to_string(index=False))
        else:
            print("⚠ 数据为空")

    except Exception as e:
        print(f"✗ 获取股票列表失败: {e}")

    print("\n" + "=" * 60)
    print("诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_tushare_connection()
