"""ML选股实验: 基于sklearn的梯度提升预测ETF/股票短期收益"""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report


DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"
INITIAL_CAPITAL = 100_000


def generate_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """基于历史K线生成机器学习特征 (向量化版本)"""
    np.seterr(divide='ignore', invalid='ignore')
    df = df.sort_values("trade_date").reset_index(drop=True)

    result = pd.DataFrame()
    result["trade_date"] = df["trade_date"]
    result["close"] = df["close"].values
    result["open"] = df["open"].values
    result["vol"] = df["vol"].values

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["vol"].values.astype(float)
    n = len(close)

    # Daily returns
    daily_ret = np.zeros(n)
    daily_ret[1:] = (close[1:] - close[:-1]) / close[:-1]

    # 20-day momentum
    momentum_20 = np.zeros(n)
    momentum_20[lookback:] = (close[lookback:] - close[:-lookback]) / close[:-lookback]
    result["momentum_20"] = momentum_20

    # 20-day rolling std of daily returns
    result["volatility_20"] = pd.Series(daily_ret).rolling(lookback, min_periods=1).std().fillna(0).values

    # 5-day return
    ret_5 = np.zeros(n)
    ret_5[5:] = (close[5:] - close[:-5]) / close[:-5]
    result["return_5d"] = ret_5

    # 5-day rolling std
    result["volatility_5d"] = pd.Series(daily_ret).rolling(5, min_periods=1).std().fillna(0).values

    # MA20 and price_to_ma20
    ma20 = pd.Series(close).rolling(lookback, min_periods=1).mean().fillna(0).values
    result["price_to_ma20"] = np.where(ma20 > 1e-8, (close - ma20) / ma20, 0.0)

    # MA5 and ma5_to_ma20
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().fillna(0).values
    result["ma5_to_ma20"] = np.where(ma20 > 1e-8, (ma5 - ma20) / ma20, 0.0)

    # Volatility to volatility MA
    vol_series = pd.Series(result["volatility_20"].values)
    vol_sma = vol_series.rolling(lookback, min_periods=1).mean().fillna(0).values
    result["vol_to_vol_ma"] = np.where(vol_sma > 1e-8, result["volatility_20"].values / vol_sma, 0.0)

    # Price to 20-day high
    high_20 = pd.Series(high).rolling(lookback, min_periods=1).max().fillna(0).values
    result["price_to_high20"] = np.where(high_20 > 1e-8, (close - high_20) / high_20, 0.0)

    # Volume ratio
    vol_ma20 = pd.Series(volume).rolling(lookback, min_periods=1).mean().fillna(0).values
    result["volume_ratio"] = np.where(vol_ma20 > 1e-8, volume / vol_ma20, 0.0)

    # RSI 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
    rs = np.where(avg_loss > 1e-8, avg_gain / avg_loss, 100.0)
    result["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # Label: next day up or down
    result["label"] = 0.0
    future_ret = np.zeros(n)
    future_ret[1:] = daily_ret[1:]  # next day's return
    result["label"] = np.where(future_ret > 0, 1.0, 0.0)

    # Replace inf/nan
    result = result.replace([np.inf, -np.inf], 0.0).fillna(0)
    np.seterr(divide='warn', invalid='warn')
    return result


def run_ml_backtest(conn: sqlite3.Connection, symbol: str, 
                    train_window: int = 500, test_window: int = 20) -> dict:
    """对单个标的运行ML选股回测"""
    df = pd.read_sql(
        f"SELECT * FROM daily_quotes WHERE ts_code='{symbol}' ORDER BY trade_date",
        conn
    )
    if len(df) < train_window + test_window + 50:
        return None
    
    features_df = generate_features(df)
    features_df = features_df.reset_index(drop=True)
    
    feature_cols = ["momentum_20", "volatility_20", "return_5d", "volatility_5d",
                    "price_to_ma20", "ma5_to_ma20", "vol_to_vol_ma", 
                    "price_to_high20", "volume_ratio"]
    
    n = len(features_df)
    capital = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    portfolio_values = []
    portfolio_dates = []
    predictions = []
    
    i = train_window
    while i < n - test_window:
        train_data = features_df.iloc[i-train_window:i]
        test_data = features_df.iloc[i:i+test_window]
        
        X_train = train_data[feature_cols].values
        y_train = train_data["label"].values
        
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        
        try:
            model.fit(X_train, y_train)
        except Exception:
            i += test_window
            continue
        
        X_test = test_data[feature_cols].values
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        if shares == 0 and y_pred[0] == 1:
            exec_price = test_data.iloc[0]["open"]
            shares = int(capital / exec_price) // 100 * 100
            if shares >= 100:
                capital -= shares * exec_price * 1.0003
                entry_price = exec_price
        elif shares > 0:
            exec_price = test_data.iloc[0]["open"]
            if y_pred[0] == 0 or (entry_price > 0 and 
                                   (exec_price / entry_price - 1) < -0.08):
                capital += shares * exec_price * (1 - 0.0003 - 0.0005)
                shares = 0
                entry_price = 0
        
        for j in range(len(test_data)):
            val = capital
            if shares > 0:
                val += shares * test_data.iloc[j]["close"]
            portfolio_values.append(val)
            portfolio_dates.append(test_data.iloc[j]["trade_date"])
        
        i += test_window
    
    if len(portfolio_values) < 100:
        return None
    
    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    max_dd = dd.min() * 100
    
    total_return = (pv[-1] / INITIAL_CAPITAL - 1) * 100
    
    returns = np.diff(pv) / pv[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
    
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_capital": pv[-1],
        "symbol": symbol,
    }


def main():
    print("=" * 70)
    print("  ML选股实验: HistGradientBoosting分类器")
    print("=" * 70)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    stocks = pd.read_sql(
        "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code LIMIT 10",
        conn
    )["ts_code"].tolist()

    print(f"测试标的: {len(stocks)} 只股票")
    print(f"训练窗口: 500天  测试窗口: 20天")
    print("=" * 70)
    
    results = []
    for sym in stocks:
        r = run_ml_backtest(conn, sym)
        if r:
            results.append(r)
            print(f"  {sym}: 收益{r['total_return']:+.1f}% 夏普{r['sharpe']:.2f} DD{r['max_drawdown']:.1f}%")
    
    if results:
        avg_return = np.mean([r["total_return"] for r in results])
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_dd = np.mean([r["max_drawdown"] for r in results])
        wins = sum(1 for r in results if r["total_return"] > 0)
        
        print()
        print("=" * 70)
        print("  ML选股汇总")
        print("=" * 70)
        print(f"  股票数量:     {len(results)}")
        print(f"  平均收益:     {avg_return:+.1f}%")
        print(f"  平均夏普:     {avg_sharpe:.2f}")
        print(f"  平均最大回撤: {avg_dd:.1f}%")
        print(f"  正收益股票:   {wins}/{len(results)} ({wins/len(results)*100:.0f}%)")
        
        best = max(results, key=lambda x: x["total_return"])
        worst = min(results, key=lambda x: x["total_return"])
        print(f"\n  最佳: {best['symbol']} ({best['total_return']:+.1f}%)")
        print(f"  最差: {worst['symbol']} ({worst['total_return']:+.1f}%)")
    else:
        print("  无足够数据生成结果")
    
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    main()
