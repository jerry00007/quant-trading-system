"""ML选股模块 — 生产化版本

使用 HistGradientBoosting 预测个股短期涨跌，输出每日选股列表。
特征: 动量、波动率、RSI、成交量比、均线偏离等。
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "stock_data.db"

TRAIN_WINDOW = 500
# Holding period: model predicts 1-day direction, hold for a few days
# to capture momentum while avoiding excessive turnover
HOLDING_PERIOD = 5
TOP_N = 10
FEATURE_COLS = [
    "momentum_20", "volatility_20", "return_5d", "volatility_5d",
    "price_to_ma20", "ma5_to_ma20", "vol_to_vol_ma",
    "price_to_high20", "volume_ratio", "rsi_14",
]


def compute_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    with np.errstate(divide='ignore', invalid='ignore'):
        return _compute_features_inner(df, lookback)


def _compute_features_inner(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.sort_values("trade_date").reset_index(drop=True)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    volume = df["vol"].values.astype(float)
    n = len(close)

    daily_ret = np.zeros(n)
    daily_ret[1:] = (close[1:] - close[:-1]) / close[:-1]

    momentum_20 = np.zeros(n)
    momentum_20[lookback:] = (close[lookback:] - close[:-lookback]) / close[:-lookback]

    volatility_20 = pd.Series(daily_ret).rolling(lookback, min_periods=1).std().fillna(0).values

    ret_5 = np.zeros(n)
    ret_5[5:] = (close[5:] - close[:-5]) / close[:-5]

    volatility_5d = pd.Series(daily_ret).rolling(5, min_periods=1).std().fillna(0).values

    ma20 = pd.Series(close).rolling(lookback, min_periods=1).mean().fillna(0).values
    price_to_ma20 = np.where(ma20 > 1e-8, (close - ma20) / ma20, 0.0)

    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().fillna(0).values
    ma5_to_ma20 = np.where(ma20 > 1e-8, (ma5 - ma20) / ma20, 0.0)

    vol_sma = pd.Series(volatility_20).rolling(lookback, min_periods=1).mean().fillna(0).values
    vol_to_vol_ma = np.where(vol_sma > 1e-8, volatility_20 / vol_sma, 0.0)

    high_20 = pd.Series(high).rolling(lookback, min_periods=1).max().fillna(0).values
    price_to_high20 = np.where(high_20 > 1e-8, (close - high_20) / high_20, 0.0)

    vol_ma20 = pd.Series(volume).rolling(lookback, min_periods=1).mean().fillna(0).values
    volume_ratio = np.where(vol_ma20 > 1e-8, volume / vol_ma20, 0.0)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
    rs = np.where(avg_loss > 1e-8, avg_gain / avg_loss, 100.0)
    rsi_14 = 100.0 - 100.0 / (1.0 + rs)

    # label[i] = whether close[i+1] > close[i], i.e., next day's return
    # Avoids data leakage: features at row i only use data up to close[i]
    label = np.zeros(n)
    label[:-1] = np.where(daily_ret[1:] > 0, 1.0, 0.0)
    label[-1] = np.nan  # no label for last row

    result = pd.DataFrame({
        "trade_date": df["trade_date"].values,
        "ts_code": df["ts_code"].values if "ts_code" in df.columns else "",
        "close": close,
        "open": df["open"].values.astype(float),
        "momentum_20": momentum_20,
        "volatility_20": volatility_20,
        "return_5d": ret_5,
        "volatility_5d": volatility_5d,
        "price_to_ma20": price_to_ma20,
        "ma5_to_ma20": ma5_to_ma20,
        "vol_to_vol_ma": vol_to_vol_ma,
        "price_to_high20": price_to_high20,
        "volume_ratio": volume_ratio,
        "rsi_14": rsi_14,
        "label": label,
    })

    result = result.replace([np.inf, -np.inf], 0.0).fillna(0)
    return result


def train_and_predict(df_features: pd.DataFrame, as_of_idx: int,
                      train_window: int = TRAIN_WINDOW,
                      holding_period: int = HOLDING_PERIOD) -> Optional[dict]:
    train_start = max(0, as_of_idx - train_window)
    train_data = df_features.iloc[train_start:as_of_idx]

    if len(train_data) < 200:
        return None

    X_train = train_data[FEATURE_COLS].values
    y_train = train_data["label"].values

    if len(np.unique(y_train)) < 2:
        return None

    model = HistGradientBoostingClassifier(
        max_iter=100, max_depth=4, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
    )

    try:
        model.fit(X_train, y_train)
    except Exception:
        return None

    current_row = df_features.iloc[as_of_idx]
    X_current = current_row[FEATURE_COLS].values.reshape(1, -1)

    pred = model.predict(X_current)[0]
    proba = model.predict_proba(X_current)[0]

    return {
        "prediction": int(pred),
        "prob_up": round(float(proba[1] if len(proba) > 1 else proba[0]), 4),
        "date": str(current_row["trade_date"]),
        "close": float(current_row["close"]),
        "features_snapshot": {col: round(float(current_row[col]), 4) for col in FEATURE_COLS},
    }


def generate_daily_stock_picks(conn, as_of_date: str = None, top_n: int = TOP_N) -> List[dict]:
    import sqlite3

    stocks = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM daily_quotes ORDER BY ts_code"
    ).fetchall()]

    if not as_of_date:
        row = conn.execute("SELECT MAX(trade_date) FROM daily_quotes").fetchone()
        as_of_date = row[0] if row and row[0] else None
    if not as_of_date:
        return []

    picks = []

    for ts_code in stocks:
        df = pd.read_sql(
            "SELECT * FROM daily_quotes WHERE ts_code=? AND trade_date <= ? ORDER BY trade_date",
            conn, params=(ts_code, as_of_date)
        )
        if len(df) < TRAIN_WINDOW + 50:
            continue

        df["ts_code"] = ts_code
        features = compute_features(df)

        last_idx = len(features) - 1
        result = train_and_predict(features, last_idx)
        if result is None:
            continue

        result["ts_code"] = ts_code
        result["name"] = _get_stock_name(ts_code)
        picks.append(result)

    picks.sort(key=lambda x: x["prob_up"], reverse=True)
    return picks[:top_n]


def run_ml_backtest_for_stock(conn, ts_code: str, start_date: str = None,
                               end_date: str = None,
                               initial_capital: float = 100_000) -> Optional[dict]:
    df = pd.read_sql(
        "SELECT * FROM daily_quotes WHERE ts_code=? ORDER BY trade_date",
        conn, params=(ts_code,)
    )
    if start_date:
        df = df[df["trade_date"] >= start_date]
    if end_date:
        df = df[df["trade_date"] <= end_date]

    if len(df) < TRAIN_WINDOW + HOLDING_PERIOD + 50:
        return None

    df["ts_code"] = ts_code
    features = compute_features(df)
    features = features.reset_index(drop=True)

    capital = initial_capital
    shares = 0
    entry_price = 0
    portfolio_values = []
    portfolio_dates = []
    trades = []

    i = TRAIN_WINDOW
    while i < len(features) - 1:
        result = train_and_predict(features, i)
        if result is None:
            portfolio_values.append(capital + shares * features.iloc[i]["close"])
            portfolio_dates.append(str(features.iloc[i]["trade_date"]))
            i += HOLDING_PERIOD
            continue

        current_date = str(features.iloc[i]["trade_date"])
        next_idx = min(i + 1, len(features) - 1)
        exec_price = float(features.iloc[next_idx]["open"])

        if shares == 0 and result["prediction"] == 1 and result["prob_up"] > 0.55:
            alloc = capital * 0.95
            new_shares = int(alloc / exec_price) // 100 * 100
            if new_shares >= 100:
                cost = new_shares * exec_price * 1.0003
                if cost <= capital:
                    capital -= cost
                    shares = new_shares
                    entry_price = exec_price
                    trades.append({
                        "date": current_date, "action": "buy",
                        "price": exec_price, "shares": shares,
                    })

        elif shares > 0:
            pnl = (exec_price / entry_price - 1)
            should_sell = (result["prediction"] == 0 or pnl < -0.08 or pnl > 0.20)
            if should_sell:
                revenue = shares * exec_price * (1 - 0.0003 - 0.0005)
                capital += revenue
                trades.append({
                    "date": current_date, "action": "sell",
                    "price": exec_price, "shares": shares,
                    "pnl_pct": round(pnl * 100, 2),
                })
                shares = 0
                entry_price = 0

        for j in range(i, min(i + HOLDING_PERIOD, len(features))):
            val = capital + shares * float(features.iloc[j]["close"])
            portfolio_values.append(val)
            portfolio_dates.append(str(features.iloc[j]["trade_date"]))

        i += HOLDING_PERIOD

    if shares > 0:
        last_price = float(features.iloc[-1]["close"])
        capital += shares * last_price * (1 - 0.0003 - 0.0005)
        trades.append({"date": portfolio_dates[-1], "action": "sell",
                       "price": last_price, "shares": shares,
                       "pnl_pct": round((last_price / entry_price - 1) * 100, 2)})

    if len(portfolio_values) < 50:
        return None

    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    max_dd = float(dd.min() * 100)
    total_return = float((pv[-1] / initial_capital - 1) * 100)

    returns = np.diff(pv) / pv[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)) if len(returns) > 1 else 0

    return {
        "ts_code": ts_code,
        "name": _get_stock_name(ts_code),
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "total_trades": len(trades),
        "win_trades": sum(1 for t in trades if t["action"] == "sell" and t.get("pnl_pct", 0) > 0),
        "sell_trades": sum(1 for t in trades if t["action"] == "sell"),
        "final_capital": round(float(pv[-1]), 2),
        "portfolio_dates": portfolio_dates,
        "portfolio_values": [round(float(v), 2) for v in pv],
    }


def _get_stock_name(ts_code: str) -> str:
    import sqlite3
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT name FROM stock_list WHERE ts_code=? LIMIT 1",
            (ts_code,)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return ts_code
