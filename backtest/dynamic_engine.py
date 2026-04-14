"""动态回测引擎（向量化优化版）"""

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from config.settings import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
BACKTEST_DB = PROJECT_ROOT / "data" / "sqlite" / "backtest.db"

COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005
BOARD_LOT = 100
LIMIT_UP = 0.0997
LIMIT_DOWN = -0.0997
STOP_LOSS = -0.08
TRAILING_STOP = 0.10
TAKE_PROFIT = 0.25
MAX_POSITION_PCT = 0.15


@dataclass
class BacktestConfig:
    strategies: List[str]
    stock_pool: str
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000
    rebalance_days: int = 20
    max_positions: int = 10
    max_position_pct: float = 0.2

    def to_dict(self):
        return {
            "strategies": ",".join(self.strategies),
            "stock_pool": self.stock_pool,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "rebalance_days": self.rebalance_days,
            "max_positions": self.max_positions,
            "max_position_pct": self.max_position_pct,
        }


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    price: float
    shares: int
    amount: float
    reason: str


@dataclass
class DailySnapshot:
    date: str
    cash: float
    positions: Dict[str, dict]
    total_value: float


@dataclass
class StrategyResult:
    strategy_name: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    trades: List[Trade]
    daily_snapshots: List[DailySnapshot]
    equity_curve: List[float]


def init_backtest_db():
    BACKTEST_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(BACKTEST_DB))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT (datetime('now')),
            strategies TEXT,
            stock_pool TEXT,
            start_date TEXT,
            end_date TEXT,
            initial_capital REAL,
            rebalance_days INTEGER,
            max_positions INTEGER,
            max_position_pct REAL,
            status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            strategy_name TEXT,
            final_value REAL,
            total_return REAL,
            sharpe REAL,
            max_drawdown REAL,
            win_rate REAL,
            total_trades INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
        );
        CREATE TABLE IF NOT EXISTS backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            strategy_name TEXT,
            trade_date TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            shares INTEGER,
            amount REAL,
            reason TEXT,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
        );
        CREATE TABLE IF NOT EXISTS backtest_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            strategy_name TEXT,
            snapshot_date TEXT,
            cash REAL,
            total_value REAL,
            positions_json TEXT,
            FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
        );
    """)
    conn.commit()
    conn.close()


def get_stock_pool_codes(pool: str, conn: sqlite3.Connection) -> List[str]:
    if pool == "hs300":
        return _get_hs300_codes(conn)
    elif pool == "zz500":
        return _get_zz500_codes(conn)
    elif pool == "zz800":
        return _get_zz800_codes(conn)
    else:
        return _get_all_codes(conn)


def _get_hs300_codes(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT ts_code FROM stock_list LIMIT 300").fetchall()
    return [r[0] for r in rows]


def _get_zz500_codes(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT ts_code FROM stock_list LIMIT 500").fetchall()
    return [r[0] for r in rows]


def _get_zz800_codes(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT ts_code FROM stock_list LIMIT 800").fetchall()
    return [r[0] for r in rows]


def _get_all_codes(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT DISTINCT ts_code FROM stock_list ").fetchall()
    return [r[0] for r in rows]


class DynamicBacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.db_path = BACKTEST_DB
        self.run_id = str(uuid.uuid4())[:8]
        self._conn: Optional[sqlite3.Connection] = None
        self._market_conn: Optional[sqlite3.Connection] = None
        # Pre-computed feature cache (populated per strategy run)
        self._feat: Optional[pd.DataFrame] = None
        self._price_map: Dict[str, Dict[str, float]] = {}
        self._open_map: Dict[str, Dict[str, float]] = {}
        self._market_trend_cache: Dict[int, bool] = {}
        self._strategy_trailing_stop: Dict[str, float] = {
            "ma": 0.08,
            "multifactor": 0.10,
            "etf_rotation": 0.10,
            "chip": 0.08,
            "ml": 0.10,
        }

    def _get_trailing_stop(self, strategy_name: str) -> float:
        return self._strategy_trailing_stop.get(strategy_name, TRAILING_STOP)

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._market_conn:
            self._market_conn.close()
            self._market_conn = None

    def run(self) -> str:
        init_backtest_db()
        self._save_run_info()

        self._market_conn = sqlite3.connect(str(config.DB_PATH))
        codes = get_stock_pool_codes(self.config.stock_pool, self._market_conn)
        if not codes:
            logger.warning("股票池为空，使用全量数据")
            codes = _get_all_codes(self._market_conn)

        daily_data = self._load_daily_data(codes, self._market_conn)

        if daily_data.empty:
            self._market_conn.close()
            self._market_conn = None
            raise ValueError("没有加载到任何数据")

        trading_dates = sorted(daily_data["trade_date"].unique())
        trading_dates = [
            d
            for d in trading_dates
            if self.config.start_date <= d <= self.config.end_date
        ]

        if len(trading_dates) < self.config.rebalance_days:
            self._market_conn.close()
            self._market_conn = None
            raise ValueError(f"交易天数不足，需要至少{self.config.rebalance_days}天")

        self._build_price_maps(daily_data, trading_dates)
        self._precompute_features(daily_data, trading_dates)

        results = {}
        for strategy_name in self.config.strategies:
            logger.info(f"运行策略: {strategy_name}")
            result = self._run_single_strategy(strategy_name, daily_data, trading_dates)
            results[strategy_name] = result
            self._save_result(strategy_name, result)

        self._update_run_status("completed")
        self._market_conn.close()
        self._market_conn = None
        return self.run_id

    def _build_price_maps(self, daily_data: pd.DataFrame, trading_dates: List[str]):
        """Build date->(ts_code->price) dicts for O(1) lookup in main loop."""
        date_set = set(trading_dates)
        mask = daily_data["trade_date"].isin(date_set)
        sub = daily_data.loc[mask, ["trade_date", "ts_code", "close", "open"]]
        for date, grp in sub.groupby("trade_date"):
            self._price_map[date] = dict(zip(grp["ts_code"], grp["close"]))
            self._open_map[date] = dict(zip(grp["ts_code"], grp["open"]))

    def _precompute_features(self, daily_data: pd.DataFrame, trading_dates: List[str]):
        """Compute ALL features for ALL stocks in one vectorized pass.

        Stores a DataFrame ``self._feat`` sorted by (ts_code, trade_date) with
        columns needed by every strategy.  Strategies then do a single-row
        lookup per (stock, date) instead of per-stock loops.
        """
        df = daily_data.sort_values(["ts_code", "trade_date"]).copy()

        for col in ("close", "open", "high", "low", "vol"):
            df[col] = df[col].astype(np.float64)

        g = df.groupby("ts_code", sort=False)

        df["returns_1d"] = g["close"].transform(lambda s: s.pct_change(1))
        df["returns_5d"] = g["close"].transform(lambda s: s.pct_change(5))
        df["returns_20d"] = g["close"].transform(lambda s: s.pct_change(20))

        df["volatility_5d"] = g["returns_1d"].transform(
            lambda s: s.rolling(5, min_periods=1).std()
        )
        df["volatility_20d"] = g["returns_1d"].transform(
            lambda s: s.rolling(20, min_periods=1).std()
        )

        df["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=1).mean())
        df["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
        df["ma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=1).mean())

        df["vol_ma20"] = g["vol"].transform(
            lambda s: s.rolling(20, min_periods=1).mean()
        )
        df["volume_ratio"] = np.where(
            df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"], 1.0
        )

        df["high20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=1).max())
        df["price_to_high20"] = np.where(
            df["high20"] > 0, df["close"] / df["high20"] - 1.0, 0.0
        )

        df["delta"] = g["close"].transform(lambda s: s.diff())
        df["gain"] = g["delta"].transform(
            lambda s: s.clip(lower=0).rolling(14, min_periods=1).mean()
        )
        df["loss"] = g["delta"].transform(
            lambda s: (-s.clip(upper=0)).rolling(14, min_periods=1).mean()
        )
        df["rsi_14"] = 100.0 - 100.0 / (1.0 + df["gain"] / (df["loss"] + 1e-10))

        df["price_to_ma20"] = np.where(
            df["ma20"] > 0, df["close"] / df["ma20"] - 1.0, 0.0
        )
        df["ma5_to_ma20"] = np.where(df["ma20"] > 0, df["ma5"] / df["ma20"] - 1.0, 0.0)
        df["vol_to_vol_ma"] = np.where(
            df["vol_ma20"] > 0, df["vol"] / df["vol_ma20"] - 1.0, 0.0
        )

        df["low_75"] = g["low"].transform(lambda s: s.rolling(75, min_periods=1).min())
        df["high_75"] = g["high"].transform(
            lambda s: s.rolling(75, min_periods=1).max()
        )
        var7_raw = np.where(
            (df["high_75"] - df["low_75"]) / 100.0 > 1e-10,
            (df["close"] - df["low_75"]) / ((df["high_75"] - df["low_75"]) / 100.0),
            0.0,
        )
        var7_raw = np.nan_to_num(var7_raw, nan=0.0)

        var7_arr = var7_raw
        var8_arr = np.empty_like(var7_arr)
        var8s_arr = np.empty_like(var7_arr)
        zlcmq_arr = np.empty_like(var7_arr)

        codes = df["ts_code"].values
        boundaries = np.where(codes[:-1] != codes[1:])[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(codes)]])
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            seg = var7_arr[start:end]
            v8 = self._tdx_sma(seg, 20, 1)
            v8s = self._tdx_sma(v8, 15, 1)
            vara = 3.0 * v8 - 2.0 * v8s
            var8_arr[start:end] = v8
            var8s_arr[start:end] = v8s
            zlcmq_arr[start:end] = 100.0 - vara

        df["zlcmq"] = zlcmq_arr

        df["zlcmq_max5"] = g["zlcmq"].transform(
            lambda s: s.rolling(5, min_periods=1).max()
        )
        df["zlcmq_prev1"] = g["zlcmq"].transform(lambda s: s.shift(1))

        df["cumcount"] = g.cumcount()

        df["future_close_5d"] = g["close"].transform(lambda s: s.shift(-5))
        df["label_5d_up"] = (df["future_close_5d"] > df["close"]).astype(np.float64)

        df.drop(columns=["delta", "gain", "loss"], inplace=True, errors="ignore")

        self._feat = df

    def _load_daily_data(
        self, codes: List[str], market_conn: sqlite3.Connection
    ) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame()

        placeholders = ",".join(["?" for _ in codes])
        query = f"""
            SELECT ts_code, trade_date, open, high, low, close, vol
            FROM daily_quotes
            WHERE ts_code IN ({placeholders})
            AND '{self.config.start_date}' <= trade_date AND trade_date <= '{self.config.end_date}'
            ORDER BY ts_code, trade_date
        """
        df = pd.read_sql_query(query, market_conn, params=codes)
        return df

    def _run_single_strategy(
        self, strategy_name: str, daily_data: pd.DataFrame, trading_dates: List[str]
    ) -> StrategyResult:
        cash = [self.config.initial_capital]
        positions: Dict[str, dict] = {}
        all_trades: List[Trade] = []
        all_snapshots: List[DailySnapshot] = []
        equity_curve: List[float] = []
        bought_today: set = set()
        position_peak: Dict[str, float] = {}

        strategy_func = self._get_strategy_func(strategy_name)
        days_since_rebalance = 0

        market_trend_up = True
        market_ma20 = 0.0

        self._market_trend_cache.clear()
        close_by_date = {}
        for d in trading_dates:
            pm = self._price_map.get(d, {})
            if pm:
                close_by_date[d] = float(np.mean(list(pm.values())))

        for i, date in enumerate(trading_dates):
            prices = self._price_map.get(date, {})
            prev_prices = self._price_map.get(trading_dates[i - 1], {}) if i > 0 else {}

            total_value = cash[0] + sum(
                pos["shares"] * prices.get(ts_code, pos["entry_price"])
                for ts_code, pos in positions.items()
            )
            equity_curve.append(total_value)

            for ts_code, pos in positions.items():
                cur_price = prices.get(ts_code, pos["entry_price"])
                pos_value = pos["shares"] * cur_price
                if ts_code not in position_peak:
                    position_peak[ts_code] = pos_value
                else:
                    position_peak[ts_code] = max(position_peak[ts_code], pos_value)

            all_snapshots.append(
                DailySnapshot(
                    date=date,
                    cash=cash[0],
                    positions=dict(positions),
                    total_value=total_value,
                )
            )

            self._check_stop_loss(
                positions, prices, prev_prices, all_trades, cash, position_peak
            )
            self._check_take_profit(
                positions, prices, prev_prices, all_trades, cash, position_peak
            )
            self._check_trailing_stop(
                positions, prices, position_peak, all_trades, cash, strategy_name
            )

            days_since_rebalance += 1
            if days_since_rebalance >= self.config.rebalance_days:
                if i >= 20:
                    market_prices = [
                        close_by_date[d]
                        for d in trading_dates[max(0, i - 19) : i + 1]
                        if d in close_by_date
                    ]
                    if len(market_prices) >= 20:
                        market_ma20 = sum(market_prices[-20:]) / 20
                        market_avg = market_prices[-1]
                        market_trend_up = market_avg >= market_ma20
                    else:
                        market_trend_up = True
                else:
                    market_trend_up = True

                target_symbols = strategy_func(
                    daily_data, trading_dates, i, self.config.max_positions
                )
                if not market_trend_up:
                    target_symbols = []
                target_symbols = self._apply_risk_filters(
                    target_symbols, prices, prev_prices, positions, bought_today, date
                )
                self._rebalance(
                    positions,
                    target_symbols,
                    pd.DataFrame({"trade_date": [date]}),
                    prices,
                    cash,
                    all_trades,
                    strategy_name,
                    position_peak,
                )
                days_since_rebalance = 0

            new_bought: set = set()
            for ts_code in positions:
                entry_date = positions[ts_code]["entry_date"]
                if entry_date == date:
                    new_bought.add(ts_code)
            bought_today = new_bought

        return StrategyResult(
            strategy_name=strategy_name,
            initial_capital=self.config.initial_capital,
            final_value=equity_curve[-1] if equity_curve else cash[0],
            total_return=(
                (equity_curve[-1] if equity_curve else cash[0])
                / self.config.initial_capital
                - 1
            )
            * 100,
            sharpe=self._calculate_sharpe(equity_curve),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=self._calculate_win_rate(all_trades),
            total_trades=len(all_trades),
            trades=all_trades,
            daily_snapshots=all_snapshots,
            equity_curve=equity_curve,
        )

    def _get_strategy_func(self, strategy_name: str) -> Callable:
        if strategy_name == "ma":
            return self._ma_strategy
        elif strategy_name == "chip":
            return self._chip_strategy
        elif strategy_name == "ml":
            return self._ml_strategy
        elif strategy_name == "multifactor":
            return self._multifactor_strategy
        elif strategy_name == "etf_rotation":
            return self._etf_rotation_strategy
        else:
            return self._ma_strategy

    def _ma_strategy(
        self,
        daily_data: pd.DataFrame,
        trading_dates: List[str],
        date_idx: int,
        top_k: int,
    ) -> List[str]:
        """均线策略: MA5 > MA20 买入，按收盘价排序取前k"""
        date = trading_dates[date_idx]
        feat = self._feat
        mask = feat["trade_date"] == date
        day = feat.loc[mask].copy()

        day = day[day["cumcount"] >= 19]
        if day.empty:
            return []

        signals = day[day["ma5"] > day["ma20"]]
        if signals.empty:
            return []

        signals = signals.sort_values("close", ascending=False)
        return signals["ts_code"].head(top_k).tolist()

    def _chip_strategy(
        self,
        daily_data: pd.DataFrame,
        trading_dates: List[str],
        date_idx: int,
        top_k: int,
    ) -> List[str]:
        """筹码策略"""
        date = trading_dates[date_idx]
        feat = self._feat
        mask = feat["trade_date"] == date
        day = feat.loc[mask].copy()

        day = day[day["cumcount"] >= 74]
        if day.empty:
            return []

        cur_z_raw = day["zlcmq"].values
        valid = (~np.isnan(cur_z_raw)) & (cur_z_raw >= 60) & (cur_z_raw <= 92)
        day = day[valid]
        if day.empty:
            return []

        cur_z = day["zlcmq"].values
        z5_max = day["zlcmq_max5"].values
        z_prev = day["zlcmq_prev1"].values
        cur_close = day["close"].values
        cur_open = day["open"].values
        prev_close = np.where(
            day["returns_1d"].isna(),
            cur_close,
            cur_close / (1.0 + day["returns_1d"].fillna(0).values),
        )

        was_high_zone = z5_max >= 85
        is_declining = cur_z < z_prev
        not_too_fast = (z5_max - cur_z) <= 35
        is_stable = (cur_close >= cur_open) | (cur_close > prev_close)

        passed = day[was_high_zone & is_declining & not_too_fast & is_stable]
        if passed.empty:
            return []

        passed = passed.sort_values("zlcmq", ascending=False)
        return passed["ts_code"].head(top_k).tolist()

    @staticmethod
    def _tdx_sma(x, n, m):
        y = np.empty(len(x))
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = (m * x[i] + (n - m) * y[i - 1]) / n
        return y

    def _calculate_zlcmq(self, close, high, low):
        """Backward-compatible wrapper — pre-computed in _precompute_features."""
        c = np.array(close, dtype=float)
        h = np.array(high, dtype=float)
        lo = np.array(low, dtype=float)

        var5 = pd.Series(lo).rolling(75, min_periods=1).min().values
        var6 = pd.Series(h).rolling(75, min_periods=1).max().values
        var7 = (var6 - var5) / 100.0
        var7 = np.where(var7 > 1e-10, (c - var5) / var7, 0.0)
        var7 = np.nan_to_num(var7, nan=0.0)

        var8 = self._tdx_sma(var7, 20, 1)
        var8_s = self._tdx_sma(var8, 15, 1)
        vara = 3.0 * var8 - 2.0 * var8_s
        return 100.0 - vara

    def _ml_strategy(
        self,
        daily_data: pd.DataFrame,
        trading_dates: List[str],
        date_idx: int,
        top_k: int,
    ) -> List[str]:
        """ML策略: HistGradientBoosting分类器"""
        date = trading_dates[date_idx]
        feat = self._feat

        FEATURE_COLS = [
            "returns_20d",
            "volatility_20d",
            "returns_5d",
            "volatility_5d",
            "price_to_ma20",
            "ma5_to_ma20",
            "vol_to_vol_ma",
            "price_to_high20",
            "volume_ratio",
            "rsi_14",
        ]

        lookback_start = max(0, date_idx - 175)
        lookback_dates = set(trading_dates[lookback_start : date_idx + 1])
        mask = feat["trade_date"].isin(lookback_dates)
        pool = feat.loc[mask]
        pool = pool[pool["cumcount"] >= 49]
        if pool.empty:
            return []

        pool = pool.dropna(subset=FEATURE_COLS)
        if pool.shape[0] < 100:
            return []

        labels = pool["label_5d_up"].values
        has_label = ~np.isnan(labels)

        X_all = pool[FEATURE_COLS].values
        y_all = labels

        X_train = X_all[has_label]
        y_train = y_all[has_label]

        if len(X_train) < 50 or len(np.unique(y_train[~np.isnan(y_train)])) < 2:
            return []

        y_train = np.nan_to_num(y_train, nan=0.0)

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                max_iter=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=10,
                random_state=42,
            )
            model.fit(X_train, y_train)
        except Exception:
            return []

        day_mask = feat["trade_date"] == date
        day = feat.loc[day_mask].dropna(subset=FEATURE_COLS)
        if day.empty:
            return []

        X_score = day[FEATURE_COLS].values
        proba = model.predict_proba(X_score)
        if proba.shape[1] > 1:
            prob_up = proba[:, 1]
        else:
            prob_up = proba[:, 0]

        passed = day[prob_up > 0.45].copy()
        if passed.empty:
            return []

        passed["prob_up"] = prob_up[prob_up > 0.45]
        passed = passed.sort_values("prob_up", ascending=False)
        return passed["ts_code"].head(top_k).tolist()

    def _multifactor_strategy(
        self,
        daily_data: pd.DataFrame,
        trading_dates: List[str],
        date_idx: int,
        top_k: int,
    ) -> List[str]:
        """多因子策略: 市值 + 动量 + 质量 三因子"""
        date = trading_dates[date_idx]
        feat = self._feat
        mask = feat["trade_date"] == date
        day = feat.loc[mask].copy()

        day = day[day["cumcount"] >= 29]
        if day.empty:
            return []

        momentum = day["returns_20d"].values
        volatility = day["volatility_20d"].values
        quality = np.where(
            (volatility == 0) | np.isnan(volatility),
            0.0,
            1.0 / (volatility + 1e-6),
        )
        market_cap_proxy = day["close"].values

        scores = (
            0.4 * (-market_cap_proxy)
            + 0.3 * np.nan_to_num(momentum, nan=0.0)
            + 0.3 * quality
        )
        day["score"] = scores

        day = day.sort_values("score", ascending=False)
        return day["ts_code"].head(top_k).tolist()

    def _etf_rotation_strategy(
        self,
        daily_data: pd.DataFrame,
        trading_dates: List[str],
        date_idx: int,
        top_k: int,
    ) -> List[str]:
        date = trading_dates[date_idx]
        feat = self._feat
        mask = feat["trade_date"] == date
        day = feat.loc[mask].copy()

        day = day[day["cumcount"] >= 29]
        if day.empty:
            return []

        momentum = day["returns_20d"].values
        volatility = day["volatility_20d"].values
        quality = np.where(
            (volatility == 0) | np.isnan(volatility),
            0.0,
            1.0 / (volatility + 1e-6),
        )
        market_cap_proxy = day["close"].values

        scores = (
            0.5 * np.nan_to_num(momentum, nan=0.0)
            + 0.3 * quality
            + 0.2 * (-market_cap_proxy)
        )
        day["score"] = scores

        day = day.sort_values("score", ascending=False)
        return day["ts_code"].head(top_k).tolist()

    def _apply_risk_filters(
        self,
        target_symbols: List[str],
        prices: Dict[str, float],
        prev_prices: Dict[str, float],
        positions: Dict[str, dict],
        bought_today: set,
        date: str,
    ) -> List[str]:
        """涨跌停 + T+1 风控过滤"""
        filtered = []
        for ts_code in target_symbols:
            if ts_code in positions:
                continue
            if ts_code in bought_today:
                continue

            cur = prices.get(ts_code)
            prev = prev_prices.get(ts_code)

            if cur is None or prev is None or prev == 0:
                continue

            change_pct = (cur - prev) / prev
            # 涨停板不能买入（涨幅接近10%）
            if change_pct >= LIMIT_UP:
                continue
            # 跌停板不能卖出（但这里只过滤买入，所以只管涨停）
            filtered.append(ts_code)

        return filtered

    def _check_stop_loss(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        prev_prices: Dict[str, float],
        trades: List[Trade],
        cash: list,
        position_peak: Dict[str, float],
    ):
        to_sell = []
        for ts_code, pos in positions.items():
            if ts_code in prices:
                pnl_pct = (prices[ts_code] - pos["entry_price"]) / pos["entry_price"]
                prev = prev_prices.get(ts_code, pos["entry_price"])
                if prev > 0:
                    change_pct = (prices[ts_code] - prev) / prev
                    if change_pct <= -LIMIT_DOWN:
                        continue
                if pnl_pct <= STOP_LOSS:
                    to_sell.append((ts_code, pnl_pct))

        for ts_code, pnl_pct in to_sell:
            pos = positions.pop(ts_code)
            position_peak.pop(ts_code, None)
            price = prices.get(ts_code, pos["entry_price"])
            shares = pos["shares"]
            buy_cost = shares * pos["entry_price"] * (1 + COMMISSION_RATE)
            sell_revenue = shares * price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            cash[0] += sell_revenue
            trades.append(
                Trade(
                    date="",
                    symbol=ts_code,
                    action="sell",
                    price=price,
                    shares=shares,
                    amount=sell_revenue - buy_cost,
                    reason=f"止损 {pnl_pct * 100:.1f}%",
                )
            )

    def _check_take_profit(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        prev_prices: Dict[str, float],
        trades: List[Trade],
        cash: list,
        position_peak: Dict[str, float],
    ):
        to_sell = []
        for ts_code, pos in positions.items():
            if ts_code in prices:
                pnl_pct = (prices[ts_code] - pos["entry_price"]) / pos["entry_price"]
                prev = prev_prices.get(ts_code, pos["entry_price"])
                if prev > 0:
                    change_pct = (prices[ts_code] - prev) / prev
                    if change_pct >= LIMIT_UP:
                        continue
                if pnl_pct >= TAKE_PROFIT:
                    to_sell.append((ts_code, pnl_pct))

        for ts_code, pnl_pct in to_sell:
            pos = positions.pop(ts_code)
            position_peak.pop(ts_code, None)
            price = prices.get(ts_code, pos["entry_price"])
            shares = pos["shares"]
            buy_cost = shares * pos["entry_price"] * (1 + COMMISSION_RATE)
            sell_revenue = shares * price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            cash[0] += sell_revenue
            trades.append(
                Trade(
                    date="",
                    symbol=ts_code,
                    action="sell",
                    price=price,
                    shares=shares,
                    amount=sell_revenue - buy_cost,
                    reason=f"止盈 {pnl_pct * 100:.1f}%",
                )
            )

    def _check_trailing_stop(
        self,
        positions: Dict[str, dict],
        prices: Dict[str, float],
        position_peak: Dict[str, float],
        trades: List[Trade],
        cash: list,
        strategy_name: str,
    ):
        ts_threshold = self._get_trailing_stop(strategy_name)
        to_sell = []
        for ts_code, pos in positions.items():
            cur_price = prices.get(ts_code)
            if cur_price is None:
                continue
            pos_value = pos["shares"] * cur_price
            peak = position_peak.get(ts_code, pos_value)
            if peak > 0:
                drawdown = (pos_value - peak) / peak
                if drawdown <= -ts_threshold:
                    to_sell.append((ts_code, drawdown))

        for ts_code, drawdown in to_sell:
            pos = positions.pop(ts_code)
            position_peak.pop(ts_code, None)
            price = prices.get(ts_code, pos["entry_price"])
            shares = pos["shares"]
            buy_cost = shares * pos["entry_price"] * (1 + COMMISSION_RATE)
            sell_revenue = shares * price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            cash[0] += sell_revenue
            trades.append(
                Trade(
                    date="",
                    symbol=ts_code,
                    action="sell",
                    price=price,
                    shares=shares,
                    amount=sell_revenue - buy_cost,
                    reason=f"移动止损 {drawdown * 100:.1f}%",
                )
            )

    def _rebalance(
        self,
        positions: Dict[str, dict],
        target_symbols: List[str],
        day_data: pd.DataFrame,
        prices: Dict[str, float],
        cash: list,
        trades: List[Trade],
        strategy_name: str,
        position_peak: Dict[str, float],
    ):
        to_sell = [ts for ts in positions if ts not in target_symbols]
        for ts_code in to_sell:
            pos = positions.pop(ts_code)
            position_peak.pop(ts_code, None)
            price = prices.get(ts_code, pos["entry_price"])
            shares = pos["shares"]
            buy_cost = shares * pos["entry_price"] * (1 + COMMISSION_RATE)
            sell_revenue = shares * price * (1 - COMMISSION_RATE - STAMP_TAX_RATE)
            cash[0] += sell_revenue
            trades.append(
                Trade(
                    date=str(day_data["trade_date"].iloc[-1]),
                    symbol=ts_code,
                    action="sell",
                    price=price,
                    shares=shares,
                    amount=sell_revenue - buy_cost,
                    reason="调仓",
                )
            )

        available = cash[0]
        per_stock = min(
            self.config.initial_capital
            * min(self.config.max_position_pct, MAX_POSITION_PCT),
            available / len(target_symbols) if target_symbols else 0,
        )

        for ts_code in target_symbols:
            if ts_code in positions:
                continue
            price = prices.get(ts_code)
            if not price:
                continue
            shares = int(per_stock / price) // BOARD_LOT * BOARD_LOT
            if shares < BOARD_LOT:
                continue
            cost = shares * price * (1 + COMMISSION_RATE)
            if cost > cash[0]:
                continue
            cash[0] -= cost
            positions[ts_code] = {
                "shares": shares,
                "entry_price": price,
                "entry_date": str(day_data["trade_date"].iloc[-1]),
            }
            trades.append(
                Trade(
                    date=str(day_data["trade_date"].iloc[-1]),
                    symbol=ts_code,
                    action="buy",
                    price=price,
                    shares=shares,
                    amount=cost,
                    reason=f"买入 市价分配",
                )
            )

    def _calculate_sharpe(self, equity_curve: List[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        return float(drawdown.min() * 100)

    def _calculate_win_rate(self, trades: List[Trade]) -> float:
        sell_trades = [t for t in trades if t.action == "sell"]
        if not sell_trades:
            return 0.0
        wins = sum(1 for t in sell_trades if t.amount > 0)
        return wins / len(sell_trades) * 100

    def _save_run_info(self):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO backtest_runs (id, strategies, stock_pool, start_date, end_date,
                                       initial_capital, rebalance_days, max_positions, max_position_pct, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
        """,
            (
                self.run_id,
                ",".join(self.config.strategies),
                self.config.stock_pool,
                self.config.start_date,
                self.config.end_date,
                self.config.initial_capital,
                self.config.rebalance_days,
                self.config.max_positions,
                self.config.max_position_pct,
            ),
        )
        conn.commit()

    def _save_result(self, strategy_name: str, result: StrategyResult):
        conn = self._get_conn()

        # Save summary
        conn.execute(
            """
            INSERT INTO backtest_results
            (run_id, strategy_name, final_value, total_return, sharpe, max_drawdown, win_rate, total_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.run_id,
                strategy_name,
                result.final_value,
                result.total_return,
                result.sharpe,
                result.max_drawdown,
                result.win_rate,
                result.total_trades,
            ),
        )

        # Save trades
        for trade in result.trades:
            conn.execute(
                """
                INSERT INTO backtest_trades
                (run_id, strategy_name, trade_date, symbol, action, price, shares, amount, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.run_id,
                    strategy_name,
                    trade.date,
                    trade.symbol,
                    trade.action,
                    trade.price,
                    trade.shares,
                    trade.amount,
                    trade.reason,
                ),
            )

        # Save snapshots (only keep first and last to save space)
        if result.daily_snapshots:
            for snapshot in [result.daily_snapshots[0], result.daily_snapshots[-1]]:
                conn.execute(
                    """
                    INSERT INTO backtest_snapshots
                    (run_id, strategy_name, snapshot_date, cash, total_value, positions_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        self.run_id,
                        strategy_name,
                        snapshot.date,
                        snapshot.cash,
                        snapshot.total_value,
                        json.dumps(snapshot.positions),
                    ),
                )
        conn.commit()

    def _update_run_status(self, status: str):
        conn = self._get_conn()
        conn.execute(
            "UPDATE backtest_runs SET status=? WHERE id=?", (status, self.run_id)
        )
        conn.commit()


def get_backtest_result(run_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(str(BACKTEST_DB))
    try:
        run_info = conn.execute(
            "SELECT * FROM backtest_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not run_info:
            return None

        columns = [
            desc[0]
            for desc in conn.execute(
                "SELECT * FROM backtest_runs WHERE id=0"
            ).description
        ]
        run_dict = dict(zip(columns, run_info))

        results = []
        cursor = conn.execute(
            "SELECT * FROM backtest_results WHERE run_id=?", (run_id,)
        )
        rows = cursor.fetchall()
        result_cols = [desc[0] for desc in cursor.description]

        for row in rows:
            r = dict(zip(result_cols, row))
            trades_cursor = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id=? AND strategy_name=? ORDER BY trade_date",
                (run_id, r["strategy_name"]),
            )
            trades = trades_cursor.fetchall()
            trade_cols = [desc[0] for desc in trades_cursor.description]
            r["trades"] = [dict(zip(trade_cols, t)) for t in trades]
            results.append(r)

        return {
            "run_id": run_id,
            "config": run_dict,
            "strategies": results,
            "winner": max(results, key=lambda x: x["total_return"])["strategy_name"]
            if results
            else None,
        }
    finally:
        conn.close()


def get_delivery(run_id: str, strategy_name: str = None) -> List[Dict]:
    conn = sqlite3.connect(str(BACKTEST_DB))
    try:
        query = "SELECT * FROM backtest_trades WHERE run_id=?"
        params = [run_id]
        if strategy_name:
            query += " AND strategy_name=?"
            params.append(strategy_name)
        query += " ORDER BY trade_date, action DESC"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        conn.close()
