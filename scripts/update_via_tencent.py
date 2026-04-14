import os
import sys
import sqlite3
import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import config

for k in list(os.environ.keys()):
    if any(x in k.lower() for x in ['proxy', 'http', 'https']):
        del os.environ[k]

TENANT_KLINE_URL = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayhfq&param={code},day,{start},{end},100,qfq"


def normalize_tencent_code(ts_code):
    if ts_code.endswith('.SH'):
        return 'sh' + ts_code[:6]
    elif ts_code.endswith('.SZ'):
        return 'sz' + ts_code[:6]
    return ts_code


def fetch_tencent_history(ts_code, start_date, end_date):
    code = normalize_tencent_code(ts_code)
    url = TENANT_KLINE_URL.format(code=code, start=start_date, end=end_date)
    
    try:
        r = requests.get(url, timeout=10)
        text = r.text
        if not text.startswith('kline_dayhfq='):
            return pd.DataFrame()
        
        json_str = text[len('kline_dayhfq='):]
        data = json.loads(json_str)
        
        stock_data = data.get('data', {}).get(code, {})
        if not stock_data:
            return pd.DataFrame()
        
        qfq_key = list(stock_data.keys())[0]
        if qfq_key != 'qfqday':
            return pd.DataFrame()
        
        rows = stock_data['qfqday']
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=['trade_date', 'open', 'close', 'high', 'low', 'vol'])
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        for col in ['open', 'close', 'high', 'low', 'vol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['amount'] = 0.0
        df['pct_chg'] = 0.0
        return df
    except Exception as e:
        return pd.DataFrame()


def update_stocks():
    conn = sqlite3.connect(str(config.DB_PATH))
    
    stocks = [r[0] for r in conn.execute("SELECT ts_code FROM stock_list ORDER BY ts_code").fetchall()]
    print(f"Total stocks: {len(stocks)}")
    
    updated = 0
    failed = []
    
    for i, ts_code in enumerate(stocks):
        last_row = conn.execute(
            "SELECT MAX(trade_date) FROM daily_quotes WHERE ts_code=?", (ts_code,)
        ).fetchone()
        last_date = last_row[0] if last_row and last_row[0] else "2020-01-01"
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        if last_date >= today:
            continue
        
        start = (datetime.strptime(last_date, '%Y-%m-%d') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if start >= today:
            continue
        
        df = fetch_tencent_history(ts_code, start, today)
        
        if df.empty:
            failed.append(ts_code)
            print(f"[{i+1}/{len(stocks)}] {ts_code}: no data")
            continue
        
        try:
            df.to_sql('daily_quotes', conn, if_exists='append', index=False)
            print(f"[{i+1}/{len(stocks)}] {ts_code}: +{len(df)} rows ({df['trade_date'].iloc[0]}~{df['trade_date'].iloc[-1]})")
            updated += 1
        except Exception as e:
            failed.append(ts_code)
            print(f"[{i+1}/{len(stocks)}] {ts_code}: insert error")
        
        time.sleep(0.15)
    
    conn.close()
    print(f"\nDone: {updated} updated, {len(failed)} failed")
    
    conn = sqlite3.connect(str(config.DB_PATH))
    r = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_quotes").fetchone()
    print(f"Data range: {r[0]} ~ {r[1]}")
    conn.close()


if __name__ == "__main__":
    print(f"Updating via Tencent API... {datetime.now()}")
    update_stocks()
