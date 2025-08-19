import os
from datetime import datetime
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_and_save(ticker: str, period: str) -> str:
    """
    Fetch daily OHLCV via yfinance and save to data/raw/{ticker}.csv.
    yfinance default auto_adjust=True now; we rely on ['Open','High','Low','Close','Volume'].
    """
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} period={period}")
    df.index.name = 'Date'
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"[✓] Saved raw {ticker} → {path} ({len(df)} rows)")
    return path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", default="6mo")
    args = ap.parse_args()
    fetch_and_save(args.ticker, args.period)
