import os
from datetime import timedelta
import pandas as pd

from features import compute_features

RAW_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
os.makedirs(PROC_DIR, exist_ok=True)

def parse_period_to_days(period: str) -> int:
    p = period.lower().strip()
    if p.endswith('mo'):
        return int(p[:-2]) * 30
    if p.endswith('y'):
        return int(p[:-1]) * 365
    return int(p)

def load_last_period(ticker: str, period_days: int) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    # keep only needed columns
    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    df = df[keep]
    # numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()
    end = df.index.max()
    start = end - timedelta(days=period_days)
    return df.loc[start:end]

def split_train_eval_chrono(df: pd.DataFrame, train_frac: float = 0.8):
    """Chronological split: first 80% train, last 20% eval."""
    if df.empty:
        return df.copy(), df.copy()
    n = len(df)
    cut = max(1, int(n * train_frac))
    train = df.iloc[:cut].copy()
    eval_ = df.iloc[cut:].copy()
    return train, eval_

def process_ticker(ticker: str, period: str = '6mo'):
    period_days = parse_period_to_days(period)
    raw = load_last_period(ticker, period_days)
    feat = compute_features(raw)

    train_df, eval_df = split_train_eval_chrono(feat, train_frac=0.8)

    train_path = os.path.join(PROC_DIR, f"{ticker}_train.csv")
    eval_path  = os.path.join(PROC_DIR, f"{ticker}_eval.csv")
    train_df.to_csv(train_path)
    eval_df.to_csv(eval_path)

    print(f"[✓] {ticker}: train rows={len(train_df)} → {train_path}")
    print(f"[✓] {ticker}: eval rows ={len(eval_df)} → {eval_path}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Preprocess data and split 80/20")
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--period', default='6mo')
    args = ap.parse_args()
    process_ticker(args.ticker, args.period)
