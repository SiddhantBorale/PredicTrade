import os
import pandas as pd
from datetime import datetime, timedelta
from features import compute_features

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def load_last_six_months(ticker: str) -> pd.DataFrame:
    """
    Loads the CSV for `ticker` from data/raw/, parses dates,
    and returns only the last ~6 months of data.
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    end = df.index.max()
    start = end - timedelta(days=6 * 30)
    return df.loc[start:end]

def split_train_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits into:
      - train = everything before the 1st of the most recent month
      - eval  = everything from the 1st of the most recent month onward
    """
    last_date = df.index.max()
    start_current_month = last_date.replace(day=1)
    train_df = df[df.index < start_current_month]
    eval_df = df[df.index >= start_current_month]
    return train_df, eval_df

def process_ticker(ticker: str):
    # 1) Load raw
    raw = load_last_six_months(ticker)

    # 2) Feature engineering
    feat = compute_features(raw)

    # 3) Split
    train_df, eval_df = split_train_eval(feat)

    # 4) Save
    os.makedirs(PROC_DIR, exist_ok=True)
    train_path = os.path.join(PROC_DIR, f"{ticker}_train.csv")
    eval_path  = os.path.join(PROC_DIR, f"{ticker}_eval.csv")

    train_df.to_csv(train_path)
    eval_df.to_csv(eval_path)

    print(f"[✓] {ticker}: train={len(train_df)} rows → {train_path}")
    print(f"[✓] {ticker}: eval ={len(eval_df)} rows → {eval_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess rolling 6-month window for a ticker")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    args = parser.parse_args()

    process_ticker(args.ticker)
