import os
import pandas as pd
from datetime import timedelta
from features import compute_features

# Directories for raw and processed data
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))


def parse_period_to_days(period: str) -> int:
    """
    Convert a period string like '6mo', '12mo', '1y' into days.
    Assumes 1 month = 30 days, 1 year = 365 days.
    """
    p = period.lower().strip()
    if p.endswith('mo'):
        months = int(p[:-2])
        return months * 30
    if p.endswith('y'):
        years = int(p[:-1])
        return years * 365
    return int(p)


def load_last_period(ticker: str, period_days: int) -> pd.DataFrame:
    """
    Load raw CSV for ticker, parse date, rename Price/Adj Close to Close,
    drop outliers, and return last `period_days` of data.
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    # Attempt to skip metadata rows
    try:
        df = pd.read_csv(path, skiprows=[1,2], parse_dates=['Price'], index_col='Price')
        df.index.rename('Date', inplace=True)
    except Exception:
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')

    # Rename to 'Close'
    if 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})
    elif 'Adj Close' in df.columns and 'Close' not in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})

    if 'Close' not in df.columns:
        raise KeyError(f"Expected 'Price', 'Adj Close' or 'Close' in {path}")


    # Limit to last period_days
    end_date = df.index.max()
    start_date = end_date - timedelta(days=period_days)
    return df.loc[start_date:end_date]


def split_train_eval(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into training and evaluation sets by ratio:
      - train: first train_ratio of data
      - eval: remaining data
    """
    df = df.sort_index()
    n = len(df)
    split_idx = int(n * train_ratio)
    train_df = df.iloc[:split_idx]
    eval_df = df.iloc[split_idx:]
    return train_df, eval_df


def process_ticker(ticker: str, period: str = '6mo', train_ratio: float = 0.8) -> None:
    """
    Full preprocessing for a ticker with ratio-based split:
      1) Load raw data last period
      2) Compute features
      3) Split into train/eval by ratio
      4) Save processed CSVs
    """
    period_days = parse_period_to_days(period)
    raw_df = load_last_period(ticker, period_days)
    feat_df = compute_features(raw_df)
    train_df, eval_df = split_train_eval(feat_df, train_ratio)

    os.makedirs(PROC_DIR, exist_ok=True)
    train_path = os.path.join(PROC_DIR, f"{ticker}_train.csv")
    eval_path = os.path.join(PROC_DIR, f"{ticker}_eval.csv")
    train_df.to_csv(train_path)
    eval_df.to_csv(eval_path)

    print(f"[✓] {ticker}: train rows={len(train_df)} → {train_path}")
    print(f"[✓] {ticker}: eval rows ={len(eval_df)} → {eval_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess data with ratio-based train/test split"
    )
    parser.add_argument('--ticker', required=True, help="Ticker (e.g., AAPL)")
    parser.add_argument('--period', default='6mo', help="Data period (e.g., 6mo, 1y)")
    args = parser.parse_args()
    process_ticker(args.ticker, args.period, args.train_ratio)
