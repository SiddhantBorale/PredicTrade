import os
import pandas as pd
from datetime import timedelta
from features import compute_features

# Directories for raw and processed data
RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
PROC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))


def load_last_six_months(ticker: str) -> pd.DataFrame:
    """
    Load the CSV for `ticker` from data/raw/, skip metadata rows,
    parse the 'Price' column as dates, set it as the Date index,
    ensure a 'Close' column exists, and return the last ~6 months.
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    # Skip the two header rows (Ticker info and blank row)
    df = pd.read_csv(path, skiprows=[1, 2], parse_dates=['Price'], index_col='Price')
    df.index.rename('Date', inplace=True)

    # Rename 'Price' column if present as a data column
    if 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Close'})

    # Ensure Close column exists for feature computation
    if 'Close' not in df.columns:
        raise KeyError(f"Expected 'Close' column in {path} after parsing.")

    # Select the last ~6 months of data
    end_date = df.index.max()
    start_date = end_date - timedelta(days=6 * 30)
    return df.loc[start_date:end_date]


def split_train_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into training (before this month) and evaluation (current month onward).
    """
    last_date = df.index.max()
    month_start = last_date.replace(day=1)
    train_df = df[df.index < month_start]
    eval_df = df[df.index >= month_start]
    return train_df, eval_df


def process_ticker(ticker: str):
    """
    End-to-end preprocessing for a ticker:
      1) Load last 6 months raw data
      2) Compute features
      3) Split into train/eval
      4) Save to CSV
    """
    raw_df = load_last_six_months(ticker)
    feat_df = compute_features(raw_df)
    train_df, eval_df = split_train_eval(feat_df)

    os.makedirs(PROC_DIR, exist_ok=True)
    train_path = os.path.join(PROC_DIR, f"{ticker}_train.csv")
    eval_path = os.path.join(PROC_DIR, f"{ticker}_eval.csv")
    train_df.to_csv(train_path)
    eval_df.to_csv(eval_path)

    print(f"[✓] {ticker}: train rows={len(train_df)} → {train_path}")
    print(f"[✓] {ticker}: eval rows ={len(eval_df)} → {eval_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess rolling-window data for a ticker")
    parser.add_argument('--ticker', required=True, help="Ticker symbol (e.g., AAPL)")
    args = parser.parse_args()
    process_ticker(args.ticker)
