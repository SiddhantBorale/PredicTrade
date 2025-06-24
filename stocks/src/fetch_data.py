import os
import argparse
import yfinance as yf

# Paths relative to this file's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR  = os.path.join(BASE_DIR, 'data', 'raw')


def get_history(ticker: str, period: str = "6mo") -> "pd.DataFrame":
    """
    Fetches historical OHLC data for `ticker` over `period` from yfinance,
    saves to data/raw/{ticker}.csv, and returns the DataFrame.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    df = yf.download(ticker, period=period, interval="1d")
    csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(csv_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch historical stock data and save to data/raw/"
    )
    parser.add_argument(
        '--ticker', required=True,
        help="Ticker symbol to fetch (e.g., AAPL)"
    )
    parser.add_argument(
        '--period', default="6mo",
        help="Data period to download (e.g., 6mo, 1y, 7d)"
    )
    args = parser.parse_args()

    df = get_history(args.ticker, args.period)
    print(f"Downloaded {len(df)} rows for {args.ticker}; saved to data/raw/{args.ticker}.csv")
