import os
import argparse

from fetch_data import get_history
from preprocess   import process_ticker
from train        import train_and_save
from evaluate     import evaluate_and_save
from predict      import forecast_sarimax

def main():
    parser = argparse.ArgumentParser(description="Stock pipeline with SARIMAX 7-day forecast")
    parser.add_argument('--ticker',    required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument('--period',    default='6mo',  help="Look-back window (e.g. 6mo, 1y)")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Train/eval split ratio")
    parser.add_argument('--horizon',   type=int, default=7,    help="Days to forecast")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    period = args.period
    horizon = args.horizon

    print(f"\n--- Stock Pipeline for {ticker} (period={period}) ---\n")
    print("[1/5] Fetching raw data...")
    get_history(ticker, period)

    print("[2/5] Preprocessing data...")
    process_ticker(ticker, period, train_ratio=args.train_ratio)

    print("[3/5] Training model...")
    train_and_save(ticker)

    print("[4/5] Evaluating model...")
    evaluate_and_save(ticker)

    print("[5/5] Generating SARIMAX forecast...")
    forecast_sarimax(ticker, period, horizon)

if __name__ == '__main__':
    main()
