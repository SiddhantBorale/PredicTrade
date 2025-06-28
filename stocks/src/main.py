#!/usr/bin/env python3
"""
Orchestrator script to run the entire stock prediction pipeline:
1. Fetch raw data
2. Preprocess (feature engineering + train/eval split)
3. Train model
4. Evaluate model
5. Generate and log next-day prediction
"""
import argparse
import sys
import os

# Add src directory to Python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import pipeline stages
from fetch_data import get_history
from preprocess import process_ticker
from train import train_and_save
from evaluate import evaluate_and_save
from predict import predict_n_days


def main():
    parser = argparse.ArgumentParser(
        description="Run full stock prediction pipeline end-to-end"
    )
    parser.add_argument(
        '--ticker', required=True,
        help="Ticker symbol to process (e.g., AAPL)"
    )
    parser.add_argument(
        '--period', default='6mo',
        help="History period for fetch_data (e.g., 6mo, 1y)"
    )
    parser.add_argument('--horizon', type=int, default=7, 
        help="Number of days to predict")


    args = parser.parse_args()

    ticker = args.ticker.upper()
    period = args.period
    horizon = args.horizon

    print(f"\n--- Stock Pipeline for {ticker} (period={period}) ---\n")
    # 1. Fetch raw data
    print("[1/5] Fetching raw data...")
    get_history(ticker, period)

    # 2. Preprocess (features + split)
    print("[2/5] Preprocessing data...")
    process_ticker(ticker, period)

    # 3. Train
    print("[3/5] Training model...")
    train_and_save(ticker)

    # 4. Evaluate
    print("[4/5] Evaluating model...")
    evaluate_and_save(ticker)

    # 5. Predict
    print("[5/5] Generating prediction...")
    predict_n_days(ticker, period, horizon)

    print("\nPipeline complete! Check data/processed/, models/, results/ for outputs.")


if __name__ == '__main__':
    main()
