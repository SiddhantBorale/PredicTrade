#!/usr/bin/env python3
import os
import argparse

from fetch_data import get_history
from preprocess import process_ticker
from train import train_and_save
from evaluate import evaluate_and_save
from predict import forecast_sarimax
from predict_xgb import forecast_xgb
from ensemble import fit_and_predict_ensemble
from train_lstm import train_lstm_model
from predict_lstm import forecast_lstm

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end stock forecasting pipeline with stacking and optional LSTM"
    )
    parser.add_argument('--ticker',      required=True,  help="Ticker symbol (e.g. AAPL)")
    parser.add_argument('--period',      default='6mo',   help="Look-back window (e.g. 6mo, 1y)")
    parser.add_argument('--train_ratio', type=float, default=0.8, help="Train/eval split ratio")
    parser.add_argument('--horizon',     type=int,   default=7,    help="Days to forecast")
    parser.add_argument('--lstm_window', type=int,   default=60,   help="LSTM input window size")
    parser.add_argument('--lstm_epochs', type=int,   default=50,   help="LSTM training epochs")
    parser.add_argument('--lstm_batch',  type=int,   default=32,   help="LSTM training batch size")
    parser.add_argument('--use_lstm',    action='store_true',   help="Train & forecast with Seq2Seq LSTM")
    args = parser.parse_args()

    ticker      = args.ticker.upper()
    period      = args.period
    train_ratio = args.train_ratio
    horizon     = args.horizon

    BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    print(f"\n--- Stock Pipeline for {ticker} (period={period}, horizon={horizon}d) ---\n")

    # 1) Fetch raw data
    print("[1/8] Fetching raw data...")
    get_history(ticker, period)

    # 2) Preprocess
    print("[2/8] Preprocessing data...")
    process_ticker(ticker, period, train_ratio)

    # 3) Train XGBoost
    print("[3/8] Training XGBoost model...")
    train_and_save(ticker)

    # 4) Evaluate XGBoost
    print("[4/8] Evaluating XGBoost model...")
    evaluate_and_save(ticker)

    # 5) SARIMAX direct forecast
    print("[5/8] Generating SARIMAX forecast...")
    forecast_sarimax(ticker, period, horizon)

    # 6) Recursive XGBoost forecast (for stacking)
    print("[6/8] Generating recursive XGBoost forecast...")
    _ = forecast_xgb(ticker, period, horizon)

    # 7) Stacked ensemble of XGB + SARIMAX
    print("[7/8] Creating stacked ensemble forecast...")
    fit_and_predict_ensemble(ticker, period, horizon)

    # 8) Optional Seq2Seq LSTM
    if args.use_lstm:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
        # Only train if no saved model exists
        if not os.path.exists(model_path):
            print("[8/8] LSTM model not found. Training Seq2Seq LSTM model...")
            train_lstm_model(
                ticker,
                period,
                window=args.lstm_window,
                horizon=horizon,
                epochs=args.lstm_epochs,
                batch_size=args.lstm_batch
            )
        else:
            print("[8/8] Found existing LSTM model; skipping training.")
        print("[-->] Generating LSTM forecast...")
        forecast_lstm(ticker, period, horizon)

if __name__ == '__main__':
    main()
