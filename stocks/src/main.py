import os
import argparse

from fetch_data import fetch_and_save
from preprocess import process_ticker
from train import train_and_save, XGB_AVAILABLE, _XGB_IMPORT_ERROR
from evaluate import evaluate_model
from predict_xgb import forecast_xgb
from sarimax_forecast import forecast_sarimax
from predict_lstm import forecast_lstm
from ensemble import fit_and_predict_ensemble
from utils import ensure_dirs

def main():
    parser = argparse.ArgumentParser(description="End-to-end stock pipeline")
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--period', default='6mo')
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--use_lstm', action='store_true')
    parser.add_argument('--skip_xgb', action='store_true', help="Skip XGB train/predict stage")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    horizon = args.horizon

    print(f"\n--- Stock Pipeline for {ticker} (period={args.period}, horizon={horizon}d) ---\n")
    ensure_dirs()

    # 1) Fetch
    print("[1/7] Fetching raw data...")
    fetch_and_save(ticker, args.period)

    # 2) Preprocess
    print("[2/7] Preprocessing data...")
    process_ticker(ticker, args.period)

    # 3) Train XGB
    if args.skip_xgb:
        print("[3/7] Skipping XGB training by flag.")
        xgb_ok = False
    else:
        print("[3/7] Training XGBoost model...")
        if not XGB_AVAILABLE:
            print(f"[!] XGBoost not available; skipping. Reason: {_XGB_IMPORT_ERROR}")
            xgb_ok = False
        else:
            xgb_ok = train_and_save(ticker)

    # 4) Evaluate XGB
    print("[4/7] Evaluating XGBoost model...")
    if xgb_ok:
        evaluate_model(ticker)
    else:
        print("[ ] Skipped evaluation (no XGB model).")

    # 5) Forecast SARIMAX
    print("[5/7] Generating SARIMAX forecast...")
    forecast_sarimax(ticker, args.period, horizon)

    # 6) Forecast XGB (recursive)
    print("[6/7] Generating XGB forecast...")
    if xgb_ok:
        forecast_xgb(ticker, args.period, horizon)
    else:
        print("[ ] Skipped XGB forecast (no model).")

    # 7) LSTM (optional) + Ensemble
    if args.use_lstm:
        print("[7/7] Training + forecasting LSTM...")
        ok = True
        try:
            from train_lstm import train_lstm_model
            ok = train_lstm_model(ticker, horizon=horizon)
        except Exception as e:
            print(f"[!] LSTM training failed: {e}")
            ok = False
        if ok:
            forecast_lstm(ticker, horizon=horizon)

    # Ensemble (average available models)
    print("[-->] Creating stacked ensemble forecast...")
    fit_and_predict_ensemble(ticker, horizon)

if __name__ == "__main__":
    main()
