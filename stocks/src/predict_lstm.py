# src/predict_lstm.py

import os
import argparse
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from preprocess import parse_period_to_days, load_last_period
from features   import compute_features

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def forecast_lstm(ticker: str, period: str, horizon: int):
    # Load trained model + scalers
    model_path  = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    feat_path   = os.path.join(MODEL_DIR, f"{ticker}_lstm_feat_scaler.pkl")
    target_path = os.path.join(MODEL_DIR, f"{ticker}_lstm_target_scaler.pkl")

    model   = load_model(model_path, compile=False)
    Xscaler = joblib.load(feat_path)
    yscaler = joblib.load(target_path)

    # Determine window & feature dims
    _, window, n_features = model.input_shape

    # Load & feature-engineer
    days = parse_period_to_days(period)
    raw  = load_last_period(ticker, days)
    feat = compute_features(raw)

    # Prepare and scale features
    X_df = feat.drop(columns=['Close'])
    if len(X_df) < window:
        raise ValueError(f"Need {window} rows, but only {len(X_df)} available.")
    X_arr   = Xscaler.transform(X_df.values)          # (n_samples, n_features)
    X_last  = X_arr[-window:].reshape(1, window, n_features)

    # Predict and inverse-scale
    preds_scaled = model.predict(X_last)[0].flatten()     # (horizon,)
    preds = yscaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

    # Build next business-day dates
    bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    last_date = feat.index.max()
    dates = pd.date_range(start=last_date + bd, periods=horizon, freq=bd)

    # Save
    out = pd.DataFrame({
        'date':      dates.date,
        'ticker':    ticker,
        'lstm_pred': preds
    })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{ticker}_lstm_{horizon}d.csv")
    out.to_csv(path, index=False)

    print(f"[LSTM] Saved {horizon}-day forecast to {path}")
    print(out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Seq2Seq LSTM multi-day forecast"
    )
    parser.add_argument('--ticker',  required=True, help="Ticker (e.g. AAPL)")
    parser.add_argument('--period',  default='1y',  help="Look-back (e.g. 6mo, 1y)")
    parser.add_argument('--horizon', type=int, default=7,   help="Days to forecast")
    args = parser.parse_args()

    forecast_lstm(args.ticker.upper(), args.period, args.horizon)
