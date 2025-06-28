# src/predict_xgb.py

import os
import joblib
import pandas as pd
from datetime import timedelta

from preprocess import parse_period_to_days, load_last_period
from features import compute_features

# Paths
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

def forecast_xgb(ticker: str, period: str, horizon: int) -> pd.DataFrame:
    """
    Generate a recursive XGBoost forecast for the next `horizon` business days.
    Returns a DataFrame with columns ['date','xgb_pred'].
    """
    # 1) Load the last N days of raw OHLCV+Close
    days = parse_period_to_days(period)
    df = load_last_period(ticker, days).copy()

    # 2) Load the trained XGB model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    model = joblib.load(model_path)

    # 3) Roll forward, predicting one day at a time
    rows = []
    for _ in range(horizon):
        feat = compute_features(df)
        X = feat.drop(columns=['Close']).iloc[[-1]]
        pred = model.predict(X)[0]
        next_date = feat.index[-1] + timedelta(days=1)
        rows.append({'date': next_date.date(), 'xgb_pred': pred})

        # append back into df so next loop has full OHLCV+Close
        last_raw = df.iloc[-1][['Open','High','Low','Volume']].to_dict()
        last_raw['Close'] = pred
        df.loc[next_date] = last_raw

    return pd.DataFrame(rows)
