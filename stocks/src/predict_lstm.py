import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from utils import RESULTS_DIR, safe_ticker, next_trading_days

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR  = os.path.join(BASE_DIR, 'data', 'raw')

def forecast_lstm(ticker: str, horizon: int = 7) -> pd.DataFrame:
    model_path = os.path.join(BASE_DIR, 'models', f"{ticker}_lstm.keras")
    meta_path  = os.path.join(BASE_DIR, 'models', f"{ticker}_lstm.pkl")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        print("[!] LSTM model or meta not found; skip.")
        return None

    model = load_model(model_path)
    data = joblib.load(meta_path)
    scaler = data['scaler']; window = int(data['meta']['window']); horizon = int(data['meta']['horizon'])

    raw = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"), parse_dates=['Date'], index_col='Date')
    close = pd.to_numeric(raw['Close'], errors='coerce').dropna().to_frame('Close')
    scaled = scaler.transform(close.values)

    if len(scaled) < window:
        # pad from the start
        pad = np.tile(scaled[0], (window - len(scaled), 1))
        window_input = np.vstack([pad, scaled])
    else:
        window_input = scaled[-window:]

    X = window_input.reshape(1, window, 1)
    y_hat_scaled = model.predict(X, verbose=0)[0]  # shape (horizon,)
    y_hat = scaler.inverse_transform(y_hat_scaled.reshape(-1,1)).flatten()

    last_date = close.index.max()
    dates = next_trading_days(last_date, horizon)
    out = pd.DataFrame({'date': dates, 'ticker': ticker, 'lstm_pred': y_hat})
    out.rename(columns={'lstm_pred':'forecast_close'}, inplace=True)

    safe = safe_ticker(ticker)
    out_path = os.path.join(RESULTS_DIR, f"{safe}_lstm_{horizon}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[✓] LSTM {horizon}-day forecast saved to {out_path}")
    return out
