import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from features import compute_features

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def _build_model(window: int, n_features: int, horizon: int):
    inputs = keras.Input(shape=(window, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(horizon)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def make_supervised(series: np.ndarray, window: int, horizon: int):
    """
    Many-to-many: X windows → y vector of length horizon (future closes).
    series: shape (N, 1) scaled close values
    """
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window, :])
        y.append(series[i+window:i+window+horizon, 0])
    return np.array(X), np.array(y)

def train_lstm_model(ticker: str, horizon: int = 7, base_window: int = 60, epochs: int = 50, batch_size: int = 32):
    # Load Close only for LSTM
    raw = pd.read_csv(os.path.join(RAW_DIR, f"{ticker}.csv"), parse_dates=['Date'], index_col='Date')
    close = pd.to_numeric(raw['Close'], errors='coerce').dropna().to_frame('Close')
    if close.shape[0] < 30:
        print("[!] Too little data for LSTM, skipping.")
        return False

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close.values)
    n = len(scaled)

    window = min(base_window, max(10, n // 4))
    if n - window - horizon < 3:
        window = max(10, n - horizon - 2)
    if n - window - horizon < 1:
        print("[!] Still not enough rows to form sequences, skipping LSTM.")
        return False

    X, y = make_supervised(scaled.reshape(-1,1), window, horizon)
    print(f"[LSTM] Training samples: {X.shape[0]}, window={window}, features=1")

    model = _build_model(window, 1, horizon)

    # If too few samples, avoid validation split
    val_split = 0.1 if X.shape[0] >= 20 else 0.0
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=2)

    # Save model & meta (scaler + window + horizon)
    model_path = os.path.join(MODELS_DIR, f"{ticker}_lstm.keras")
    model.save(model_path)
    meta = {"window": int(window), "horizon": int(horizon)}
    joblib.dump({"scaler": scaler, "meta": meta}, os.path.join(MODELS_DIR, f"{ticker}_lstm.pkl"))
    print(f"[LSTM] Saved model to {model_path}")
    return True
