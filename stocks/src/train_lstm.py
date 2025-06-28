# src/train_lstm.py

import os
import argparse
import numpy as np
import joblib

from preprocess import parse_period_to_days, load_last_period
from features import compute_features

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def build_seq2seq(window: int, n_features: int, horizon: int) -> Model:
    inp = Input(shape=(window, n_features))
    _, state_h, state_c = LSTM(64, return_state=True)(inp)
    dec = RepeatVector(horizon)(state_h)
    dec = LSTM(64, return_sequences=True)(dec, initial_state=[state_h, state_c])
    out = TimeDistributed(Dense(1))(dec)
    return Model(inp, out)


def train_lstm_model(
    ticker: str,
    period: str,
    window: int,
    horizon: int,
    epochs: int,
    batch_size: int
):
    # 1) Load & feature-engineer
    days = parse_period_to_days(period)
    raw = load_last_period(ticker, days)
    feat = compute_features(raw)

    # 2) Split features vs. target
    X_df = feat.drop(columns=['Close'])
    y_df = feat[['Close']]

    # 3) Scale both
    Xscaler = StandardScaler().fit(X_df.values)
    yscaler = StandardScaler().fit(y_df.values)

    X_arr = Xscaler.transform(X_df.values)        # (n_samples, n_features)
    y_arr = yscaler.transform(y_df.values).flatten()  # (n_samples,)

    n = len(X_arr)
    if n < window + horizon:
        available = n - horizon
        if available < 2:
            print(f"[LSTM] Not enough data ({n} rows) for window+ horizon; skipping LSTM training.")
            return
        print(f"[LSTM] Only {n} rows; reducing window from {window} to {available}.")
        window = available

    # 4) Build sliding-window arrays
    X, y = [], []
    for i in range(n - window - horizon + 1):
        X.append(X_arr[i : i + window])
        y.append(y_arr[i + window : i + window + horizon])
    X = np.array(X)                          # (samples, window, features)
    y = np.array(y).reshape(-1, horizon, 1) # (samples, horizon, 1)

    num_samples = X.shape[0]
    print(f"[LSTM] Training samples: {num_samples}, window={window}, features={X.shape[2]}")

    if num_samples < 5:
        print(f"[LSTM] Only {num_samples} samples; skipping LSTM training.")
        return

    # 5) Build & train the model
    model = build_seq2seq(window, X.shape[2], horizon)
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              callbacks=[es],
              verbose=2)

    # 6) Save model + scalers
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path   = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    feat_path    = os.path.join(MODEL_DIR, f"{ticker}_lstm_feat_scaler.pkl")
    target_path  = os.path.join(MODEL_DIR, f"{ticker}_lstm_target_scaler.pkl")

    model.save(model_path)
    joblib.dump(Xscaler, feat_path)
    joblib.dump(yscaler, target_path)

    print(f"[LSTM] Saved model to {model_path}")
    print(f"[LSTM] Saved feature-scaler to {feat_path}")
    print(f"[LSTM] Saved target-scaler to {target_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Seq2Seq LSTM for multi-day stock forecasting"
    )
    parser.add_argument('--ticker',     required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument('--period',     default='1y',  help="Look-back window (e.g. 6mo, 1y)")
    parser.add_argument('--window',     type=int, default=60, help="Input window size")
    parser.add_argument('--horizon',    type=int, default=7,  help="Days to forecast")
    parser.add_argument('--epochs',     type=int, default=50, help="Max training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    train_lstm_model(
        ticker=args.ticker.upper(),
        period=args.period,
        window=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
