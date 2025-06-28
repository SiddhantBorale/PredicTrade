# src/predict.py

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.neighbors import NearestNeighbors

from preprocess import parse_period_to_days, load_last_period
from features import compute_features

# Paths
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def adjust_by_pattern(df_history, base_preds, horizon, window=7, k=5):
    """
    Given df_history with 'Close', and base_preds list of floats,
    find the last `window` days' return pattern, locate the k most similar
    patterns in history, average their next-horizon returns, and apply
    that adjustment to base_preds.
    """
    closes = df_history['Close']
    # 1) current pattern: pct_change over last `window` days
    recent = closes.iloc[-(window+1):]
    curr_pattern = recent.pct_change().dropna().values  # shape (window,)

    # 2) build historical patterns + their future returns
    hist_pats = []
    hist_futures = []
    total_len = len(closes)
    for start in range(0, total_len - window - horizon):
        seg = closes.iloc[start : start + window + 1]
        pat = seg.pct_change().dropna().values
        if len(pat) != window:
            continue
        # compute next `horizon` returns
        fut = []
        base_price = seg.iloc[-1]
        for h in range(1, horizon + 1):
            fut_price = closes.iloc[start + window + h]
            fut.append((fut_price / base_price) - 1.0)
        hist_pats.append(pat)
        hist_futures.append(fut)

    if not hist_pats:
        return base_preds  # no history to match

    # 3) find k-nearest
    nbrs = NearestNeighbors(n_neighbors=min(k, len(hist_pats)), metric='euclidean')
    nbrs.fit(hist_pats)
    dists, idxs = nbrs.kneighbors([curr_pattern])

    # 4) average the future returns across neighbors
    neigh = np.array([ hist_futures[i] for i in idxs[0] ])  # shape (k, horizon)
    avg_returns = np.nanmean(neigh, axis=0)  # shape (horizon,)

    # 5) apply multiplicative adjustment: pred * (1 + avg_return)
    return [ base * (1 + ret) for base, ret in zip(base_preds, avg_returns) ]


def predict_n_days(ticker: str, period: str, horizon: int, adjust: bool = True):
    # 1) Load raw history (with Open/High/Low/Volume/Close)
    period_days = parse_period_to_days(period)
    df_history = load_last_period(ticker, period_days).copy()

    # 2) Load trained model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    model = joblib.load(model_path)

    # 3) Generate base multi-day forecasts
    base_preds = []
    for _ in range(horizon):
        feat = compute_features(df_history)
        X_last = feat.drop(columns=['Close']).iloc[[-1]]
        pred = model.predict(X_last)[0]
        base_preds.append(pred)

        # append new row for next iteration
        last_raw = df_history.iloc[-1][['Open','High','Low','Volume']].to_dict()
        last_raw['Close'] = pred
        next_date = feat.index[-1] + timedelta(days=1)
        df_history.loc[next_date] = last_raw

    # 4) Optionally adjust by pattern
    final_preds = (adjust
                   and adjust_by_pattern(df_history, base_preds, horizon)
                   or base_preds)

    # 5) Save both sets of forecasts
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = pd.DataFrame({
        'date': [(pd.Timestamp(df_history.index[-1]) + timedelta(days=i+1)).date() 
                 for i in range(horizon)],
        'ticker': ticker,
        'base_close': base_preds,
        'adj_close': final_preds
    })
    out_path = os.path.join(RESULTS_DIR, f"{ticker}_predictions_{horizon}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[✓] Saved {horizon}-day base & adjusted forecasts to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker',  required=True)
    parser.add_argument('--period',  default='6mo')
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--no-adjust', action='store_true',
                        help="Skip trend-based adjustment")
    args = parser.parse_args()
    predict_n_days(args.ticker, args.period, args.horizon,
                   adjust=not args.no_adjust)
