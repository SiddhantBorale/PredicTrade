import os
import joblib
import numpy as np
import pandas as pd

from features import compute_features, feature_columns
from utils import RESULTS_DIR, safe_ticker, unify_features, next_trading_days

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))

def forecast_xgb(ticker: str, period: str, horizon: int = 7) -> pd.DataFrame:
    """
    Recompute features on the latest raw window, then recursively predict Close for next N trading days.
    Saves results/{SAFE_TICKER}_xgb_{horizon}d.csv and returns the DataFrame.
    """
    raw_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df_raw = pd.read_csv(raw_path, parse_dates=['Date'], index_col='Date')
    df_raw = df_raw[['Close']].copy()
    feat = compute_features(df_raw)

    # Load model bundle
    bundle = joblib.load(os.path.join(os.path.dirname(RESULTS_DIR), 'models', f"{ticker}_model.pkl"))
    model = bundle['model']; feature_names = bundle['feature_names']; scaler = bundle['scaler']

    last_date = feat.index[-1]
    pred_dates = next_trading_days(last_date, horizon)

    preds = []
    working = feat.copy()
    for d in pred_dates:
        X = unify_features(working.drop(columns=['Close']), feature_names)
        X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        x_today = X_scaled.iloc[[-1]]
        y_hat = float(model.predict(x_today)[0])
        preds.append((d, y_hat))

        # append synthetic row with predicted close to continue features
        new_row = pd.DataFrame({'Close': [y_hat]}, index=[d])
        working = pd.concat([working[['Close']], new_row]).sort_index()
        working = compute_features(working)  # recompute indicators with appended close

    out = pd.DataFrame(preds, columns=['date','forecast_close'])
    out['ticker'] = ticker
    safe = safe_ticker(ticker)
    out_path = os.path.join(RESULTS_DIR, f"{safe}_xgb_{horizon}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[✓] XGB {horizon}-day forecast saved to {out_path}")
    return out
