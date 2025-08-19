import os
import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')

def evaluate_model(ticker: str):
    eval_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ticker}_eval.csv"), index_col='Date', parse_dates=['Date'])
    if eval_df.empty:
        print(f"[!] No eval data for {ticker}")
        return None
    X_eval = eval_df.drop(columns=['Close']).apply(pd.to_numeric, errors='coerce').astype(float)
    y_eval = pd.to_numeric(eval_df['Close'], errors='coerce').astype(float)

    bundle_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(bundle_path):
        print(f"[!] Model not found: {bundle_path}")
        return None
    bundle = joblib.load(bundle_path)
    model = bundle["model"]; feature_names = bundle["feature_names"]; scaler = bundle["scaler"]

    # align features
    from utils import unify_features
    X_eval = unify_features(X_eval, feature_names)
    X_eval_scaled = pd.DataFrame(scaler.transform(X_eval), index=X_eval.index, columns=X_eval.columns)

    preds = model.predict(X_eval_scaled)
    mse = mean_squared_error(y_eval, preds)
    mae = mean_absolute_error(y_eval, preds)
    print(f"[✓] Eval results for {ticker} → MSE: {mse:.4f}, MAE: {mae:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "ticker": ticker,
        "mse": float(mse),
        "mae": float(mae),
        "n": int(len(y_eval))
    }
    with open(os.path.join(RESULTS_DIR, f"{ticker}_eval_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
