import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

# safe import for macOS users without libomp
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    _XGB_IMPORT_ERROR = None
except Exception as e:
    XGB_AVAILABLE = False
    _XGB_IMPORT_ERROR = e

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_train_eval(ticker: str):
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ticker}_train.csv"), index_col='Date', parse_dates=['Date'])
    eval_df  = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ticker}_eval.csv"),  index_col='Date', parse_dates=['Date'])
    X_train = train_df.drop(columns=['Close'])
    y_train = train_df['Close']
    X_eval  = eval_df.drop(columns=['Close'])
    y_eval  = eval_df['Close']
    # enforce numeric
    X_train = X_train.apply(pd.to_numeric, errors='coerce').astype(float)
    X_eval  = X_eval.apply(pd.to_numeric, errors='coerce').astype(float)
    y_train = pd.to_numeric(y_train, errors='coerce').astype(float)
    y_eval  = pd.to_numeric(y_eval, errors='coerce').astype(float)
    X_train, y_train = X_train.dropna(), y_train.loc[X_train.index]
    X_eval,  y_eval  = X_eval.dropna(),  y_eval.loc[X_eval.index]
    return X_train, y_train, X_eval, y_eval

def remove_outliers_robust(X: pd.DataFrame, y: pd.Series, z=4.0):
    """
    Winsorize targets via robust z-score and mask extreme target outliers.
    """
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + 1e-9
    rz = 0.6745 * (y - med) / mad
    mask = np.abs(rz) < z
    removed = int((~mask).sum())
    if removed:
        print(f"[ ] Removed {removed} outlier rows from training data")
    return X.loc[mask], y.loc[mask]

def recency_weights(index: pd.Index, recent_window: int = 7, base_weight: float = 1.0, recent_weight: float = 3.0):
    """
    Give more weight to last 'recent_window' observations.
    """
    w = np.full(shape=(len(index),), fill_value=base_weight, dtype=float)
    if len(index) == 0:
        return w
    # last N rows get higher weight
    w[-recent_window:] = recent_weight
    return w

def train_and_save(ticker: str):
    if not XGB_AVAILABLE:
        print(f"[!] XGBoost unavailable; skipping XGB training. Reason: {_XGB_IMPORT_ERROR}")
        return False

    X_train, y_train, X_eval, y_eval = load_train_eval(ticker)
    if X_train.empty:
        print(f"[!] No training data for {ticker}")
        return False

    # Outlier removal on target
    X_train, y_train = remove_outliers_robust(X_train, y_train, z=4.0)

    # (Optional) robust scale features
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_eval_scaled  = pd.DataFrame(scaler.transform(X_eval), index=X_eval.index, columns=X_eval.columns)

    # Recency weights
    sample_weight = recency_weights(X_train_scaled.index, recent_window=min(7, len(X_train_scaled)))

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=4,
        random_state=42,
        verbosity=1,
        tree_method="hist"
    )
    print(f"[ ] Training XGBRegressor on {len(X_train_scaled)} samples with recency weighting...")
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

    if not X_eval_scaled.empty:
        preds = model.predict(X_eval_scaled)
        mse = mean_squared_error(y_eval, preds)
        print(f"[✓] Eval MSE for {ticker}: {mse:.4f}")

    # Save model bundle with feature names + scaler
    bundle = {
        "model": model,
        "feature_names": list(X_train.columns),
        "scaler": scaler
    }
    out_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    joblib.dump(bundle, out_path)
    print(f"[✓] Saved model to {out_path}")
    return True
