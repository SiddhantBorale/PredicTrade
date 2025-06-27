import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import argparse

# Paths relative to this file's directory
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR      = os.path.join(BASE_DIR, 'models')


def load_train_eval(ticker: str):
    """
    Load processed train/eval CSVs for the given ticker.
    Returns:
        X_train (DataFrame), y_train (Series), X_eval (DataFrame), y_eval (Series)
    """
    train_path = os.path.join(PROCESSED_DIR, f"{ticker}_train.csv")
    eval_path  = os.path.join(PROCESSED_DIR, f"{ticker}_eval.csv")

    train_df = pd.read_csv(train_path, index_col='Date', parse_dates=['Date'])
    eval_df  = pd.read_csv(eval_path,  index_col='Date', parse_dates=['Date'])

    # Convert all columns to numeric to ensure correct dtypes
    train_df = train_df.apply(pd.to_numeric, errors='coerce')
    eval_df  = eval_df.apply(pd.to_numeric,  errors='coerce')

    # Drop any rows with NaNs introduced by coercion
    train_df = train_df.dropna()
    eval_df  = eval_df.dropna()

    # Separate features and target
    X_train = train_df.drop(columns=['Close'])
    y_train = train_df['Close']
    X_eval  = eval_df.drop(columns=['Close'])
    y_eval  = eval_df['Close']

    return X_train, y_train, X_eval, y_eval


def train_and_save(ticker: str):
    """
    Train an XGBoost regressor with recency weighting (last days emphasized),
    evaluate on the eval set, then serialize the trained model to disk.
    """
    print(f"[ ] Loading data for {ticker}")
    X_train, y_train, X_eval, y_eval = load_train_eval(ticker)

    # Remove outliers from training target
    lower = y_train.quantile(0.0)
    upper = y_train.quantile(1)
    mask = (y_train >= lower) & (y_train <= upper)
    removed = len(y_train) - mask.sum()
    X_train, y_train = X_train[mask], y_train[mask]
    print(f"[ ] Removed {removed} outlier rows from training data")

    # Check for empty training set
    if X_train.empty:
        print(f"[!] No training data available after outlier removal for {ticker}. Skipping training.")
        return

    # Compute recency weights: emphasize more recent data
    dates = X_train.index
    earliest = dates.min()
    days_since = (dates - earliest).days.astype(float)
    # Add 1 to avoid zero weight, normalize to [0,1]
    weights = (days_since + 1) / (days_since.max() + 1)
    
    print(f"[ ] Training XGBRegressor on {len(X_train)} samples with recency weighting...")
    model = XGBRegressor(n_estimators=100, verbosity=1)
    model.fit(X_train, y_train, sample_weight=weights)

    print("[ ] Evaluating on current-month data...")
    preds = model.predict(X_eval)
    mse = mean_squared_error(y_eval, preds)
    print(f"[✓] Eval MSE for {ticker}: {mse:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    joblib.dump(model, out_path)
    print(f"[✓] Saved model to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and save a recency-weighted stock-prediction model"
    )
    parser.add_argument(
        '--ticker', required=True,
        help="Ticker symbol to train on (e.g., AAPL)"
    )
    args = parser.parse_args()
    train_and_save(args.ticker)
