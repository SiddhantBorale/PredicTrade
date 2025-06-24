import os
import argparse
import joblib
import pandas as pd
from datetime import timedelta

# Local imports (src/ is on PYTHONPATH when invoked from project root)
from preprocess import load_last_six_months
from features import compute_features

# Paths relative to project root
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def predict_next_day(ticker: str):
    # 1) Load last six months of raw data
    print(f"[ ] Loading raw data for {ticker}")
    raw_df = load_last_six_months(ticker)

    # 2) Compute features
    print("[ ] Computing features on latest data window")
    feat_df = compute_features(raw_df)

    # 3) Extract the most recent feature vector
    X_today = feat_df.drop(columns=['Close']).iloc[[-1]]
    last_date = feat_df.index[-1]
    pred_date = last_date + timedelta(days=1)

    # 4) Load trained model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    print(f"[ ] Loading model from {model_path}")
    model = joblib.load(model_path)

    # 5) Predict
    print(f"[ ] Generating prediction for {pred_date.date()}")
    pred_value = model.predict(X_today)[0]
    print(f"[✓] {ticker} next-day predicted Close: {pred_value:.4f} for {pred_date.date()}")

    # 6) Append to predictions CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, 'predictions.csv')
    row = pd.DataFrame([{  
        'date': pred_date.date(),
        'ticker': ticker,
        'predicted_close': pred_value
    }])
    if not os.path.exists(out_csv):
        row.to_csv(out_csv, index=False)
        print(f"[✓] Created predictions log at {out_csv}")
    else:
        row.to_csv(out_csv, mode='a', header=False, index=False)
        print(f"[✓] Appended to predictions log at {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict next-day stock price and log the result"
    )
    parser.add_argument(
        '--ticker', required=True,
        help="Ticker symbol to predict (e.g., AAPL)"
    )
    args = parser.parse_args()
    predict_next_day(args.ticker)
