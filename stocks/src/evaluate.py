import os
import joblib
import pandas as pd
import argparse
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths relative to this file's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def load_eval_data(ticker: str):
    """
    Load processed evaluation data for the given ticker.
    Returns:
        X_eval (DataFrame), y_eval (Series)
    """
    eval_path = os.path.join(PROCESSED_DIR, f"{ticker}_eval.csv")
    df = pd.read_csv(eval_path, index_col='Date', parse_dates=['Date'])
    X_eval = df.drop(columns=['Close'])
    y_eval = df['Close']
    return X_eval, y_eval


def evaluate_and_save(ticker: str):
    """
    Load the trained model, evaluate on the current-month data,
    and save metrics to a JSON file.
    """
    # Load model
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    print(f"[ ] Loading model from {model_path}")
    model = joblib.load(model_path)

    # Load eval set
    print(f"[ ] Loading evaluation data for {ticker}")
    X_eval, y_eval = load_eval_data(ticker)

    # Predict and compute metrics
    print(f"[ ] Running predictions on {len(X_eval)} samples...")
    preds = model.predict(X_eval)
    mse = mean_squared_error(y_eval, preds)
    mae = mean_absolute_error(y_eval, preds)
    print(f"[✓] Eval results for {ticker} → MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'ticker': ticker,
        'n_samples': len(X_eval),
        'mse': mse,
        'mae': mae
    }
    out_path = os.path.join(RESULTS_DIR, f"{ticker}_eval_results.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"[✓] Saved evaluation results to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a stock-prediction model on current-month data"
    )
    parser.add_argument(
        '--ticker', required=True,
        help="Ticker symbol to evaluate (e.g., AAPL)"
    )
    args = parser.parse_args()
    evaluate_and_save(args.ticker)
