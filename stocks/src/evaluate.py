import os
import joblib
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR     = os.path.join(BASE_DIR, 'models')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')


def load_eval_data(ticker: str):
    """
    Load evaluation DataFrame for the given ticker, ensuring numeric dtypes.
    Returns:
        X_eval (DataFrame), y_eval (Series)
    """
    eval_path = os.path.join(PROCESSED_DIR, f"{ticker}_eval.csv")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    df = pd.read_csv(eval_path, index_col='Date', parse_dates=['Date'])
    # Convert all columns to numeric, coerce errors
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    if 'Close' not in df.columns:
        raise KeyError(f"Expected 'Close' column in evaluation data for {ticker}")

    X_eval = df.drop(columns=['Close'])
    y_eval = df['Close']
    return X_eval, y_eval


def evaluate_and_save(ticker: str):
    """
    Load the trained model, evaluate on the evaluation split,
    and save metrics to a JSON file.
    Skips evaluation if no data.
    """
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    eval_results_path = os.path.join(RESULTS_DIR, f"{ticker}_eval_results.json")

    if not os.path.exists(model_path):
        print(f"[!] Model file not found: {model_path}. Skipping evaluation.")
        return

    print(f"[ ] Loading evaluation data for {ticker}")
    try:
        X_eval, y_eval = load_eval_data(ticker)
    except (FileNotFoundError, KeyError) as e:
        print(f"[!] {e}. Skipping evaluation.")
        return

    if X_eval.empty:
        print(f"[!] No evaluation samples for {ticker}. Skipping evaluation.")
        return

    print(f"[ ] Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"[ ] Running predictions on {len(X_eval)} samples...")
    preds = model.predict(X_eval)
    mse = mean_squared_error(y_eval, preds)
    mae = mean_absolute_error(y_eval, preds)
    print(f"[✓] Eval results for {ticker} → MSE: {mse:.4f}, MAE: {mae:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        'ticker': ticker,
        'n_samples': len(X_eval),
        'mse': mse,
        'mae': mae
    }
    with open(eval_results_path, 'w') as f:
        import json
        json.dump(results, f, indent=4)
    print(f"[✓] Saved evaluation results to {eval_results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a stock-prediction model on a split"
    )
    parser.add_argument('--ticker', required=True, help="Ticker symbol (e.g., AAPL)")
    args = parser.parse_args()
    evaluate_and_save(args.ticker)
