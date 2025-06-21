# server/model/predict.py
import yfinance as yf
import pandas as pd
import joblib
import json
import os
from datetime import datetime, timedelta

LOG_PATH = "data/predictions_log.json"

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def download_recent_data(ticker, days=30):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df = df[['Close', 'Volume']].copy()
    df['Return'] = df['Close'].pct_change()
    return df.dropna()

def engineer_features(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['Volatility'] = df['Return'].rolling(5).std()
    return df.dropna()[['Close', 'Volume', 'MA5', 'Volatility']]

def evaluate_previous_predictions(ticker, df_today):
    logs = load_json(LOG_PATH)
    updated_logs = []
    today = datetime.today().date()

    for entry in logs:
        if entry["ticker"] != ticker or "actual_result" in entry:
            updated_logs.append(entry)
            continue

        pred_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
        if today - pred_date >= timedelta(days=7):
            # time to evaluate
            new_price = df_today['Close'].iloc[-1]
            old_price = entry['close_price_at_prediction']
            correct = (new_price > old_price and entry["prediction"] == "UP") or \
                      (new_price < old_price and entry["prediction"] == "DOWN")
            entry["actual_price"] = round(new_price, 2)
            entry["actual_result"] = "correct" if correct else "wrong"

        updated_logs.append(entry)

    save_json(updated_logs, LOG_PATH)

def make_prediction(ticker):
    model_path = f"model/{ticker}_model.pkl"
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}")

    model = joblib.load(model_path)
    df = download_recent_data(ticker)
    features = engineer_features(df)
    latest = features.iloc[-1:]

    pred_class = model.predict(latest)[0]
    pred_prob = model.predict_proba(latest)[0]

    result = {
        "date": datetime.today().strftime("%Y-%m-%d"),
        "ticker": ticker,
        "prediction": "UP" if pred_class == 1 else "DOWN",
        "confidence": round(float(max(pred_prob)), 4),
        "close_price_at_prediction": round(df['Close'].iloc[-1], 2)
    }

    logs = load_json(LOG_PATH)
    log

def auto_retrain_if_needed(ticker):
    logs = load_json(LOG_PATH)
    recent = [log for log in logs if log["ticker"] == ticker and "actual_result" in log][-5:]

    if len(recent) < 5:
        return

    accuracy = sum(1 for r in recent if r["actual_result"] == "correct") / 5
    print(f"Recent accuracy for {ticker}: {accuracy:.2f}")

    if accuracy < 0.6:
        print("🔁 Retraining model due to low accuracy...")
        from model.train import train_model
        train_model(ticker)