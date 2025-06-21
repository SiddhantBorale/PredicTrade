# server/model/predict.py
import yfinance as yf
import pandas as pd
import joblib
import json
import os
from datetime import datetime

def download_recent_data(ticker, days=30):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df = df[['Close', 'Volume']].copy()
    df['Return'] = df['Close'].pct_change()
    return df.dropna()

def engineer_features(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['Volatility'] = df['Return'].rolling(5).std()
    df = df.dropna()
    return df[['Close', 'Volume', 'MA5', 'Volatility']]

def make_prediction(ticker):
    model_path = f"model/{ticker}_model.pkl"
    output_path = f"data/{ticker}_prediction.json"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}")

    model = joblib.load(model_path)
    df = download_recent_data(ticker)
    features = engineer_features(df)
    
    latest = features.iloc[-1:]
    pred_prob = model.predict_proba(latest)[0]
    pred_class = model.predict(latest)[0]

    result = {
        "ticker": ticker,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "prediction": "UP" if pred_class == 1 else "DOWN",
        "confidence": round(float(max(pred_prob)), 4)
    }

    os.makedirs("data", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Prediction saved to {output_path}")
    return result

if __name__ == "__main__":
    make_prediction("AAPL")
