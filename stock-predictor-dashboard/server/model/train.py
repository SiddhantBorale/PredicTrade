# server/model/train.py
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os

def download_data(ticker, start="2020-01-01"):
    df = yf.download(ticker, start=start, interval="1d")
    df = df[['Close', 'Volume']].copy()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = df['Return'].shift(-5) > 0  # Will price go UP after 5 days?
    df = df.dropna()
    return df

def engineer_features(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['Volatility'] = df['Return'].rolling(5).std()
    df = df.dropna()
    return df

def train_model(ticker):
    df = download_data(ticker)
    df = engineer_features(df)

    X = df[['Close', 'Volume', 'MA5', 'Volatility']]
    y = df['Target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, f"model/{ticker}_model.pkl")
    print(f"Model trained for {ticker}. Accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    train_model("AAPL")
