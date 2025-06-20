# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

@app.get("/predict")
def predict(ticker: str):
    # dummy logic
    return {"ticker": ticker, "prediction": "UP", "confidence": 0.81}

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}