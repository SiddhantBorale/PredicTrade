from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from model.predict import make_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For now allow all; you can later restrict to GitHub Codespace URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is live"}

@app.get("/predict")
def predict(ticker: str = "AAPL"):
    result = make_prediction(ticker)
    return result

@app.get("/predictions")
def get_all_predictions():
    path = "stock-predictor-dashboard/server/data/predictions_log.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []
