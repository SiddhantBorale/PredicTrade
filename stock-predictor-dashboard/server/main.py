# server/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from model.predict import make_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend origin later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_LOG = "data/predictions_log.json"

@app.get("/")
def root():
    return {"message": "Stock predictor API is running!"}

@app.get("/predict")
def predict(ticker: str = "AAPL"):
    result = make_prediction(ticker)
    return result

@app.get("/predictions")
def get_all_predictions():
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "r") as f:
            return json.load(f)
    return []
