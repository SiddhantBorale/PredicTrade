# PredicTrade — Live Stock Price Prediction Dashboard

Predict short-term stock prices from recent history, visualize trends with a sleek D3 dashboard, and manage a personal watchlist.  
**Stack:** Python (TensorFlow/Keras) LSTM • FastAPI • Node/Express • MongoDB • React (Vite) • D3

> **Use:** Forecast next-day (or next-N) closing prices using the last **9 months** of price/volume data plus technical indicators, and stream results to a modern web UI.

---

## Features

- **LSTM forecasts** trained on ~9 months of OHLCV data per ticker
- **Technical features**: SMA/EMA, RSI, MACD, Bollinger bands, returns, volatility
- **FastAPI ML service** for training & inference
- **Express API** for auth, watchlists, caching, and proxying to ML service
- **React + D3** dashboard: candlesticks, overlays, error analytics, portfolio slice
- **Live updates** via WebSocket/SSE (optional)
- **MongoDB** for users, watchlists, cached predictions, and metrics
- **Testing**: `pytest` (ML), `jest`/`react-testing-library` (UI), `supertest` + `mongodb-memory-server` (API)


---

## Quick Start

### 0) Prerequisites
- **Python** 3.10+
- **Node.js** 18+ and **npm** 9+
- **MongoDB** 6+ (local) or MongoDB Atlas

### 1) Clone
```bash
git clone <your-repo-url> predictrade
cd predictrade
```

### 2) ML Service (Python + FastAPI)
```bash
cd ml
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (Optional) Download 9mo of data for a few tickers
python scripts/download_data.py --tickers AAPL MSFT NVDA --period 9mo

# Train or fine-tune an LSTM for a ticker
python scripts/train.py --ticker AAPL --lookback 60 --horizon 1

# Start the ML API
uvicorn app.main:app --reload --port 8001
```

### 3) API Server (Node/Express)
```bash
cd ../server
npm i
cp .env.example .env        # then edit values (see ENV section below)
npm run dev                 # starts on http://localhost:3001
```

### 4) Client (React + Vite + D3)
```bash
cd ../client
npm i
cp .env.example .env        # then set VITE_API_URL
npm run dev                 # opens http://localhost:5173
```

Now open the client and search a ticker (e.g., AAPL). The server proxies inference to the ML service and returns predictions + charts.

---

## ⚙️ Environment Variables

### `server/.env`
```
NODE_ENV=development
PORT=3001
MONGO_URI=mongodb://localhost:27017/predictrade
JWT_SECRET=change-me
PYTHON_SERVICE_URL=http://localhost:8001
# Optional market data provider keys (fallback to yfinance if unset)
ALPHA_VANTAGE_KEY=
```

### `client/.env`
```
VITE_API_URL=http://localhost:3001
VITE_WS_URL=ws://localhost:3001   # if using websockets
```

### `ml/.env`
```
# Paths and toggles for the ML service
MODEL_DIR=./models
DATA_DIR=./data
USE_GPU=false
```

---

## How It Works

1. **Data**: 9 months of daily OHLCV per ticker (via yfinance/Alpha Vantage).  
2. **Feature Engineering**: rolling window (e.g., 60 days), normalized OHLCV + indicators (SMA/EMA/RSI/MACD, bands, returns).  
3. **Model**: Keras LSTM → Dense head; loss: MSE; metrics: RMSE/MAPE.  
4. **Serving**: FastAPI exposes `/predict` and `/train`.  
5. **API**: Express validates requests, handles users/watchlists, caches model outputs in MongoDB, and proxies to FastAPI.  
6. **UI**: React + D3 renders candlesticks with overlays, forecast bands, error charts, and a portfolio pie (5–6 slices).


---


