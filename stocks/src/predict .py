# src/predict.py

import os
import argparse
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from preprocess import parse_period_to_days, load_last_period

# Paths
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def forecast_sarimax(ticker: str, period: str, horizon: int):
    # 1) Load raw history (no weekends/holidays) and get Close series
    days = parse_period_to_days(period)
    df = load_last_period(ticker, days)
    raw_close = df['Close']

    # 2) Build a business-day index with US federal holidays
    bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    full_idx = pd.date_range(
        start=raw_close.index.min(),
        end=raw_close.index.max(),
        freq=bd
    )
    # Reindex + forward-fill so it's truly regular
    ts = raw_close.reindex(full_idx).ffill()
    ts.index.freq = bd  # attach the freq so SARIMAX won't warn

    # 3) Decide seasonal_order vs. plain ARIMA
    seasonal_period = 7
    if len(ts) < seasonal_period * 4:
        # Not enough data to fit a reliable weekly seasonal component
        seasonal = (0, 0, 0, 0)
    else:
        seasonal = (1, 1, 1, seasonal_period)

    # 4) Fit the SARIMAX (or ARIMA) model
    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    # 5) Forecast the next `horizon` business days
    pred = res.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci   = pred.conf_int(alpha=0.05)

    # 6) Build output DataFrame
    #    The forecast index already carries the correct freq, so we can use it directly:
    forecast_index = mean.index
    out = pd.DataFrame({
        'date':         forecast_index.date,
        'ticker':       ticker,
        'forecast_close': mean.values,
        'lower_95ci':   ci.iloc[:, 0].values,
        'upper_95ci':   ci.iloc[:, 1].values
    })

    # 7) Save & display
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{ticker}_sarimax_{horizon}d.csv")
    out.to_csv(path, index=False)
    print(f"[✓] SARIMAX {horizon}-day forecast saved to {path}")
    print(out)

    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="7-day SARIMAX forecast")
    p.add_argument('--ticker',  required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument('--period',  default='6mo',  help="Look-back window (e.g. 6mo, 1y)")
    p.add_argument('--horizon', type=int, default=7,    help="Days to forecast")
    args = p.parse_args()
    forecast_sarimax(args.ticker, args.period, args.horizon)
