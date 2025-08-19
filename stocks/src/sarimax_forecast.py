import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import RESULTS_DIR, safe_ticker, next_trading_days

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))

def forecast_sarimax(ticker: str, period: str, horizon: int = 7) -> pd.DataFrame:
    """
    Fit a simple SARIMAX on Close and forecast next N trading days.
    """
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
    y = pd.to_numeric(df['Close'], errors='coerce').dropna()

    if len(y) < 15:
        warnings.warn("Too few rows for SARIMAX; returning flat forecast.")
        last = float(y.iloc[-1]) if len(y) else 0.0
        dates = next_trading_days(df.index.max(), horizon)
        out = pd.DataFrame({'date': dates, 'ticker': ticker, 'forecast_close': [last]*horizon})
    else:
        # make index have freq so statsmodels won't warn
        y = y.asfreq('B')  # business daily
        y = y.ffill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            fcast = res.get_forecast(steps=horizon)
            mean = fcast.predicted_mean
            conf = fcast.conf_int(alpha=0.05)
        dates = next_trading_days(df.index.max(), horizon)
        out = pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'forecast_close': mean.values
        })
        out['lower_95ci'] = conf.iloc[:,0].values
        out['upper_95ci'] = conf.iloc[:,1].values

    safe = safe_ticker(ticker)
    out_path = os.path.join(RESULTS_DIR, f"{safe}_sarimax_{horizon}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[✓] SARIMAX {horizon}-day forecast saved to {out_path}")
    return out
