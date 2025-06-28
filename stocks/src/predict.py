import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import argparse
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

from preprocess import parse_period_to_days, load_last_period

# Paths
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def forecast_sarimax(ticker: str, period: str, horizon: int):
    """
    Fits a SARIMAX(1,1,1)x(1,1,1,7) on the close series and
    forecasts the next `horizon` days with 95% CI.
    """
    # 1) Load history
    days = parse_period_to_days(period)
    df = load_last_period(ticker, days)
    ts = df['Close']

    # 2) Fit SARIMAX with weekly seasonality
    model = SARIMAX(
        ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    # 3) Forecast
    pred = res.get_forecast(steps=horizon)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    # 4) Build output DataFrame
    # last_date = ts.index.max()
    # dates = [ last_date + pd.Timedelta(days=i) for i in range(1, horizon+1) ]
    last_date = ts.index.max()
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=last_date + us_bd,
                          periods=horizon,
                          freq=us_bd)

    out = pd.DataFrame({
        'date': dates,
        'ticker': ticker,
        'forecast_close': mean.values,
        'lower_95ci':   ci.iloc[:, 0].values,
        'upper_95ci':   ci.iloc[:, 1].values
    })

    # 5) Save & display
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{ticker}_sarimax_{horizon}d.csv")
    out.to_csv(path, index=False)
    print(f"[✓] SARIMAX {horizon}-day forecast saved to {path}")
    print(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="7-day SARIMAX forecast")
    p.add_argument('--ticker',  required=True, help="Ticker symbol (e.g. AAPL)")
    p.add_argument('--period',  default='6mo',  help="Look-back window (e.g. 6mo, 1y)")
    p.add_argument('--horizon', type=int, default=7,    help="Days to forecast")
    args = p.parse_args()
    forecast_sarimax(args.ticker, args.period, args.horizon)
