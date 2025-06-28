import pandas as pd

def compute_features(df: pd.DataFrame,
                     ema_windows=(5, 10, 20),
                     rsi_period: int = 14) -> pd.DataFrame:
    """
    Given a DataFrame with a Date index and a 'Close' column,
    returns a DataFrame with:
      - Exponential moving averages (EMA) for each window in ema_windows
      - Simple moving averages (SMA) with min_periods=1
      - Lagged returns (1-day and 5-day), filled with zero where undefined
      - RSI over rsi_period days, with the first value forward/back-filled
    No rows are dropped.
    """
    df = df.copy()

    # 1) EMAs (no NaNs) and SMAs (min_periods=1)
    for w in ema_windows:
        df[f'ema_{w}'] = df['Close'].ewm(span=w, adjust=False, min_periods=1).mean()
        df[f'sma_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()

    # 2) Lagged returns
    df['return_1d'] = df['Close'].pct_change(1).fillna(0)
    df['return_5d'] = df['Close'].pct_change(5).fillna(0)

    # 3) RSI
    delta = df['Close'].diff().fillna(0)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False, min_periods=1).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, 1e-8)
    df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
    # Fill any NaN that crept in (e.g. first row)
    df[f'rsi_{rsi_period}'] = df[f'rsi_{rsi_period}'].fillna(method='bfill').fillna(method='ffill')

    return df
