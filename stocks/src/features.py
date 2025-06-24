import pandas as pd

def compute_features(df: pd.DataFrame, ema_windows=(5, 10, 20), rsi_period: int = 14) -> pd.DataFrame:
    """
    Given a DataFrame with a Date index and a 'Close' column,
    returns a new DataFrame with additional technical features:
      - Exponential moving averages (EMA) for each window in ema_windows
      - Simple moving averages (SMA) for each window in ema_windows
      - Lagged returns (1-day and 5-day)
      - RSI over rsi_period days
    Rows with any NaNs are dropped.
    """
    df = df.copy()
    
    # EMAs and SMAs
    for w in ema_windows:
        df[f'ema_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
        df[f'sma_{w}'] = df['Close'].rolling(window=w).mean()

    # Lagged returns
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = ma_up / ma_down
    df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))

    # Drop any rows with NaNs introduced by rolling/calculations
    df = df.dropna()
    return df
