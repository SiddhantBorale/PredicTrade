import pandas as pd
import numpy as np

# We stick to Close-based indicators so recursive forecasting doesn't need future OHLCV.
DEFAULT_EMAS = (5, 10, 20)
RSI_PERIOD = 14

def compute_features(df: pd.DataFrame,
                     ema_windows=DEFAULT_EMAS,
                     rsi_period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Expects df with Date index and a numeric 'Close' column.
    Returns df with engineered features + original 'Close'.
    """
    out = df.copy()
    if 'Close' not in out.columns:
        raise KeyError("compute_features requires 'Close' column")

    out['Close'] = pd.to_numeric(out['Close'], errors='coerce')

    # EMAs & SMAs
    for w in ema_windows:
        out[f'ema_{w}'] = out['Close'].ewm(span=w, adjust=False).mean()
        out[f'sma_{w}'] = out['Close'].rolling(window=w, min_periods=1).mean()

    # Returns
    out['return_1d'] = out['Close'].pct_change(1)
    out['return_5d'] = out['Close'].pct_change(5)

    # RSI
    delta = out['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period - 1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period - 1, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    out[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))

    # Fill edges reasonably then drop residual NA rows
    out = out.ffill().bfill().dropna()

    # ensure numeric types
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna()

    return out

def feature_columns(df: pd.DataFrame) -> list:
    """All columns except target 'Close' are features."""
    return [c for c in df.columns if c != 'Close']
