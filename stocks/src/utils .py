import os
import re
from datetime import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
MODELS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def safe_ticker(ticker: str) -> str:
    return re.sub(r'[^A-Za-z0-9_-]', '_', str(ticker))

def to_float_df(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def next_trading_days(start_date: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    """
    Generate next n trading (Mon–Fri) days after start_date.
    (Simple weekend-skipper; for full holiday support integrate pandas-market-calendars later.)
    """
    days = []
    d = pd.Timestamp(start_date)
    while len(days) < n:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:  # Mon=0..Fri=4
            days.append(d.normalize())
    return days

def unify_features(X: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Reindex/align columns so they match the order used at training time.
    Missing cols are filled with 0.0; extra cols are dropped.
    """
    cols = list(feature_names)
    out = pd.DataFrame(0.0, index=X.index, columns=cols)
    for c in cols:
        if c in X.columns:
            out[c] = pd.to_numeric(X[c], errors='coerce').astype(float)
    return out
