import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
os.makedirs(RAW_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def _period_to_days(period: str) -> int:
    p = str(period).lower().strip()
    if p.endswith('mo'):
        return int(p[:-2]) * 30
    if p.endswith('y'):
        return int(p[:-1]) * 365
    if p == 'max':
        return 365 * 50
    return int(p)

def _stooq_symbol(ticker: str) -> str:
    """Best-effort mapping to Stooq symbols."""
    t = ticker.upper()
    # Common indices
    idx_map = {
        '^GSPC': '^SPX',     # S&P 500
        '^IXIC': '^NDQ',     # NASDAQ Composite
        '^DJI' : '^DJI',     # Dow Jones
    }
    if t in idx_map:
        return idx_map[t]
    # Most US equities use .US suffix on Stooq
    if '.' not in t and not t.startswith('^'):
        return f"{t}.US"
    return t

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Keep canonical OHLCV if present
    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
    if not keep and 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
        keep = ['Close']
    out = df[keep].copy()
    # Ensure numeric and sorted by date
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(how='all')
    out = out[~out.index.duplicated(keep='last')].sort_index()
    out.index.name = 'Date'
    return out

def _save_csv(ticker: str, df: pd.DataFrame) -> str:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"[✓] Saved raw {ticker} → {path} ({len(df)} rows)")
    return path


# ----------------------------
# Fetchers with fallbacks
# ----------------------------
def _fetch_yahoo_download(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        for attempt in range(3):
            try:
                df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f"[yfinance.download] attempt {attempt+1} failed: {e}")
            time.sleep(1.5 * (attempt + 1))
    except Exception as e:
        print(f"[yfinance] import/use failed: {e}")
    return None

def _fetch_yahoo_history(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        T = yf.Ticker(ticker)
        df = T.history(period=period, interval="1d", auto_adjust=True)
        if df is not None and not df.empty:
            # yfinance history sometimes uses 'Dividends'/'Stock Splits' cols too; ignore them
            return df
    except Exception as e:
        print(f"[yfinance.history] failed: {e}")
    return None

def _fetch_stooq(ticker: str, period: str) -> Optional[pd.DataFrame]:
    try:
        from pandas_datareader.stooq import StooqDailyReader
        days = _period_to_days(period)
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        sym = _stooq_symbol(ticker)
        reader = StooqDailyReader(symbols=sym, start=start, end=end)
        df = reader.read()
        if df is None or df.empty:
            return None
        # Stooq returns multiindex when multiple symbols; ensure single
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(sym)
        # Stooq is already OHLCV; ensure datetime index
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        print(f"[stooq] failed: {e}")
    return None

def _load_local_if_exists(ticker: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
            if df is not None and not df.empty:
                print(f"[i] Using existing local CSV for {ticker} at {path}")
                return df
        except Exception as e:
            print(f"[local csv read] failed: {e}")
    return None


# ----------------------------
# Public API
# ----------------------------
def fetch_and_save(ticker: str, period: str) -> str:
    """
    Robust fetch:
      1) yfinance.download (retry)
      2) yfinance.Ticker.history
      3) Stooq (pandas-datareader)
      4) existing local CSV (last resort)
    Saves normalized OHLCV to data/raw/{ticker}.csv
    """
    ticker = str(ticker).strip()

    # Try Yahoo (download)
    df = _fetch_yahoo_download(ticker, period)
    source = 'yahoo.download'

    # Fallback: Yahoo history
    if df is None or df.empty:
        df = _fetch_yahoo_history(ticker, period)
        source = 'yahoo.history'

    # Fallback: Stooq
    if df is None or df.empty:
        df = _fetch_stooq(ticker, period)
        source = 'stooq'

    # Fallback: local CSV (if present)
    if df is None or df.empty:
        df = _load_local_if_exists(ticker)
        source = 'local_csv'

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker} period={period}")

    # Normalize & save
    if not isinstance(df.index, pd.DatetimeIndex):
        # yfinance might return tz-aware index; convert
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = _normalize(df)
    print(f"[i] Data source used: {source}; rows={len(df)}")

    return _save_csv(ticker, df)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", default="6mo")
    args = ap.parse_args()
    fetch_and_save(args.ticker, args.period)
