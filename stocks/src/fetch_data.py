# src/fetch_data.py
import os, io, time
from datetime import timedelta
import pandas as pd

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
os.makedirs(RAW_DIR, exist_ok=True)

def _period_to_days(period: str) -> int:
    p = str(period).lower().strip()
    if p == 'max': return 365 * 50
    if p.endswith('mo'): return int(p[:-2]) * 30
    if p.endswith('y'):  return int(p[:-1]) * 365
    # allow plain days number
    return int(p)

def _crop_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty: return df
    end = df.index.max()
    start = end - timedelta(days=days)
    return df.loc[start:end]

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    # standardize columns
    cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
    if not cols:
        # try title-case (stooq sometimes gives lower case)
        df.columns = [str(c).title() for c in df.columns]
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
    df = df[cols].copy()
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'Close'})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(how='all').sort_index()
    df.index.name = 'Date'
    return df

def _requests_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    retry = Retry(
        total=5, backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s

def _fetch_stooq_http(ticker: str) -> pd.DataFrame | None:
    import requests
    sym = ticker.upper()
    if '.' not in sym and not sym.startswith('^'):
        sym = f"{sym}.US"  # PLTR -> PLTR.US
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    try:
        r = _requests_session().get(url, timeout=15)
        txt = (r.text or "").strip()
        if r.status_code != 200 or not txt or txt.startswith('<'):
            return None
        df = pd.read_csv(io.StringIO(txt))
        if 'Date' not in df.columns:
            return None
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date').sort_index()
        # Standardize column names
        df.columns = [str(c).title() for c in df.columns]
        print("[i] fetched from Stooq CSV")
        return df
    except Exception as e:
        print(f"[stooq http] failed: {e}")
        return None

def _fetch_yahoo(ticker: str, period: str) -> pd.DataFrame | None:
    # Robust yfinance fetch with retries; still subject to 429 throttling
    try:
        import yfinance as yf
        sess = _requests_session()

        # Try download()
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker, period=period, interval="1d",
                    auto_adjust=True, threads=False, progress=False, session=sess
                )
                if df is not None and not df.empty:
                    print("[i] fetched from yfinance.download")
                    return df
            except Exception as e:
                print(f"[yfinance.download] attempt {attempt+1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))

        # Fallback to Ticker.history()
        try:
            T = yf.Ticker(ticker, session=sess)
            df = T.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                print("[i] fetched from yfinance.Ticker.history")
                return df
        except Exception as e:
            print(f"[yfinance.history] failed: {e}")
    except Exception as e:
        print(f"[yfinance import/use] failed: {e}")
    return None

def _save_csv(ticker: str, df: pd.DataFrame) -> str:
    path = os.path.join(RAW_DIR, f"{ticker.upper()}.csv")
    df.to_csv(path)
    print(f"[✓] Saved raw {ticker} → {path} ({len(df)} rows)")
    return path

def _recent_cached_path(ticker: str, ttl_seconds: int) -> str | None:
    path = os.path.join(RAW_DIR, f"{ticker.upper()}.csv")
    if not os.path.exists(path): return None
    age = time.time() - os.path.getmtime(path)
    return path if age <= ttl_seconds else None

def fetch_and_save(ticker: str, period: str) -> str:
    """Fetches daily OHLCV, normalizes columns, crops to 'period', saves CSV to data/raw, returns path.
       Honors env: PREFERRED_SOURCE (e.g., 'stooq,yahoo' or 'yahoo,stooq'), RAW_TTL_SECONDS (cache)."""
    # light caching to avoid re-fetch spam in dev/server
    ttl = int(os.environ.get("RAW_TTL_SECONDS", "0"))  # default off
    if ttl > 0:
        cached = _recent_cached_path(ticker, ttl)
        if cached:
            print(f"[i] using cached raw for {ticker} (TTL={ttl}s): {cached}")
            return cached

    order_env = os.environ.get("PREFERRED_SOURCE", "").strip().lower()
    if not order_env:
        order = ["stooq", "yahoo"]  # default: prefer stooq to avoid 429s
    else:
        order = [s.strip() for s in order_env.split(",") if s.strip()]

    days = _period_to_days(period)
    df = None
    for src in order:
        if src == "stooq":
            df = _fetch_stooq_http(ticker)
        elif src == "yahoo":
            df = _fetch_yahoo(ticker, period)
        else:
            print(f"[!] unknown source '{src}', skipping")
            continue
        if df is not None and not df.empty:
            df = _normalize(df)
            df = _crop_last_days(df, days)
            return _save_csv(ticker, df)

    raise RuntimeError(f"No data returned for {ticker} (sources tried: {order}, period={period})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", default="6mo")
    args = ap.parse_args()
    fetch_and_save(args.ticker, args.period)
