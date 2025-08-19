import os
import pandas as pd
from utils import RESULTS_DIR, safe_ticker

def _load_series(path: str, value_col: str = 'forecast_close'):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df[['date', value_col]].rename(columns={value_col: os.path.splitext(os.path.basename(path))[0]})

def fit_and_predict_ensemble(ticker: str, horizon: int = 7) -> pd.DataFrame:
    """
    Average any available model forecasts among: xgb, sarimax, lstm.
    Saves results/{SAFE}_ensemble_{h}d.csv
    """
    safe = safe_ticker(ticker)

    paths = {
        'xgb':     os.path.join(RESULTS_DIR, f"{safe}_xgb_{horizon}d.csv"),
        'sarimax': os.path.join(RESULTS_DIR, f"{safe}_sarimax_{horizon}d.csv"),
        'lstm':    os.path.join(RESULTS_DIR, f"{safe}_lstm_{horizon}d.csv"),
    }

    frames = []
    for model, p in paths.items():
        if os.path.exists(p):
            df = pd.read_csv(p)[['date','forecast_close']].rename(columns={'forecast_close': model})
            frames.append(df.set_index('date'))
    if not frames:
        print("[!] No model forecasts found to ensemble.")
        return None

    merged = pd.concat(frames, axis=1, join='inner')
    merged['forecast_close'] = merged.mean(axis=1)
    out = merged[['forecast_close']].reset_index()
    out['ticker'] = ticker

    out_path = os.path.join(RESULTS_DIR, f"{safe}_ensemble_{horizon}d.csv")
    out.to_csv(out_path, index=False)
    print(f"[✓] Saved ensemble {horizon}-day forecast to {out_path}")
    return out
