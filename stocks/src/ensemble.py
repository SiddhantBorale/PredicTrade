import os
import pandas as pd
from sklearn.linear_model import Ridge

from predict_xgb import forecast_xgb
from predict import forecast_sarimax  # your SARIMAX script

# Paths
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def fit_and_predict_ensemble(ticker: str, period: str, horizon: int):
    """
    1) Load base forecasts from XGB + SARIMAX
    2) If there's past overlap in results/, train a Ridge meta‐model
    3) Apply meta‐model (or fallback to simple average) for the final horizon
    4) Save to results/{ticker}_ensemble_{horizon}d.csv
    """
    # --- 1) Get base forecasts ---
    xgb_df     = forecast_xgb(ticker, period, horizon).set_index('date')
    sarimax_df = forecast_sarimax(ticker, period, horizon).set_index('date')[['forecast_close']].rename(columns={'forecast_close':'sarimax_pred'})

    base = xgb_df.join(sarimax_df, how='inner')  # only dates both methods predict
    base['avg_pred'] = base[['xgb_pred','sarimax_pred']].mean(axis=1)

    # --- 2) Look for historical results to train meta-model ---
    hist_path = os.path.join(RESULTS_DIR, f"{ticker}_history_preds.csv")
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path, parse_dates=['date']).set_index('date')
        # hist must have columns ['xgb_pred','sarimax_pred','actual_close']
        df_train = hist.dropna(subset=['actual_close'])
        X_train = df_train[['xgb_pred','sarimax_pred']]
        y_train = df_train['actual_close']
        meta = Ridge(alpha=1.0)
        meta.fit(X_train, y_train)
        # 3) apply meta‐model
        base['ensemble'] = meta.predict(base[['xgb_pred','sarimax_pred']])
    else:
        # fallback to simple average if no history
        base['ensemble'] = base['avg_pred']

    # --- 4) Save ---
    out = base.reset_index()[['date','ensemble']].rename(columns={'ensemble':'forecast_close'})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{ticker}_ensemble_{horizon}d.csv")
    out.to_csv(path, index=False)
    print(f"[✓] Saved ensemble {horizon}-day forecast to {path}")
    print(out)

    # *** Optional: record this batch into history_preds.csv ***
    # You’ll need to merge in the actual future closes once they occur.
