from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def load_artifact(prefix_path: str):
    """
    prefix_path contoh: models/kia_forecast
    Akan mencari file prefix_path_latest.pkl
    """
    p = Path(f"{prefix_path}_latest.pkl")
    if not p.exists():
        raise FileNotFoundError(f"Artifact tidak ditemukan: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def forecast_iterative_naive(df_hist: pd.DataFrame, cfg: dict, horizon: int) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    hist = df_hist[[dcol, ycol]].copy()
    hist[dcol] = pd.to_datetime(hist[dcol])
    hist = hist.sort_values(dcol)
    last_date = hist[dcol].iloc[-1]
    last_val = float(hist[ycol].iloc[-1])
    rows = []
    for h in range(1, horizon + 1):
        nd = last_date + pd.DateOffset(months=h)
        rows.append({dcol: nd, "y_pred": last_val})
    return pd.DataFrame(rows)


def forecast_iterative_naive_drift(df_hist: pd.DataFrame, cfg: dict, horizon: int) -> pd.DataFrame:
    """
    Naive dengan drift linear:
    drift = mean(y_t - y_{t-1}) pada seluruh history.
    Forecast(h) = last_val + h * drift
    """
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    hist = df_hist[[dcol, ycol]].copy()
    hist[dcol] = pd.to_datetime(hist[dcol])
    hist = hist.sort_values(dcol)
    if len(hist) < 2:
        drift = 0.0
    else:
        diffs = hist[ycol].diff().dropna()
        drift = float(diffs.mean())
    last_date = hist[dcol].iloc[-1]
    last_val = float(hist[ycol].iloc[-1])
    rows = []
    for h in range(1, horizon + 1):
        nd = last_date + pd.DateOffset(months=h)
        yhat = last_val + h * drift
        rows.append({dcol: nd, "y_pred": yhat})
    return pd.DataFrame(rows)


def forecast_iterative_seasonal_naive(df_hist: pd.DataFrame,
                                      cfg: dict,
                                      horizon: int,
                                      season_length: int = 12) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    hist = df_hist[[dcol, ycol]].copy()
    hist[dcol] = pd.to_datetime(hist[dcol])
    hist = hist.sort_values(dcol)
    last_date = hist[dcol].iloc[-1]

    rows = []
    for h in range(1, horizon + 1):
        nd = last_date + pd.DateOffset(months=h)
        prev_season = nd - pd.DateOffset(months=season_length)
        val = hist.loc[hist[dcol] == prev_season, ycol]
        if val.empty:
            # fallback: bulan sebelumnya
            prev1 = nd - pd.DateOffset(months=1)
            val = hist.loc[hist[dcol] == prev1, ycol]
        if val.empty:
            # fallback akhir
            yhat = float(hist[ycol].iloc[-1])
        else:
            yhat = float(val.values[0])
        rows.append({dcol: nd, "y_pred": yhat})
    return pd.DataFrame(rows)


def forecast_iterative_xgb(df_hist: pd.DataFrame,
                           artifact: dict,
                           horizon: int) -> pd.DataFrame:
    import xgboost as xgb
    import numpy as np

    dcol = artifact["date_column"]
    ycol = artifact["target_column"]
    feature_cols = artifact["feature_columns"]
    use_log = artifact.get("use_log_target", False)

    hist = df_hist[[dcol, ycol]].copy()
    hist[dcol] = pd.to_datetime(hist[dcol])
    hist = hist.sort_values(dcol).reset_index(drop=True)

    booster_raw = artifact.get("xgb_model_raw")
    if booster_raw is None:
        raise ValueError("xgboost model_raw tidak tersedia.")
    booster = xgb.Booster()
    booster.load_model(bytearray(booster_raw, encoding="latin1", errors="ignore"))

    rows = []
    for _ in range(horizon):
        next_date = hist[dcol].iloc[-1] + pd.DateOffset(months=1)
        feat_map = {c:0.0 for c in feature_cols}

        # Rekonstruksi fitur (mirip di train)
        for c in feature_cols:
            if c.startswith("lag_"):
                lag_n = int(c.split("_")[1])
                if len(hist) >= lag_n:
                    feat_map[c] = float(hist[ycol].iloc[-lag_n])
            elif c.startswith("roll_mean_"):
                w = int(c.split("_")[-1])
                if len(hist) >= w:
                    feat_map[c] = float(hist[ycol].tail(w).mean())
            elif c.startswith("roll_std_"):
                w = int(c.split("_")[-1])
                if len(hist) >= w:
                    feat_map[c] = float(hist[ycol].tail(w).std(ddof=0))
            elif c.startswith("roll_med_"):
                w = int(c.split("_")[-1])
                if len(hist) >= w:
                    feat_map[c] = float(hist[ycol].tail(w).median())
            elif c == "month":
                feat_map[c] = next_date.month
            elif c == "year":
                feat_map[c] = next_date.year
            elif c == "t":
                feat_map[c] = len(hist)
            elif c == "month_sin":
                feat_map[c] = float(np.sin(2*np.pi*next_date.month/12))
            elif c == "month_cos":
                feat_map[c] = float(np.cos(2*np.pi*next_date.month/12))
            elif c == "diff_1":
                if len(hist) >= 2:
                    feat_map[c] = float(hist[ycol].iloc[-1] - hist[ycol].iloc[-2])
            elif c == "diff_12":
                if len(hist) >= 13:
                    feat_map[c] = float(hist[ycol].iloc[-1] - hist[ycol].iloc[-13])
            elif c == "pct_1":
                if len(hist) >= 2 and hist[ycol].iloc[-2] != 0:
                    feat_map[c] = (hist[ycol].iloc[-1] - hist[ycol].iloc[-2]) / hist[ycol].iloc[-2] * 100
            elif c == "pct_12":
                if len(hist) >= 13 and hist[ycol].iloc[-13] != 0:
                    feat_map[c] = (hist[ycol].iloc[-1] - hist[ycol].iloc[-13]) / hist[ycol].iloc[-13] * 100
            elif c.startswith("ratio_to_mean_"):
                w = int(c.split("_")[-1])
                if len(hist) >= w:
                    meanw = hist[ycol].tail(w).mean()
                    if meanw != 0:
                        feat_map[c] = hist[ycol].iloc[-1] / meanw

        # event / intensitas (future default 0)
        for c in ["event_flag","event_intensity","event_flag_lag1","post_event_1"]:
            if c in feature_cols and c not in feat_map:
                feat_map[c] = 0.0
        if "event_flag_lag1" in feature_cols:
            feat_map["event_flag_lag1"] = feat_map.get("event_flag",0.0)

        feat_vec = np.array([[feat_map[c] for c in feature_cols]], dtype=float)
        dm = xgb.DMatrix(feat_vec, feature_names=feature_cols)
        pred_raw = booster.predict(dm)[0]
        if use_log:
            yhat = float(np.expm1(pred_raw))
        else:
            yhat = float(pred_raw)

        rows.append({dcol: next_date, "y_pred": yhat})
        hist = pd.concat([hist, pd.DataFrame({dcol:[next_date], ycol:[yhat]})], ignore_index=True)

    return pd.DataFrame(rows)
