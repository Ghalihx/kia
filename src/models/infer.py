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
    """
    Iteratif per horizon. Jika fitur lag/rolling yang dibutuhkan tidak bisa dibangun (karena
    dependensi ke horizon sebelumnya), akan berhenti lebih cepat.
    """
    import xgboost as xgb

    dcol = artifact["date_column"]
    ycol = artifact["target_column"]
    feature_cols = artifact["feature_columns"]

    hist = df_hist[[dcol, ycol]].copy()
    hist[dcol] = pd.to_datetime(hist[dcol])
    hist = hist.sort_values(dcol).reset_index(drop=True)

    # Rekonstruksi booster
    booster_raw = artifact.get("xgb_model_raw")
    if booster_raw is None:
        raise ValueError("xgboost model_raw tidak tersedia.")
    booster = xgb.Booster()
    booster.load_model(bytearray(booster_raw, encoding="latin1", errors="ignore"))

    rows = []
    # Kita butuh generate fitur baru setiap langkah
    for h in range(1, horizon + 1):
        next_date = hist[dcol].iloc[-1] + pd.DateOffset(months=1)
        # Bangun baris fitur dari hist terbaru
        temp = hist.copy()

        # Fitur dasar
        month = next_date.month
        year = next_date.year
        t_val = len(temp)

        # Lags
        feat = {}
        for col in feature_cols:
            # Isi default
            feat[col] = 0.0

        # Isi fitur lags kalau ada
        for c in feature_cols:
            if c.startswith("lag_"):
                try:
                    lag_n = int(c.split("_")[1])
                    if len(temp) >= lag_n:
                        feat[c] = float(temp[ycol].iloc[-lag_n])
                except Exception:
                    pass
            elif c.startswith("roll_mean_"):
                try:
                    w = int(c.split("_")[-1])
                    if len(temp) >= w:
                        feat[c] = float(temp[ycol].tail(w).mean())
                except Exception:
                    pass
            elif c.startswith("roll_std_"):
                try:
                    w = int(c.split("_")[-1])
                    if len(temp) >= w:
                        feat[c] = float(temp[ycol].tail(w).std(ddof=0))
                except Exception:
                    pass
            elif c == "month":
                feat[c] = month
            elif c == "year":
                feat[c] = year
            elif c == "t":
                feat[c] = t_val
            elif c == "month_sin":
                feat[c] = float(np.sin(2 * np.pi * month / 12))
            elif c == "month_cos":
                feat[c] = float(np.cos(2 * np.pi * month / 12))
            elif c == "diff_1":
                if len(temp) >= 1:
                    feat[c] = float(temp[ycol].iloc[-1] - temp[ycol].iloc[-2]) if len(temp) >= 2 else 0.0
            elif c == "diff_12":
                if len(temp) >= 13:
                    feat[c] = float(temp[ycol].iloc[-1] - temp[ycol].iloc[-13])

        # Cek kelengkapan minimal (misal butuh lag terbesar)
        needed_lags = [int(c.split("_")[1]) for c in feature_cols if c.startswith("lag_")]
        if needed_lags:
            max_lag = max(needed_lags)
            if len(temp) < max_lag:
                # Tidak bisa lanjut horizon berikut
                break

        feat_vec = np.array([[feat[c] for c in feature_cols]], dtype=float)
        dmatrix = xgb.DMatrix(feat_vec, feature_names=feature_cols)
        y_pred = float(booster.predict(dmatrix)[0])

        rows.append({dcol: next_date, "y_pred": y_pred})
        # Append ke hist untuk horizon selanjutnya
        hist = pd.concat([hist, pd.DataFrame({dcol: [next_date], ycol: [y_pred]})],
                         ignore_index=True)

    return pd.DataFrame(rows)
