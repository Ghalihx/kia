from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

try:
    import holidays as pyholidays
except Exception:
    pyholidays = None  # type: ignore


def _to_month_start(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _holiday_calendar(years, country_code: str):
    if pyholidays is None:
        return set()
    try:
        return pyholidays.country_holidays(country_code=country_code, years=sorted(set(years)))
    except Exception:
        return set()


def _holiday_count_for_month(ts: pd.Timestamp, hcal) -> int:
    if not hcal:
        return 0
    start = ts
    end = ts + pd.offsets.MonthEnd(0)
    days = pd.date_range(start, end, freq="D")
    return sum(1 for d in days if d in hcal)


def _prepare_hist(df_hist: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    if dcol not in df_hist.columns or ycol not in df_hist.columns:
        raise ValueError(f"Data historis harus mengandung kolom {dcol} dan {ycol}.")
    out = df_hist[[dcol, ycol]].copy()
    out[dcol] = out[dcol].apply(_to_month_start)
    out = out.groupby(dcol, as_index=False)[ycol].sum()
    out = out.sort_values(dcol).reset_index(drop=True)

    dates = out[dcol]
    full_range = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="MS")
    if len(full_range) != len(dates):
        missing = full_range.difference(dates)
        if len(missing) > 0:
            miss_list = [d.strftime("%Y-%m") for d in missing]
            raise ValueError(
                f"Data historis tidak kontigu. Bulan hilang: {miss_list}. Lengkapi atau isi 0."
            )
    return out


def _build_feature_row(
    hist: pd.DataFrame,
    next_date: pd.Timestamp,
    cfg: Dict,
    hcal,
) -> Dict:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    lags: List[int] = cfg["data"]["lags"]
    rollings: List[int] = cfg["data"]["rollings"]
    add_sin_cos = cfg["data"].get("add_sin_cos", True)

    row = {
        dcol: next_date,
        "month": next_date.month,
        "quarter": ((next_date.month - 1) // 3) + 1,
        "year": next_date.year,
        "holiday_count": _holiday_count_for_month(next_date, hcal),
    }
    if add_sin_cos:
        row["sin_month"] = np.sin(2 * np.pi * next_date.month / 12.0)
        row["cos_month"] = np.cos(2 * np.pi * next_date.month / 12.0)

    hist_indexed = hist.set_index(dcol)

    for lag in lags:
        lag_date = next_date - pd.DateOffset(months=lag)
        row[f"{ycol}_lag{lag}"] = float(hist_indexed.at[lag_date, ycol]) if lag_date in hist_indexed.index else np.nan

    for win in rollings:
        start_date = next_date - pd.DateOffset(months=win)
        window_vals = hist[(hist[dcol] >= start_date) & (hist[dcol] < next_date)][ycol]
        row[f"{ycol}_roll{win}"] = float(window_vals.mean()) if len(window_vals) == win else np.nan

    return row


def forecast_iterative_xgb(df_hist: pd.DataFrame, artifact: Dict, horizon: int) -> pd.DataFrame:
    if "model" not in artifact:
        raise ValueError("Artifact tidak memiliki 'model'.")
    if "feature_names" not in artifact:
        raise ValueError("Artifact tidak memiliki 'feature_names'.")
    model = artifact["model"]
    cfg = artifact["cfg"]
    feature_names = artifact["feature_names"]

    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]

    hist = _prepare_hist(df_hist, cfg)
    last_date = hist[dcol].iloc[-1]
    years_needed = list(range(hist[dcol].dt.year.min(), (last_date + pd.DateOffset(months=horizon)).year + 2))
    hcal = _holiday_calendar(years_needed, cfg["data"].get("holiday_country", "ID"))

    preds = []
    for step in range(1, horizon + 1):
        next_date = _to_month_start(last_date + pd.DateOffset(months=step))
        feat_row = _build_feature_row(hist, next_date, cfg, hcal)
        row_df = pd.DataFrame([feat_row])
        if row_df[feature_names].isna().any(axis=None):
            break
        yhat = float(model.predict(row_df[feature_names])[0])
        preds.append({dcol: next_date, "y_pred": yhat})
        hist = pd.concat([hist, pd.DataFrame([{dcol: next_date, ycol: yhat}])], ignore_index=True)
    return pd.DataFrame(preds)


def forecast_iterative_seasonal_naive(df_hist: pd.DataFrame, cfg: Dict, horizon: int, season_length: int = 12) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    hist = _prepare_hist(df_hist, cfg)
    hist_indexed = hist.set_index(dcol)
    last_date = hist[dcol].iloc[-1]
    preds = []
    for h in range(1, horizon + 1):
        next_date = _to_month_start(last_date + pd.DateOffset(months=h))
        season_date = next_date - pd.DateOffset(months=season_length)
        val = float(hist_indexed.at[season_date, ycol]) if season_date in hist_indexed.index else float(hist_indexed[ycol].iloc[-1])
        preds.append({dcol: next_date, "y_pred": val})
        hist = pd.concat([hist, pd.DataFrame([{dcol: next_date, ycol: val}])], ignore_index=True)
        hist_indexed = hist.set_index(dcol)
    return pd.DataFrame(preds)


def forecast_iterative_naive(df_hist: pd.DataFrame, cfg: Dict, horizon: int) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    hist = _prepare_hist(df_hist, cfg)
    last_value = float(hist[ycol].iloc[-1])
    last_date = hist[dcol].iloc[-1]
    preds = []
    for h in range(1, horizon + 1):
        next_date = _to_month_start(last_date + pd.DateOffset(months=h))
        preds.append({dcol: next_date, "y_pred": last_value})
    return pd.DataFrame(preds)


def load_artifact(prefix_path: str) -> Dict:
    meta_path = Path(prefix_path + ".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("model_name") == "xgboost":
        model_path = Path(prefix_path + "_model.pkl")
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                meta["model"] = model
            except Exception as e:
                meta["model_load_error"] = f"Gagal memuat model: {e}"
        else:
            meta["model_load_error"] = f"File model tidak ditemukan: {model_path}"
    return meta


__all__ = [
    "forecast_iterative_xgb",
    "forecast_iterative_seasonal_naive",
    "forecast_iterative_naive",
    "load_artifact",
]