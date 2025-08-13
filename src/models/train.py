from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Config loader (gunakan tomllib kalau Python >=3.11)
try:
    import tomllib  # type: ignore
    def _read_toml(path: Path) -> Dict[str, Any]:
        return tomllib.loads(path.read_bytes())
except ModuleNotFoundError:
    import toml  # type: ignore
    def _read_toml(path: Path) -> Dict[str, Any]:
        return toml.load(path.open("r", encoding="utf-8"))

# Event utilities (optional). Jika file tidak ada, kita fallback.
try:
    from src.utils.events import load_events, merge_event_features
    _HAS_EVENT_UTILS = True
except Exception:
    _HAS_EVENT_UTILS = False


# =========================
# Public API (diekspos)
# =========================
__all__ = [
    "load_config",
    "train_pipeline",
    "save_artifact",
    "load_artifact"
]


# =========================
# CONFIG
# =========================
def load_config(path: str | Path = "config.toml") -> dict:
    p = Path(path)
    if not p.exists():
        # fallback minimal agar app tetap jalan
        return {
            "data": {
                "date_column": "periode",
                "target_column": "permohonan_kia",
                "lags": [1, 2, 3, 6, 12],
                "rollings": [3, 6, 12],
                "add_sin_cos": True,
                "use_log_target": False,
                "use_events": False,
                "cap_outliers": False
            },
            "training": {
                "holdout_months": 6,
                "season_length": 12,
                "cv_folds": 3,
                "min_train_months": 18,
                "blend_enable": True,
                "xgb": {
                    "eta": 0.05,
                    "max_depth": 4,
                    "n_rounds": 500,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9
                }
            }
        }
    return _read_toml(p)


# =========================
# METRICS
# =========================
def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.any():
        return np.nan
    denom = (np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100)


# =========================
# BASELINE FORECASTERS
# =========================
def predict_naive_last(train_df: pd.DataFrame,
                       future_dates: List[pd.Timestamp],
                       date_col: str,
                       target_col: str) -> np.ndarray:
    """
    Naive last value repeated.
    """
    if train_df.empty:
        return np.array([np.nan] * len(future_dates))
    last_val = float(train_df[target_col].iloc[-1])
    return np.array([last_val] * len(future_dates), dtype=float)


def predict_seasonal_naive(train_df: pd.DataFrame,
                           future_dates: List[pd.Timestamp],
                           date_col: str,
                           target_col: str,
                           season_length: int = 12) -> np.ndarray:
    """
    Seasonal naive: y_{t} = y_{t - season_length}.
    Jika data belum cukup panjang, fallback ke naive_last.
    """
    values = train_df.sort_values(date_col)[target_col].values.astype(float)
    if len(values) < season_length:
        return predict_naive_last(train_df, future_dates, date_col, target_col)
    result = []
    # Simulasikan iteratif: ambil shift season_length ke belakang terhadap 'virtual' timeline
    hist = list(values)
    for _ in future_dates:
        if len(hist) >= season_length:
            val = hist[-season_length]
        else:
            val = hist[-1]
        result.append(float(val))
        hist.append(val)
    return np.array(result, dtype=float)


# =========================
# OUTLIER HANDLING
# =========================
def _cap_outliers(series: pd.Series, method: str = "iqr", k: float = 1.5) -> pd.Series:
    s = series.copy()
    if method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr
        return s.clip(lower=low, upper=high)
    elif method == "zscore":
        m = s.mean()
        std = s.std(ddof=0)
        if std == 0:
            return s
        z = (s - m) / std
        return s.where(z.abs() <= k, m + np.sign(z) * k * std)
    return s


# =========================
# EXPANDING CV
# =========================
def _expanding_cv_indices(n: int, n_folds: int, min_train: int, horizon: int = 1):
    folds = []
    start_test = min_train
    while start_test + horizon <= n and len(folds) < n_folds:
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, start_test + horizon)
        folds.append((train_idx, test_idx))
        start_test += horizon
    return folds


# =========================
# FEATURE ENGINEERING
# =========================
def build_features(df: pd.DataFrame,
                   date_col: str,
                   target_col: str,
                   lags: List[int],
                   rollings: List[int],
                   add_sin_cos: bool = True) -> pd.DataFrame:
    feat = df[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)

    for l in lags:
        feat[f"lag_{l}"] = feat[target_col].shift(l)

    for w in rollings:
        feat[f"roll_mean_{w}"] = feat[target_col].rolling(w).mean()
        feat[f"roll_std_{w}"] = feat[target_col].rolling(w).std()
        feat[f"roll_med_{w}"] = feat[target_col].rolling(w).median()

    feat["month"] = feat[date_col].dt.month
    feat["year"] = feat[date_col].dt.year
    feat["t"] = np.arange(len(feat))

    if add_sin_cos:
        feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
        feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    feat["diff_1"] = feat[target_col].diff(1)
    feat["diff_12"] = feat[target_col].diff(12)

    feat["pct_1"] = feat[target_col].pct_change(1) * 100
    feat["pct_12"] = feat[target_col].pct_change(12) * 100

    for w in [6, 12]:
        colm = f"roll_mean_{w}"
        if colm in feat:
            feat[f"ratio_to_mean_{w}"] = feat[target_col] / feat[colm]

    return feat


# =========================
# ARTIFACT I/O
# =========================
def save_artifact(artifact: Dict[str, Any],
                  out_dir: str = "models",
                  filename_prefix: str = "kia_forecast") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{filename_prefix}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    return file_path


def load_artifact(path_prefix: str) -> Dict[str, Any]:
    p = Path(f"{path_prefix}.json")
    if not p.exists():
        raise FileNotFoundError(f"Artifact tidak ditemukan: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# =========================
# TRAIN PIPELINE
# =========================
def train_pipeline(df: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    date_col = data_cfg.get("date_column", "periode")
    target_col = data_cfg.get("target_column", "permohonan_kia")
    lags = data_cfg.get("lags", [1, 2, 3, 6, 12])
    rollings = data_cfg.get("rollings", [3, 6, 12])
    add_sin_cos = bool(data_cfg.get("add_sin_cos", True))
    use_log = bool(data_cfg.get("use_log_target", False))
    use_events = bool(data_cfg.get("use_events", False))
    cap_outliers = bool(data_cfg.get("cap_outliers", False))
    outlier_method = data_cfg.get("outlier_method", "iqr")
    outlier_iqr_k = float(data_cfg.get("outlier_iqr_k", 1.5))

    holdout_months = int(train_cfg.get("holdout_months", 6))
    season_length = int(train_cfg.get("season_length", 12))
    cv_folds = int(train_cfg.get("cv_folds", 3))
    min_train_months = int(train_cfg.get("min_train_months", 18))

    # Validasi dasar
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Kolom {date_col}/{target_col} tidak ditemukan di data.")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[target_col].isna().any():
        raise ValueError("Ada nilai target NaN – bersihkan dulu.")

    df = df.sort_values(date_col).reset_index(drop=True)
    if len(df) <= holdout_months + 6:
        raise ValueError("Data terlalu sedikit untuk holdout konfigurasi saat ini.")

    # Outlier capping (opsional)
    if cap_outliers:
        df["_target_capped"] = _cap_outliers(df[target_col], method=outlier_method, k=outlier_iqr_k)
    else:
        df["_target_capped"] = df[target_col]

    # Event merge (opsional)
    if use_events and _HAS_EVENT_UTILS:
        events_path = data_cfg.get("event_file", "data/raw/events.csv")
        try:
            ev = load_events(events_path)
            df = df.rename(columns={target_col: "_orig_target"})
            df = merge_event_features(df.rename(columns={"_orig_target": target_col}),
                                      ev, date_col)
        except Exception as e:
            warnings.warn(f"Gagal merge event: {e}")
            df["event_flag"] = 0
            df["event_intensity"] = 0.0
    else:
        df["event_flag"] = 0
        df["event_intensity"] = 0.0

    # Kolom fit (bisa log)
    df["_fit_target"] = df["_target_capped"]
    if use_log:
        df["_fit_target_log"] = np.log1p(df["_fit_target"])
        fit_col = "_fit_target_log"
    else:
        fit_col = "_fit_target"

    # Split train/holdout
    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()
    test_dates = list(test_df[date_col])
    y_test_actual = test_df[target_col].astype(float).values

    # Build fitur untuk seluruh timeline pada basis fit_col
    feat_all = build_features(
        df[[date_col, fit_col]].rename(columns={fit_col: target_col}),
        date_col, target_col, lags, rollings, add_sin_cos
    )
    feat_all = feat_all.merge(df[[date_col, "event_flag", "event_intensity"]], on=date_col, how="left")

    feat_train = feat_all[feat_all[date_col].isin(train_df[date_col])]
    feat_test = feat_all[feat_all[date_col].isin(test_df[date_col])]

    X_cols = [c for c in feat_train.columns if c not in [date_col, target_col]]
    X_train = feat_train[X_cols].fillna(0).values
    X_test = feat_test[X_cols].fillna(0).values
    y_train = feat_train[target_col].astype(float).values
    y_test = feat_test[target_col].astype(float).values

    artifact: Dict[str, Any] = {
        "schema_version": "1.2.1",
        "model_name": None,
        "scores": {},
        "holdout_dates": [d.isoformat() for d in test_dates],
        "holdout_y_actual": y_test_actual.tolist(),
        "holdout_preds": {},
        "holdout_residuals": {},
        "feature_columns": X_cols,
        "cutoff_date": str(train_df[date_col].max().date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "holdout_months": holdout_months,
        "blend_weight_final": None,
        "xgboost_error": None,
        "xgb_model_raw": None,
        "date_column": date_col,
        "target_column": target_col,
        "use_log_target": use_log,
        "use_events": use_events
    }

    # -------------- Baseline: Naive --------------
    naive_pred = predict_naive_last(
        train_df.rename(columns={fit_col: target_col}),
        test_dates, date_col, fit_col if use_log else target_col
    )
    if use_log:
        naive_pred = np.expm1(naive_pred)
    artifact["scores"]["naive"] = {
        "MAPE": mape(y_test_actual, naive_pred),
        "RMSE": rmse(y_test_actual, naive_pred),
        "sMAPE": _smape(y_test_actual, naive_pred)
    }
    artifact["holdout_preds"]["naive"] = {"y_pred": naive_pred.tolist()}
    artifact["holdout_residuals"]["naive"] = (y_test_actual - naive_pred).tolist()

    # -------------- Baseline: Seasonal Naive --------------
    if len(df) > season_length + holdout_months:
        sn_pred = predict_seasonal_naive(
            train_df.rename(columns={fit_col: target_col}),
            test_dates, date_col, fit_col if use_log else target_col, season_length
        )
        if use_log:
            sn_pred = np.expm1(sn_pred)
        artifact["scores"]["seasonal_naive"] = {
            "MAPE": mape(y_test_actual, sn_pred),
            "RMSE": rmse(y_test_actual, sn_pred),
            "sMAPE": _smape(y_test_actual, sn_pred)
        }
        artifact["holdout_preds"]["seasonal_naive"] = {"y_pred": sn_pred.tolist()}
        artifact["holdout_residuals"]["seasonal_naive"] = (y_test_actual - sn_pred).tolist()

    # -------------- XGBoost --------------
    xgb_pred = None
    try:
        import xgboost as xgb  # pastikan xgboost ada di requirements
        xgb_cfg = train_cfg.get("xgb", {})
        params = {
            "objective": "reg:squarederror",
            "eta": xgb_cfg.get("eta", 0.05),
            "max_depth": xgb_cfg.get("max_depth", 4),
            "subsample": xgb_cfg.get("subsample", 0.9),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.9),
            "seed": 42
        }

        fold_indices = _expanding_cv_indices(
            n=len(feat_train),
            n_folds=cv_folds,
            min_train=min_train_months,
            horizon=1
        )
        cv_mapes: List[float] = []
        for tr_idx, va_idx in fold_indices:
            dtr = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx], feature_names=X_cols)
            dva = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx], feature_names=X_cols)
            booster_cv = xgb.train(
                params,
                dtr,
                num_boost_round=xgb_cfg.get("n_rounds", 600),
                evals=[(dtr, "train"), (dva, "valid")],
                early_stopping_rounds=train_cfg.get("early_stopping_rounds", 40),
                verbose_eval=False
            )
            pred_cv = booster_cv.predict(dva)
            if use_log:
                pred_cv = np.expm1(pred_cv)
                y_val_actual = np.expm1(y_train[va_idx]) if fit_col.endswith("_log") else y_train[va_idx]
            else:
                y_val_actual = y_train[va_idx]
            cv_mapes.append(mape(y_val_actual, pred_cv))

        # Train final
        dtrain_full = xgb.DMatrix(X_train, label=y_train, feature_names=X_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_cols)
        booster = xgb.train(
            params,
            dtrain_full,
            num_boost_round=xgb_cfg.get("n_rounds", 600),
            evals=[(dtrain_full, "train")],
            verbose_eval=False
        )
        raw_pred = booster.predict(dtest)
        xgb_pred = np.expm1(raw_pred) if use_log else raw_pred

        artifact["scores"]["xgboost"] = {
            "MAPE": mape(y_test_actual, xgb_pred),
            "RMSE": rmse(y_test_actual, xgb_pred),
            "sMAPE": _smape(y_test_actual, xgb_pred),
            "CV_MAPE_mean": float(np.mean(cv_mapes)) if cv_mapes else None
        }
        artifact["holdout_preds"]["xgboost"] = {"y_pred": xgb_pred.tolist()}
        artifact["holdout_residuals"]["xgboost"] = (y_test_actual - xgb_pred).tolist()
        artifact["xgb_model_raw"] = booster.save_raw().decode("latin1", errors="ignore")

    except Exception as e:
        warnings.warn(f"XGBoost gagal: {e}")
        artifact["xgboost_error"] = f"XGBoost gagal: {e}"

    # -------------- Blend --------------
    if train_cfg.get("blend_enable", True):
        components = [m for m in ["xgboost", "seasonal_naive", "naive"] if m in artifact["holdout_preds"]]
        if len(components) >= 2:
            preds_dict = {
                m: np.array(artifact["holdout_preds"][m]["y_pred"], dtype=float)
                for m in components
            }
            best = (None, 1e18, None)
            # Grid 0..1 step 0.1 untuk maksimal 3 model
            if len(components) == 2:
                A, B = components
                for w in np.linspace(0, 1, 21):
                    blend = w * preds_dict[A] + (1 - w) * preds_dict[B]
                    mm = mape(y_test_actual, blend)
                    if mm < best[1]:
                        best = ({"w_"+A: w, "w_"+B: 1-w}, mm, blend)
            else:
                A, B, C = components
                for w1 in np.linspace(0, 1, 11):
                    for w2 in np.linspace(0, 1 - w1, 11):
                        w3 = 1 - w1 - w2
                        blend = (
                            w1 * preds_dict[A] +
                            w2 * preds_dict[B] +
                            w3 * preds_dict[C]
                        )
                        mm = mape(y_test_actual, blend)
                        if mm < best[1]:
                            best = ({"w_"+A: w1, "w_"+B: w2, "w_"+C: w3}, mm, blend)
            if best[0]:
                artifact["scores"]["blend"] = {
                    "MAPE": best[1],
                    "RMSE": rmse(y_test_actual, best[2]),
                    "sMAPE": _smape(y_test_actual, best[2]),
                    "weights": best[0]
                }
                artifact["holdout_preds"]["blend"] = {"y_pred": best[2].tolist()}
                artifact["holdout_residuals"]["blend"] = (y_test_actual - best[2]).tolist()
                artifact["blend_weight_final"] = best[0]

    # -------------- Model Selection --------------
    ranking = []
    for m_name, sc in artifact["scores"].items():
        val = sc.get("MAPE")
        if val is not None and not np.isnan(val):
            ranking.append((m_name, val))
    if ranking:
        ranking.sort(key=lambda x: x[1])
        artifact["model_name"] = ranking[0][0]
    else:
        artifact["model_name"] = "naive"

    return artifact
