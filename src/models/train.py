from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# =====================================================
# DEFAULT CONFIG
# =====================================================
_DEFAULT_BASE_CONFIG: Dict[str, Any] = {
    "data": {
        "date_column": "periode",
        "target_column": "permohonan_kia",
        "lags": [1, 2, 3, 6, 12],
        "rollings": [3, 6, 12],
        "add_sin_cos": True,
        "use_log_target": False,
        "use_events": False,
        "cap_outliers": False,
        "outlier_method": "iqr",
        "outlier_iqr_k": 1.5,
        "event_file": "data/raw/events.csv",
    },
    "training": {
        "holdout_months": 6,
        "season_length": 12,
        "cv_folds": 3,
        "min_train_months": 18,
        "blend_enable": True,
        "early_stopping_rounds": 40,
        "xgb": {
            "eta": 0.05,
            "max_depth": 4,
            "n_rounds": 600,
            "subsample": 0.9,
            "colsample_bytree": 0.9
        }
    },
    "forecast": {
        "horizon": 6,
        "max_horizon": 24
    }
}

# =====================================================
# TOML loader (aman Python 3.11+ atau fallback)
# =====================================================
try:
    import tomllib  # type: ignore

    def _read_toml(path: Path) -> Dict[str, Any]:
        with path.open("rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:
    import toml  # type: ignore

    def _read_toml(path: Path) -> Dict[str, Any]:
        return toml.load(path.open("r", encoding="utf-8"))

# =====================================================
# Event utils (opsional)
# =====================================================
try:
    from src.utils.events import load_events, merge_event_features
    _HAS_EVENT_UTILS = True
except Exception:
    _HAS_EVENT_UTILS = False

__all__ = [
    "load_config",
    "train_pipeline",
    "save_artifact",
    "load_artifact"
]

# =====================================================
# CONFIG
# =====================================================
def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in base.items():
        if k in override:
            if isinstance(v, dict) and isinstance(override[k], dict):
                out[k] = _merge_dicts(v, override[k])
            else:
                out[k] = override[k]
        else:
            out[k] = v
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out

def load_config(path: str | Path = "config.toml") -> dict:
    p = Path(path)
    if not p.exists():
        return _DEFAULT_BASE_CONFIG.copy()
    try:
        cfg = _read_toml(p)
        return _merge_dicts(_DEFAULT_BASE_CONFIG, cfg)
    except Exception as e:
        warnings.warn(f"Gagal membaca config.toml: {e}. Menggunakan default.")
        return _DEFAULT_BASE_CONFIG.copy()

# =====================================================
# METRICS
# =====================================================
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

# =====================================================
# BASELINE (array based, hindari masalah DataFrame duplikat kolom)
# =====================================================
def naive_recursive_last(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    """
    Multi-step one-by-one naive:
    Prediksi bulan t = nilai aktual bulan t-1 (actual already known in sequential evaluation).
    """
    preds = []
    hist = list(train_values)
    for actual in test_values:
        if len(hist) == 0:
            preds.append(np.nan)
        else:
            preds.append(hist[-1])
        # Append actual (karena evaluasi 1-step ahead bergulir)
        hist.append(actual)
    return np.array(preds, dtype=float)

def seasonal_recursive_last(train_values: np.ndarray,
                            test_values: np.ndarray,
                            season_length: int) -> np.ndarray:
    """
    Seasonal naive iterative:
    Prediksi t = nilai aktual t - season_length (dari hist yang ter-update).
    """
    preds = []
    hist = list(train_values)
    for actual in test_values:
        if len(hist) >= season_length:
            preds.append(hist[-season_length])
        elif len(hist) > 0:
            preds.append(hist[-1])
        else:
            preds.append(np.nan)
        hist.append(actual)
    return np.array(preds, dtype=float)

# =====================================================
# OUTLIER
# =====================================================
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

# =====================================================
# EXPANDING CV
# =====================================================
def _expanding_cv_indices(n: int, n_folds: int, min_train: int, horizon: int = 1):
    folds = []
    start_test = min_train
    while start_test + horizon <= n and len(folds) < n_folds:
        train_idx = np.arange(0, start_test)
        test_idx = np.arange(start_test, start_test + horizon)
        folds.append((train_idx, test_idx))
        start_test += horizon
    return folds

# =====================================================
# FEATURE ENGINEERING
# =====================================================
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

# =====================================================
# ARTIFACT I/O
# =====================================================
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

# =====================================================
# TRAIN PIPELINE
# =====================================================
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

    # Validasi kolom
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Kolom {date_col}/{target_col} tidak ditemukan di data.")

    # Tipe & NaN
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[target_col].isna().any():
        raise ValueError("Ada nilai target NaN – bersihkan dulu.")

    # Urutkan
    df = df.sort_values(date_col).reset_index(drop=True)

    if len(df) <= holdout_months + 6:
        raise ValueError("Data terlalu sedikit untuk konfigurasi holdout saat ini.")

    # Outlier capping
    if cap_outliers:
        df["_target_capped"] = _cap_outliers(df[target_col], method=outlier_method, k=outlier_iqr_k)
    else:
        df["_target_capped"] = df[target_col]

    # Event merge
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

    # Split
    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()
    test_dates = list(test_df[date_col])
    y_test_actual = test_df[target_col].astype(float).values

    # ================= BASELINES =================
    full_values = df[target_col].values.astype(float)
    train_values = full_values[:-holdout_months]
    test_values = full_values[-holdout_months:]

    # Naive recursive one-step
    naive_pred = naive_recursive_last(train_values, test_values)
    artifact_scores: Dict[str, Any] = {}
    holdout_preds: Dict[str, Any] = {}
    holdout_residuals: Dict[str, Any] = {}

    artifact_scores["naive"] = {
        "MAPE": mape(y_test_actual, naive_pred),
        "RMSE": rmse(y_test_actual, naive_pred),
        "sMAPE": _smape(y_test_actual, naive_pred)
    }
    holdout_preds["naive"] = {"y_pred": naive_pred.tolist()}
    holdout_residuals["naive"] = (y_test_actual - naive_pred).tolist()

    # Seasonal naive (jika cukup panjang)
    if len(train_values) >= season_length:
        seasonal_pred = seasonal_recursive_last(train_values, test_values, season_length)
        artifact_scores["seasonal_naive"] = {
            "MAPE": mape(y_test_actual, seasonal_pred),
            "RMSE": rmse(y_test_actual, seasonal_pred),
            "sMAPE": _smape(y_test_actual, seasonal_pred)
        }
        holdout_preds["seasonal_naive"] = {"y_pred": seasonal_pred.tolist()}
        holdout_residuals["seasonal_naive"] = (y_test_actual - seasonal_pred).tolist()

    # ================= FITUR untuk MODEL ML =================
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
    y_train_fit = feat_train[target_col].astype(float).values  # bisa log jika use_log
    y_test_fit = feat_test[target_col].astype(float).values

    # ================= XGBoost =================
    xgb_pred_actual_scale = None
    xgb_error = None
    try:
        import xgboost as xgb  # pastikan di requirements
        xgb_cfg = train_cfg.get("xgb", {})
        params = {
            "objective": "reg:squarederror",
            "eta": xgb_cfg.get("eta", 0.05),
            "max_depth": xgb_cfg.get("max_depth", 4),
            "subsample": xgb_cfg.get("subsample", 0.9),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.9),
            "seed": 42
        }

        # Expanding CV dalam train portion
        fold_indices = _expanding_cv_indices(
            n=len(feat_train),
            n_folds=train_cfg.get("cv_folds", 3),
            min_train=train_cfg.get("min_train_months", 18),
            horizon=1
        )
        cv_mapes: List[float] = []
        for tr_idx, va_idx in fold_indices:
            dtr = xgb.DMatrix(X_train[tr_idx], label=y_train_fit[tr_idx], feature_names=X_cols)
            dva = xgb.DMatrix(X_train[va_idx], label=y_train_fit[va_idx], feature_names=X_cols)
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
                pred_cv_actual = np.expm1(pred_cv)
                y_val_actual = np.expm1(y_train_fit[va_idx])
            else:
                pred_cv_actual = pred_cv
                y_val_actual = y_train_fit[va_idx]
            cv_mapes.append(mape(y_val_actual, pred_cv_actual))

        dtrain_full = xgb.DMatrix(X_train, label=y_train_fit, feature_names=X_cols)
        dtest = xgb.DMatrix(X_test, label=y_test_fit, feature_names=X_cols)
        booster = xgb.train(
            params,
            dtrain_full,
            num_boost_round=xgb_cfg.get("n_rounds", 600),
            evals=[(dtrain_full, "train")],
            verbose_eval=False
        )
        raw_pred = booster.predict(dtest)
        xgb_pred_actual_scale = np.expm1(raw_pred) if use_log else raw_pred

        artifact_scores["xgboost"] = {
            "MAPE": mape(y_test_actual, xgb_pred_actual_scale),
            "RMSE": rmse(y_test_actual, xgb_pred_actual_scale),
            "sMAPE": _smape(y_test_actual, xgb_pred_actual_scale),
            "CV_MAPE_mean": float(np.mean(cv_mapes)) if cv_mapes else None
        }
        holdout_preds["xgboost"] = {"y_pred": xgb_pred_actual_scale.tolist()}
        holdout_residuals["xgboost"] = (y_test_actual - xgb_pred_actual_scale).tolist()
        xgb_model_raw = booster.save_raw().decode("latin1", errors="ignore")
    except Exception as e:
        xgb_error = f"XGBoost gagal: {e}"
        warnings.warn(xgb_error)
        xgb_model_raw = None

    # ================= BLEND =================
    if train_cfg.get("blend_enable", True):
        components = [m for m in ["xgboost", "seasonal_naive", "naive"] if m in holdout_preds]
        if len(components) >= 2:
            preds_dict = {m: np.array(holdout_preds[m]["y_pred"], dtype=float) for m in components}
            best = (None, 1e18, None)
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
                        if w3 < -1e-9:
                            continue
                        blend = w1 * preds_dict[A] + w2 * preds_dict[B] + w3 * preds_dict[C]
                        mm = mape(y_test_actual, blend)
                        if mm < best[1]:
                            best = ({"w_"+A: w1, "w_"+B: w2, "w_"+C: w3}, mm, blend)
            if best[0]:
                artifact_scores["blend"] = {
                    "MAPE": best[1],
                    "RMSE": rmse(y_test_actual, best[2]),
                    "sMAPE": _smape(y_test_actual, best[2]),
                    "weights": best[0]
                }
                holdout_preds["blend"] = {"y_pred": best[2].tolist()}
                holdout_residuals["blend"] = (y_test_actual - best[2]).tolist()

    # ================= Seleksi model =================
    ranking = []
    for m_name, sc in artifact_scores.items():
        val = sc.get("MAPE")
        if val is not None and not np.isnan(val):
            ranking.append((m_name, val))
    ranking.sort(key=lambda x: x[1])
    best_model = ranking[0][0] if ranking else "naive"

    artifact: Dict[str, Any] = {
        "schema_version": "1.3.0",
        "model_name": best_model,
        "scores": artifact_scores,
        "holdout_dates": [d.isoformat() for d in test_dates],
        "holdout_y_actual": y_test_actual.tolist(),
        "holdout_preds": holdout_preds,
        "holdout_residuals": holdout_residuals,
        "feature_columns": X_cols,
        "cutoff_date": str(train_df[date_col].max().date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "holdout_months": holdout_months,
        "blend_weight_final": artifact_scores.get("blend", {}).get("weights"),
        "xgboost_error": xgb_error,
        "xgb_model_raw": xgb_model_raw,
        "date_column": date_col,
        "target_column": target_col,
        "use_log_target": use_log,
        "use_events": use_events
    }

    return artifact
