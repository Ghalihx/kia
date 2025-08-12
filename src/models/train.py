from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import toml


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.toml"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config.toml tidak ditemukan di {p}")
    return toml.load(p.open("r", encoding="utf-8"))


def save_artifact(artifact: dict,
                  out_dir: str = "models",
                  filename_prefix: str = "kia_forecast") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_ts = out_path / f"{filename_prefix}_{ts}.pkl"
    with fname_ts.open("wb") as f:
        pickle.dump(artifact, f)
    # Simpan symlink / latest copy
    with (out_path / f"{filename_prefix}_latest.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    return fname_ts


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

    feat["month"] = feat[date_col].dt.month
    feat["year"] = feat[date_col].dt.year
    feat["t"] = np.arange(len(feat))

    if add_sin_cos:
        feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
        feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # Fitur tambahan sederhana (optional—bisa dipakai XGBoost)
    feat["diff_1"] = feat[target_col].diff(1)
    feat["diff_12"] = feat[target_col].diff(12)

    return feat


def predict_naive_last(train_df: pd.DataFrame,
                       test_dates: List[pd.Timestamp],
                       date_col: str,
                       target_col: str) -> np.ndarray:
    preds = []
    last_fallback = float(train_df[target_col].iloc[-1])
    for d in test_dates:
        prev = d - pd.DateOffset(months=1)
        val = train_df.loc[train_df[date_col] == prev, target_col]
        preds.append(float(val.values[0]) if not val.empty else last_fallback)
    return np.array(preds, dtype=float)


def predict_seasonal_naive(train_df: pd.DataFrame,
                           test_dates: List[pd.Timestamp],
                           date_col: str,
                           target_col: str,
                           season_length: int = 12) -> np.ndarray:
    preds = []
    last_fallback = float(train_df[target_col].iloc[-1])
    for d in test_dates:
        prev_season = d - pd.DateOffset(months=season_length)
        val = train_df.loc[train_df[date_col] == prev_season, target_col]
        if val.empty:
            prev1 = d - pd.DateOffset(months=1)
            val = train_df.loc[train_df[date_col] == prev1, target_col]
        preds.append(float(val.values[0]) if not val.empty else last_fallback)
    return np.array(preds, dtype=float)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (y_true != 0)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))


def _optimize_blend(y_true: np.ndarray,
                    xgb_pred: np.ndarray,
                    seas_pred: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """
    Grid search bobot w untuk kombinasi:
    blend = w * xgb + (1-w) * seasonal
    """
    best = (0.5, 1e12, 1e12, None)  # w, mape, rmse, preds
    for w in np.linspace(0, 1, 11):  # 0.0 .. 1.0 step 0.1
        bp = w * xgb_pred + (1 - w) * seas_pred
        mm = mape(y_true, bp)
        rr = rmse(y_true, bp)
        if mm < best[1]:
            best = (w, mm, rr, bp)
    return best


def train_pipeline(df: pd.DataFrame, cfg: dict) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    date_col = data_cfg.get("date_column", "periode")
    target_col = data_cfg.get("target_column", "permohonan_kia")
    lags = data_cfg.get("lags", [1, 2, 3, 12])
    rollings = data_cfg.get("rollings", [3, 6, 12])
    add_sin_cos = bool(data_cfg.get("add_sin_cos", True))

    holdout_months = int(train_cfg.get("holdout_months", 6))
    season_length = int(train_cfg.get("season_length", 12))

    # Pastikan numeric
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])

    if df[target_col].isna().any():
        raise ValueError("Ada nilai target yang tidak numerik / NaN. Bersihkan data.")

    df = df.sort_values(date_col).reset_index(drop=True)

    if len(df) <= holdout_months + 6:
        raise ValueError("Data terlalu sedikit untuk holdout. Tambahkan data.")

    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()
    test_dates = list(test_df[date_col])
    y_test_actual = test_df[target_col].values.astype(float)

    feat_all = build_features(df, date_col, target_col, lags, rollings, add_sin_cos)
    feat_train = feat_all[feat_all[date_col].isin(train_df[date_col])]
    feat_test = feat_all[feat_all[date_col].isin(test_df[date_col])]

    X_cols = [c for c in feat_train.columns if c not in [date_col, target_col]]
    X_train = feat_train[X_cols].copy().fillna(0)
    X_test = feat_test[X_cols].copy().fillna(0)
    y_train = feat_train[target_col].values.astype(float)
    y_test = feat_test[target_col].values.astype(float)

    artifact: Dict[str, Any] = {
        "schema_version": "1.1.0",
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
        "target_column": target_col
    }

    # Naive
    naive_pred = predict_naive_last(train_df, test_dates, date_col, target_col)
    artifact["scores"]["naive"] = {
        "MAPE": mape(y_test_actual, naive_pred),
        "RMSE": rmse(y_test_actual, naive_pred)
    }
    artifact["holdout_preds"]["naive"] = {"y_pred": naive_pred.tolist()}
    artifact["holdout_residuals"]["naive"] = (y_test_actual - naive_pred).tolist()

    # Seasonal naive
    if len(df) > season_length + holdout_months:
        sn_pred = predict_seasonal_naive(train_df, test_dates, date_col, target_col, season_length)
        artifact["scores"]["seasonal_naive"] = {
            "MAPE": mape(y_test_actual, sn_pred),
            "RMSE": rmse(y_test_actual, sn_pred)
        }
        artifact["holdout_preds"]["seasonal_naive"] = {"y_pred": sn_pred.tolist()}
        artifact["holdout_residuals"]["seasonal_naive"] = (y_test_actual - sn_pred).tolist()

    # XGBoost
    xgb_pred = None
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=X_cols)
        params = {
            "objective": "reg:squarederror",
            "eta": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "seed": 42
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=600,
            evals=[(dtrain, "train"), (dtest, "valid")],
            early_stopping_rounds=40,
            verbose_eval=False
        )
        xgb_pred = booster.predict(dtest)
        artifact["scores"]["xgboost"] = {
            "MAPE": mape(y_test_actual, xgb_pred),
            "RMSE": rmse(y_test_actual, xgb_pred)
        }
        artifact["holdout_preds"]["xgboost"] = {"y_pred": xgb_pred.tolist()}
        artifact["holdout_residuals"]["xgboost"] = (y_test_actual - xgb_pred).tolist()
        artifact["xgb_model_raw"] = booster.save_raw().decode("latin1", errors="ignore")
    except Exception as e:
        artifact["xgboost_error"] = f"XGBoost gagal: {e}"

    # Blend (optimasi w) hanya jika kedua model ada
    if "xgboost" in artifact["scores"] and "seasonal_naive" in artifact["scores"]:
        xp = np.array(artifact["holdout_preds"]["xgboost"]["y_pred"], dtype=float)
        sp = np.array(artifact["holdout_preds"]["seasonal_naive"]["y_pred"], dtype=float)
        w_opt, mape_opt, rmse_opt, bp = _optimize_blend(y_test_actual, xp, sp)
        artifact["scores"]["blend"] = {
            "MAPE": mape_opt,
            "RMSE": rmse_opt
        }
        artifact["holdout_preds"]["blend"] = {"y_pred": bp.tolist()}
        artifact["holdout_residuals"]["blend"] = (y_test_actual - bp).tolist()
        artifact["blend_weight_final"] = float(w_opt)

    # Pilih model dengan MAPE terendah
    if artifact["scores"]:
        valid = [(m, sc["MAPE"]) for m, sc in artifact["scores"].items()
                 if sc["MAPE"] is not None and not np.isnan(sc["MAPE"])]
        artifact["model_name"] = min(valid, key=lambda x: x[1])[0] if valid else "naive"
    else:
        artifact["model_name"] = "naive"

    # (Opsional) jika naive menang tipis (<0.3%) dibanding seasonal_naive → pilih seasonal_naive (agar tidak flat)
    if artifact["model_name"] == "naive" and "seasonal_naive" in artifact["scores"]:
        diff = artifact["scores"]["seasonal_naive"]["MAPE"] - artifact["scores"]["naive"]["MAPE"]
        if diff < 0.3:
            artifact["model_name"] = "seasonal_naive"

    return artifact
