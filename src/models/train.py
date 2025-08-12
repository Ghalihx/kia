from __future__ import annotations

import pickle
from pathlib import Path
from datetime import datetime
import hashlib
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
import toml

# ====== UTIL DASAR (MAPE / RMSE)—pastikan sama seperti definisi di versi kamu ======
def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# ====== LOAD / SAVE CONFIG & ARTIFACT ======
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
    with (out_path / f"{filename_prefix}_latest.pkl").open("wb") as f:
        pickle.dump(artifact, f)
    return fname_ts

# ====== FEATURE ENGINEERING ======
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
    feat["diff_1"] = feat[target_col].diff(1)
    feat["diff_12"] = feat[target_col].diff(12)
    return feat

# ====== BASELINE PREDIKTOR ======
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
        preds.append(float(val.values[0]) if not val.empty else last_fallback)
    return np.array(preds, dtype=float)

# ====== HASH DATA (untuk cek perubahan dataset) ======
def hash_data(df: pd.DataFrame, date_col: str, target_col: str) -> str:
    h = hashlib.md5()
    h.update(df[date_col].astype(str).str.cat(sep="|").encode())
    h.update(b"::")
    h.update(df[target_col].astype(str).str.cat(sep="|").encode())
    return h.hexdigest()

# ====== PIPELINE TRAIN ======
def train_pipeline(df: pd.DataFrame,
                   cfg: dict,
                   fast_mode: bool = False,
                   progress_callback: Optional[Callable[[str, dict], None]] = None) -> Dict[str, Any]:
    def cb(event: str, info: dict | None = None):
        if progress_callback:
            try:
                progress_callback(event, info or {})
            except Exception:
                pass

    cb("phase", {"label": "Inisialisasi pipeline"})

    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    date_col = data_cfg.get("date_column", "periode")
    target_col = data_cfg.get("target_column", "permohonan_kia")
    lags = data_cfg.get("lags", [1, 2, 3, 12])
    rollings = data_cfg.get("rollings", [3, 6, 12])
    add_sin_cos = bool(data_cfg.get("add_sin_cos", True))

    holdout_months = int(train_cfg.get("holdout_months", 6))
    season_length = int(train_cfg.get("season_length", 12)) if "season_length" in train_cfg else 12
    min_rows_xgb = int(train_cfg.get("min_rows_xgb", 36))

    # Fast mode: potong fitur
    if fast_mode:
        lags = [l for l in lags if l in (1, 12)]
        if 1 not in lags:
            lags.insert(0, 1)
        rollings = []  # skip rolling

    # Pastikan tipe data
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[target_col].isna().any():
        raise ValueError("Ada nilai target tidak numerik / NaN, bersihkan dulu.")
    df = df.sort_values(date_col).reset_index(drop=True)

    total_rows = len(df)
    if total_rows <= holdout_months + 6:
        raise ValueError("Data terlalu sedikit untuk holdout. Tambah data atau kecilkan holdout_months.")

    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()
    test_dates = list(test_df[date_col])
    y_test_actual = test_df[target_col].values.astype(float)

    cb("phase", {"label": "Bangun fitur"})
    feat_all = build_features(df, date_col, target_col, lags, rollings, add_sin_cos)

    # Drop fitur rolling dengan valid ratio rendah
    to_drop = []
    for c in feat_all.columns:
        if c.startswith("roll_"):
            if feat_all[c].notna().mean() < 0.4:
                to_drop.append(c)
    if to_drop:
        feat_all = feat_all.drop(columns=to_drop)

    feat_train = feat_all[feat_all[date_col].isin(train_df[date_col])]
    feat_test = feat_all[feat_all[date_col].isin(test_df[date_col])]
    X_cols = [c for c in feat_train.columns if c not in [date_col, target_col]]
    X_train = feat_train[X_cols].fillna(0).to_numpy()
    X_test = feat_test[X_cols].fillna(0).to_numpy()
    y_train = feat_train[target_col].values.astype(float)
    y_test = feat_test[target_col].values.astype(float)

    artifact: Dict[str, Any] = {
        "schema_version": "1.2.0",
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
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
        "data_hash": hash_data(df, date_col, target_col),
        "fast_mode": fast_mode
    }

    # Baseline naive
    cb("phase", {"label": "Baseline naive"})
    naive_pred = predict_naive_last(train_df, test_dates, date_col, target_col)
    artifact["scores"]["naive"] = {
        "MAPE": mape(y_test_actual, naive_pred),
        "RMSE": rmse(y_test_actual, naive_pred)
    }
    artifact["holdout_preds"]["naive"] = {"y_pred": naive_pred.tolist()}
    artifact["holdout_residuals"]["naive"] = (y_test_actual - naive_pred).tolist()

    # Baseline seasonal naive
    cb("phase", {"label": "Baseline seasonal"})
    seasonal_pred = predict_seasonal_naive(train_df, test_dates, date_col, target_col, season_length=season_length)
    artifact["scores"]["seasonal_naive"] = {
        "MAPE": mape(y_test_actual, seasonal_pred),
        "RMSE": rmse(y_test_actual, seasonal_pred)
    }
    artifact["holdout_preds"]["seasonal_naive"] = {"y_pred": seasonal_pred.tolist()}
    artifact["holdout_residuals"]["seasonal_naive"] = (y_test_actual - seasonal_pred).tolist()

    # XGBoost (opsional)
    use_xgb = (not fast_mode) and (len(train_df) >= min_rows_xgb)
    xgb_pred = None
    if use_xgb:
        cb("phase", {"label": "Training XGBoost"})
        try:
            import xgboost as xgb
            params = cfg.get("model", {}).get("xgb_params", {}).copy()
            params.setdefault("tree_method", "hist")
            params.setdefault("objective", "reg:squarederror")
            params.setdefault("n_estimators", 300)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 4)
            params.setdefault("subsample", 0.9)
            params.setdefault("colsample_bytree", 0.9)

            early_rounds = int(train_cfg.get("xgb_early_stopping_rounds", 30))
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                early_stopping_rounds=early_rounds
            )
            xgb_pred = model.predict(X_test)
            artifact["scores"]["xgboost"] = {
                "MAPE": mape(y_test_actual, xgb_pred),
                "RMSE": rmse(y_test_actual, xgb_pred)
            }
            artifact["holdout_preds"]["xgboost"] = {"y_pred": xgb_pred.tolist()}
            artifact["holdout_residuals"]["xgboost"] = (y_test_actual - xgb_pred).tolist()
            artifact["xgb_model_raw"] = model.get_booster().save_raw("json").decode()
        except Exception as e:
            artifact["xgboost_error"] = str(e)
            use_xgb = False

    # Blend (jika ada xgb)
    if use_xgb and xgb_pred is not None:
        cb("phase", {"label": "Blend"})
        y1 = seasonal_pred
        y2 = xgb_pred
        best_w = 0.5
        best_mape = 1e9
        for w in np.linspace(0, 1, 21):
            blend_tmp = w * y2 + (1 - w) * y1
            sc = mape(y_test_actual, blend_tmp)
            if sc < best_mape:
                best_mape = sc
                best_w = w
        blend_pred = best_w * y2 + (1 - best_w) * y1
        artifact["scores"]["blend"] = {
            "MAPE": mape(y_test_actual, blend_pred),
            "RMSE": rmse(y_test_actual, blend_pred)
        }
        artifact["holdout_preds"]["blend"] = {"y_pred": blend_pred.tolist()}
        artifact["holdout_residuals"]["blend"] = (y_test_actual - blend_pred).tolist()
        artifact["blend_weight_final"] = best_w

        # Pilih model final: MAPE terkecil dari seasonal_naive, xgboost, blend
        ranking = []
        for m in ["seasonal_naive", "xgboost", "blend"]:
            if m in artifact["scores"]:
                ranking.append((artifact["scores"][m]["MAPE"], m))
        ranking.sort()
        artifact["model_name"] = ranking[0][1]
    else:
        # Pilih baseline terbaik
        ranking = [
            (artifact["scores"]["naive"]["MAPE"], "naive"),
            (artifact["scores"]["seasonal_naive"]["MAPE"], "seasonal_naive")
        ]
        ranking.sort()
        artifact["model_name"] = ranking[0][1]

    cb("done", {})
    return artifact
