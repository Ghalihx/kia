from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import yaml
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.pipeline.data_prep import load_and_validate, make_supervised
from src.models.evaluate import metrics_all
from src.models.baselines import (
    naive_forecast,
    seasonal_naive_forecast,
    blend_forecast,
)
from src.models.infer import forecast_iterative_xgb


def load_config(path: str = "config/config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_metrics(full: Dict[str, float]) -> Dict[str, float]:
    return {
        "RMSE": full.get("RMSE"),
        "MAPE": full.get("MAPE"),
    }


def _optimal_blend_weight_rmse(y_true: np.ndarray, y_naive: np.ndarray, y_seasonal: np.ndarray) -> float:
    diff = (y_naive - y_seasonal)
    denom = np.sum(diff * diff)
    if denom == 0:
        return 0.5
    numer = np.sum((y_true - y_seasonal) * diff)
    w = numer / denom
    return float(min(1.0, max(0.0, w)))


def _evaluate_baselines_with_preds(
    hist_df: pd.DataFrame,
    cfg: Dict,
    horizon: int,
) -> Tuple[Dict[str, Dict], Dict[str, np.ndarray], np.ndarray, List[pd.Timestamp], Optional[float]]:
    y = hist_df[cfg["data"]["target_column"]].values
    dcol = cfg["data"]["date_column"]
    n = y.size
    y_train = y[: n - horizon]
    y_test = y[n - horizon :]
    dates = hist_df[dcol].iloc[n - horizon :].tolist()

    preds_dict: Dict[str, np.ndarray] = {}
    preds_dict["naive"] = naive_forecast(y_train, horizon=horizon)
    preds_dict["seasonal_naive"] = seasonal_naive_forecast(y_train, horizon=horizon, season_length=12)

    blend_weight_used = None
    training_cfg = cfg.get("training", {})
    if training_cfg.get("use_blend", False):
        bw = training_cfg.get("blend_weight", 0.5)
        if isinstance(bw, str) and bw.lower() == "auto":
            w_opt = _optimal_blend_weight_rmse(y_test, preds_dict["naive"], preds_dict["seasonal_naive"])
            blend_weight_used = w_opt
            preds_dict["blend"] = blend_forecast(y_train, horizon=horizon, w=w_opt, season_length=12)
        else:
            w_fix = float(bw)
            blend_weight_used = w_fix
            preds_dict["blend"] = blend_forecast(y_train, horizon=horizon, w=w_fix, season_length=12)

    scores: Dict[str, Dict] = {}
    for name, yhat in preds_dict.items():
        base_full = metrics_all(y_test, yhat)
        scores[name] = _strip_metrics(base_full)

    return scores, preds_dict, y_test, dates, blend_weight_used


def _fit_xgb(
    sup_df: pd.DataFrame,
    feature_names: List[str],
    cutoff_date: pd.Timestamp,
    cfg: Dict,
) -> XGBRegressor:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    train_mask = sup_df[dcol] <= cutoff_date
    X_train = sup_df.loc[train_mask, feature_names]
    y_train = sup_df.loc[train_mask, ycol]

    params = cfg["model"]["xgb_params"]
    es_rounds = int(cfg["training"].get("xgb_early_stopping_rounds", 0))

    model = XGBRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 4),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        random_state=params.get("random_state", 42),
        tree_method=params.get("tree_method", "auto"),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        n_jobs=-1,
        objective=params.get("objective", "reg:squarederror"),
    )

    if es_rounds > 0 and X_train.shape[0] > 20:
        vsize = max(6, int(0.1 * X_train.shape[0]))
        X_tr = X_train.iloc[:-vsize]
        y_tr = y_train.iloc[:-vsize]
        X_val = X_train.iloc[-vsize:]
        y_val = y_train.iloc[-vsize:]
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=es_rounds,
        )
    else:
        model.fit(X_train, y_train)
    return model


def train_pipeline(df: pd.DataFrame, cfg: Dict) -> Dict:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]
    holdout = int(cfg["training"]["holdout_months"])
    min_rows = int(cfg["training"]["min_train_rows"])

    df = load_and_validate(df, cfg)
    hist = df[[dcol, ycol]].copy()
    n = len(hist)
    if n <= holdout + 1:
        raise ValueError("Data terlalu pendek untuk evaluasi holdout.")

    baseline_scores, baseline_preds, y_test, holdout_dates, blend_weight_used = _evaluate_baselines_with_preds(
        hist, cfg, horizon=holdout
    )

    sup, feature_names = make_supervised(df, cfg)
    test_first_date = hist[dcol].iloc[n - holdout]
    cutoff_date = test_first_date - pd.offsets.MonthBegin(1)

    xgb_scores: Optional[Dict] = None
    xgb_preds: Optional[np.ndarray] = None
    model = None
    xgb_error = None
    can_train_xgb = HAS_XGB and (sup[sup[dcol] <= cutoff_date].shape[0] >= min_rows)

    if can_train_xgb:
        try:
            model = _fit_xgb(sup, feature_names, cutoff_date, cfg)
            hist_train = hist[hist[dcol] <= cutoff_date]
            fc = forecast_iterative_xgb(
                df_hist=hist_train,
                artifact={"model": model, "cfg": cfg, "feature_names": feature_names},
                horizon=holdout,
            )
            fc_sorted = fc.set_index(dcol).reindex(holdout_dates)
            y_pred = fc_sorted["y_pred"].values
            xgb_preds = y_pred
            base_full = metrics_all(y_test, y_pred)
            xgb_scores = _strip_metrics(base_full)
        except Exception as e:
            xgb_error = f"XGBoost training/forecast failed: {e}"

    candidates = [(name, s["MAPE"]) for name, s in baseline_scores.items()]
    if xgb_scores is not None:
        candidates.append(("xgboost", xgb_scores["MAPE"]))
    best_name, _ = sorted(candidates, key=lambda x: x[1])[0]

    tolerance = float(cfg.get("training", {}).get("seasonal_tolerance_mape", 0.0))
    if tolerance > 0 and "naive" in baseline_scores and "seasonal_naive" in baseline_scores:
        if baseline_scores["seasonal_naive"]["MAPE"] <= baseline_scores["naive"]["MAPE"] + tolerance:
            if best_name == "naive":
                best_name = "seasonal_naive"

    scores = dict(baseline_scores)
    if xgb_scores is not None:
        scores["xgboost"] = xgb_scores

    holdout_preds = {}
    holdout_residuals = {}
    for name, arr in baseline_preds.items():
        holdout_preds[name] = {
            "dates": [d.strftime("%Y-%m-%d") for d in holdout_dates],
            "y_pred": arr.tolist(),
        }
        holdout_residuals[name] = (y_test - arr).tolist()

    if xgb_preds is not None:
        holdout_preds["xgboost"] = {
            "dates": [d.strftime("%Y-%m-%d") for d in holdout_dates],
            "y_pred": xgb_preds.tolist(),
        }
        holdout_residuals["xgboost"] = (y_test - xgb_preds).tolist()

    artifact = {
        "model_name": best_name,
        "scores": scores,
        "cfg": cfg,
        "cutoff_date": str(cutoff_date.date()),
        "holdout_months": holdout,
        "train_rows": int((hist[dcol] <= cutoff_date).sum()),
        "test_rows": int((hist[dcol] > cutoff_date).sum()),
        "selected_by": "MAPE",
        "holdout_y_actual": y_test.tolist(),
        "holdout_dates": [d.strftime("%Y-%m-%d") for d in holdout_dates],
        "holdout_preds": holdout_preds,
        "holdout_residuals": holdout_residuals,  # <-- tambahan
    }
    if blend_weight_used is not None:
        artifact["blend_weight_final"] = blend_weight_used
    if xgb_error:
        artifact["xgboost_error"] = xgb_error
    if best_name == "xgboost" and model is not None:
        artifact["model"] = model
        artifact["feature_names"] = feature_names

    return artifact


def save_artifact(
    artifact: Dict,
    out_dir: str = "models",
    filename_prefix: str = "kia_forecast",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = dict(artifact)
    model_path = ""
    if meta.get("model_name") == "xgboost" and "model" in meta:
        model_path = str(out / f"{filename_prefix}_model.pkl")
        joblib.dump(meta["model"], model_path)
        meta.pop("model", None)

    meta_path = str(out / f"{filename_prefix}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return model_path, meta_path