# Tambah import di atas:
import hashlib
from datetime import datetime

def hash_data(df: pd.DataFrame, date_col: str, target_col: str) -> str:
    h = hashlib.md5()
    h.update(df[date_col].astype(str).str.cat(sep="|").encode())
    h.update(b"::")
    h.update(df[target_col].astype(str).str.cat(sep="|").encode())
    return h.hexdigest()

def train_pipeline(df: pd.DataFrame, cfg: dict,
                   fast_mode: bool = False,
                   progress_callback=None) -> Dict[str, Any]:
    def cb(event, info=None):
        if progress_callback:
            try:
                progress_callback(event, info or {})
            except Exception:
                pass

    cb("phase", {"label": "Mulai persiapan data"})
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

    # Fast mode override
    if fast_mode:
        lags = [1, 12] if 12 in lags else ([1] + lags[:1])
        rollings = []
    
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[target_col].isna().any():
        raise ValueError("Ada nilai target yang tidak numerik / NaN.")

    df = df.sort_values(date_col).reset_index(drop=True)

    total_rows = len(df)
    if total_rows <= holdout_months + 6:
        raise ValueError("Data terlalu sedikit untuk holdout. Tambah data atau kurangi holdout_months.")

    train_df = df.iloc[:-holdout_months].copy()
    test_df = df.iloc[-holdout_months:].copy()
    test_dates = list(test_df[date_col])
    y_test_actual = test_df[target_col].values.astype(float)

    cb("phase", {"label": "Bangun fitur"})
    feat_all = build_features(df, date_col, target_col, lags, rollings, add_sin_cos)

    # Drop rolling fitur yang terlalu banyak NaN (misal valid < 40% baris train)
    drop_cols = []
    for c in feat_all.columns:
        if c.startswith("roll_"):
            valid_ratio = feat_all[c].notna().mean()
            if valid_ratio < 0.4:
                drop_cols.append(c)
    if drop_cols:
        feat_all = feat_all.drop(columns=drop_cols)

    feat_train = feat_all[feat_all[date_col].isin(train_df[date_col])]
    feat_test = feat_all[feat_all[date_col].isin(test_df[date_col])]

    X_cols = [c for c in feat_train.columns if c not in [date_col, target_col]]
    X_train = feat_train[X_cols].fillna(0).to_numpy()
    X_test = feat_test[X_cols].fillna(0).to_numpy()
    y_train = feat_train[target_col].values.astype(float)
    y_test = feat_test[target_col].values.astype(float)

    artifact: Dict[str, Any] = {
        "schema_version": "1.2.0",
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
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "date_column": date_col,
        "target_column": target_col,
        "data_hash": hash_data(df, date_col, target_col),
        "fast_mode": fast_mode
    }

    # Baseline Naive
    cb("phase", {"label": "Baseline naive"})
    naive_pred = predict_naive_last(train_df, test_dates, date_col, target_col)
    artifact["scores"]["naive"] = {
        "MAPE": mape(y_test_actual, naive_pred),
        "RMSE": rmse(y_test_actual, naive_pred)
    }
    artifact["holdout_preds"]["naive"] = {"y_pred": naive_pred.tolist()}
    artifact["holdout_residuals"]["naive"] = (y_test_actual - naive_pred).tolist()

    # Baseline Seasonal Naive
    cb("phase", {"label": "Baseline seasonal naive"})
    seasonal_pred = predict_seasonal_naive(train_df, test_dates, date_col, target_col, season_length=season_length)
    artifact["scores"]["seasonal_naive"] = {
        "MAPE": mape(y_test_actual, seasonal_pred),
        "RMSE": rmse(y_test_actual, seasonal_pred)
    }
    artifact["holdout_preds"]["seasonal_naive"] = {"y_pred": seasonal_pred.tolist()}
    artifact["holdout_residuals"]["seasonal_naive"] = (y_test_actual - seasonal_pred).tolist()

    use_xgb = (not fast_mode) and (len(train_df) >= min_rows_xgb)
    xgb_scores = None
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
            # Gunakan API sklearn
            early_rounds = cfg.get("training", {}).get("xgb_early_stopping_rounds", 30)
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
                early_stopping_rounds=early_rounds
            )
            xgb_pred = model.predict(X_test)
            xgb_scores = {
                "MAPE": mape(y_test_actual, xgb_pred),
                "RMSE": rmse(y_test_actual, xgb_pred)
            }
            artifact["scores"]["xgboost"] = xgb_scores
            artifact["holdout_preds"]["xgboost"] = {"y_pred": xgb_pred.tolist()}
            artifact["holdout_residuals"]["xgboost"] = (y_test_actual - xgb_pred).tolist()
            # Simpan model raw (opsional kecil); kalau mau hemat simpan model.save_model ke file terpisah
            artifact["xgb_model_raw"] = model.get_booster().save_raw("json").decode()
        except Exception as e:
            artifact["xgboost_error"] = str(e)
            use_xgb = False

    # Blend sederhana (jika ada xgb)
    if use_xgb and xgb_pred is not None:
        cb("phase", {"label": "Blend"})
        # Contoh bobot otomatis: cari w meminimalkan MAPE: pred = w*xgb + (1-w)*seasonal
        y1 = seasonal_pred
        y2 = xgb_pred
        best_w = 0.5
        best_mape = 1e9
        for w in np.linspace(0, 1, 21):
            blend_p = w * y2 + (1 - w) * y1
            sc = mape(y_test_actual, blend_p)
            if sc < best_mape:
                best_mape = sc
                best_w = w
        blend_pred = best_w * y2 + (1 - best_w) * y1
        artifact["holdout_preds"]["blend"] = {"y_pred": blend_pred.tolist()}
        artifact["holdout_residuals"]["blend"] = (y_test_actual - blend_pred).tolist()
        artifact["scores"]["blend"] = {
            "MAPE": mape(y_test_actual, blend_pred),
            "RMSE": rmse(y_test_actual, blend_pred)
        }
        artifact["blend_weight_final"] = best_w
        # Pilih model_name final
        # Ambil yang MAPE paling kecil di antara seasonal_naive, xgboost, blend
        ranking = []
        for m in ["seasonal_naive", "xgboost", "blend"]:
            if m in artifact["scores"]:
                ranking.append((artifact["scores"][m]["MAPE"], m))
        ranking.sort()
        artifact["model_name"] = ranking[0][1]
    else:
        # Pilih terbaik dari baseline saja
        baseline_rank = [
            (artifact["scores"]["naive"]["MAPE"], "naive"),
            (artifact["scores"]["seasonal_naive"]["MAPE"], "seasonal_naive")
        ]
        baseline_rank.sort()
        artifact["model_name"] = baseline_rank[0][1]

    cb("done", {})
    return artifact
