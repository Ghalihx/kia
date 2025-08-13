import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from time_series_cv import expanding_window_splits

FEATURES = [
    "month","month_sin","month_cos",
    "lag_1","lag_2","lag_3","lag_6","lag_12",
    "roll3_mean","roll6_mean","roll3_std","roll6_std",
    "mom_pct","yoy_pct","outlier_flag"
]

TARGET = "permohonan_kia"

FEAT_PATH = Path("data/processed/permohonan_kia_features.csv")
METRIC_OUT = Path("reports/xgboost_cv_metrics.csv")
PRED_OUT = Path("data/processed/xgb_cv_predictions.csv")
METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
PRED_OUT.parent.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

df = pd.read_csv(FEAT_PATH, parse_dates=["periode"])
# Drop rows with NaN in features (awal-awal lags)
df_model = df.dropna(subset=FEATURES)

y = df_model[TARGET].values
X = df_model[FEATURES].values
periods = df_model["periode"].values

records = []
pred_rows = []

n = len(df_model)
initial = 18  # minimal train size (sesuaikan)
for train_idx, test_idx in expanding_window_splits(n=n, initial=initial, step=1, horizon=1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rec = {
        "periode": periods[test_idx][0],
        "y_true": y_test[0],
        "y_pred": y_pred[0]
    }
    pred_rows.append(rec)

df_pred = pd.DataFrame(pred_rows)
df_pred["abs_error"] = (df_pred["y_true"] - df_pred["y_pred"]).abs()
df_pred["pct_error"] = df_pred["abs_error"] / df_pred["y_true"] * 100

overall_mape = df_pred["pct_error"].mean()
overall_rmse = rmse(df_pred["y_true"], df_pred["y_pred"])

df_pred.to_csv(PRED_OUT, index=False)

pd.DataFrame([{
    "model":"xgboost",
    "MAPE":overall_mape,
    "RMSE":overall_rmse,
    "rows_eval":len(df_pred)
}]).to_csv(METRIC_OUT, index=False)

print("MAPE:", overall_mape, "RMSE:", overall_rmse)
print("Saved predictions to", PRED_OUT)
