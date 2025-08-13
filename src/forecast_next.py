import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from pathlib import Path

FEATURES = [
    "month","month_sin","month_cos",
    "lag_1","lag_2","lag_3","lag_6","lag_12",
    "roll3_mean","roll6_mean","roll3_std","roll6_std",
    "mom_pct","yoy_pct","outlier_flag"
]
TARGET = "permohonan_kia"

FEAT_PATH = Path("data/processed/permohonan_kia_features.csv")
OUT_PATH = Path("data/forecast/permohonan_kia_forecast.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HORIZON = 6  # contoh forecast 6 bulan ke depan

df = pd.read_csv(FEAT_PATH, parse_dates=["periode"])
df_model = df.dropna(subset=FEATURES).copy()

X = df_model[FEATURES].values
y = df_model[TARGET].values

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
model.fit(X, y)

# Recursive forecast
future_records = []
current_df = df.copy()

for step in range(1, HORIZON+1):
    last_row = current_df.iloc[-1]
    next_periode = (last_row["periode"] + pd.offsets.MonthBegin(1))

    # Bangun row baru dengan lags dari current_df
    temp = {}
    temp["periode"] = next_periode
    temp["year"] = next_periode.year
    temp["month"] = next_periode.month
    temp["month_sin"] = np.sin(2*np.pi*temp["month"]/12)
    temp["month_cos"] = np.cos(2*np.pi*temp["month"]/12)

    # Lags
    for lag in [1,2,3,6,12]:
        if len(current_df) >= lag:
            temp[f"lag_{lag}"] = current_df["permohonan_kia"].iloc[-lag]
        else:
            temp[f"lag_{lag}"] = np.nan

    # Rolling (recompute with available series)
    series = current_df["permohonan_kia"]
    temp["roll3_mean"] = series.tail(3).mean()
    temp["roll6_mean"] = series.tail(6).mean() if len(series) >= 6 else series.mean()
    temp["roll3_std"]  = series.tail(3).std()
    temp["roll6_std"]  = series.tail(6).std()

    # Growth (need last available)
    last_val = series.iloc[-1]
    prev_val = series.iloc[-2] if len(series) >= 2 else np.nan
    temp["mom_pct"] = (last_val - prev_val)/prev_val*100 if prev_val and prev_val != 0 else np.nan

    if len(series) >= 13:
        val_12 = series.iloc[-12]
        temp["yoy_pct"] = (last_val - val_12)/val_12*100 if val_12 != 0 else np.nan
    else:
        temp["yoy_pct"] = np.nan

    # simple outlier flag (0 for future by default)
    temp["outlier_flag"] = 0

    feat_vector = [temp.get(f, np.nan) for f in FEATURES]

    # Jika ada NaN penting (misal lag_12 pada awal), bisa fallback ke rata-rata
    if np.isnan(feat_vector).any():
        # Sederhana: ganti NaN dengan median kolom pelatihan
        medians = df_model[FEATURES].median()
        feat_vector = [medians[f] if np.isnan(v) else v for f,v in zip(FEATURES, feat_vector)]

    y_pred = model.predict([feat_vector])[0]
    temp["permohonan_kia"] = y_pred

    future_records.append(temp)
    # Append ke current_df untuk langkah berikutnya
    current_df = pd.concat([current_df, pd.DataFrame([temp])], ignore_index=True)

future_df = pd.DataFrame(future_records)
future_df = future_df[["periode","permohonan_kia"]]

future_df.to_csv(OUT_PATH, index=False)
print("Saved future forecast to", OUT_PATH)
