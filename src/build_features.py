import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/Rekap Kia Disdukcapil Jan 2023 - Jun 2025.csv")
OUT = Path("data/processed/permohonan_kia_features.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, parse_dates=["periode"])
df = df.sort_values("periode").reset_index(drop=True)

# Basic time parts
df["year"] = df["periode"].dt.year
df["month"] = df["periode"].dt.month

# Cyclical encoding
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Lags
for lag in [1,2,3,6,12]:
    df[f"lag_{lag}"] = df["permohonan_kia"].shift(lag)

# Rolling stats
df["roll3_mean"] = df["permohonan_kia"].rolling(3).mean()
df["roll6_mean"] = df["permohonan_kia"].rolling(6).mean()
df["roll3_std"]  = df["permohonan_kia"].rolling(3).std()
df["roll6_std"]  = df["permohonan_kia"].rolling(6).std()

# Growth
df["mom_pct"] = df["permohonan_kia"].pct_change() * 100
df["yoy_pct"] = df["permohonan_kia"].pct_change(12) * 100

# Outlier flag (berdasarkan deviasi dari roll6_mean)
threshold = 2
df["outlier_flag"] = 0
mask_roll = df["roll6_mean"].notna()
df.loc[mask_roll & (np.abs(df["permohonan_kia"] - df["roll6_mean"]) > threshold * df["roll6_std"]), "outlier_flag"] = 1

# Target forward (untuk supervized next-month forecast)
df["target_next_1"] = df["permohonan_kia"].shift(-1)

df.to_csv(OUT, index=False)
print("Saved features to", OUT)
