import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Rekap Kia Disdukcapil Jan 2023 - Jun 2025.csv")

def load_data(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["periode"])
    # Sort & reset
    df = df.sort_values("periode").reset_index(drop=True)
    # Validasi frekuensi bulanan
    full_range = pd.date_range(df["periode"].min(), df["periode"].max(), freq="MS")
    if len(full_range) != len(df):
        missing = set(full_range) - set(df["periode"])
        raise ValueError(f"Missing months: {sorted(missing)}")
    if df["permohonan_kia"].le(0).any():
        raise ValueError("Ada nilai <=0 pada permohonan_kia (cek data).")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Rows:", len(df))
