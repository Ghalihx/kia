from pathlib import Path
import pandas as pd

def load_events(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["periode","event_name","event_type","intensity"])
    df = pd.read_csv(p, parse_dates=["periode"])
    # pastikan 1 baris per periode
    df = df.sort_values("periode").drop_duplicates(subset=["periode"], keep="first")
    return df

def merge_event_features(main_df: pd.DataFrame,
                         events_df: pd.DataFrame,
                         date_col: str) -> pd.DataFrame:
    if events_df.empty:
        main_df["event_flag"] = 0
        main_df["event_intensity"] = 0.0
        return main_df
    out = main_df.merge(events_df, on=date_col, how="left")
    out["event_flag"] = out["event_type"].notna().astype(int)
    out["event_intensity"] = out["intensity"].fillna(0.0)
    # one-hot paling sering (maks 2 agar tidak sparce)
    top_types = out["event_type"].value_counts().index[:2]
    for et in top_types:
        out[f"evt_{et}"] = (out["event_type"] == et).astype(int)
    # lag event 1 bulan
    out["event_flag_lag1"] = out["event_flag"].shift(1).fillna(0).astype(int)
    out["post_event_1"] = ((out["event_flag_lag1"]==1) & (out["event_flag"]==0)).astype(int)
    return out
