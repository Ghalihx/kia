from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

try:
    import holidays as pyholidays
except Exception:
    pyholidays = None  # type: ignore


@dataclass
class PrepConfig:
    date_column: str
    target_column: str
    lags: List[int]
    rollings: List[int]
    holiday_country: str = "ID"
    add_sin_cos: bool = True


def _to_month_start(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def _check_monthly_contiguity(dates: pd.Series):
    dates = dates.sort_values().reset_index(drop=True)
    full_range = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="MS")
    if len(full_range) != len(dates):
        missing = full_range.difference(dates)
        if len(missing) > 0:
            miss_list = [d.strftime("%Y-%m") for d in missing]
            raise ValueError(
                f"Data bulanan tidak kontigu. Bulan hilang: {miss_list}. "
                "Lengkapi data atau isi baris dengan 0 jika memang tidak ada permohonan."
            )


def load_and_validate(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    dcol = cfg["data"]["date_column"]
    ycol = cfg["data"]["target_column"]

    if dcol not in df.columns or ycol not in df.columns:
        raise ValueError(f"Kolom wajib tidak ditemukan. Harus ada: {dcol}, {ycol}")

    out = df[[dcol, ycol]].copy()

    try:
        out[dcol] = out[dcol].apply(_to_month_start)
    except Exception as e:
        raise ValueError(f"Gagal parsing kolom tanggal '{dcol}': {e}")

    if out[ycol].isna().any():
        raise ValueError("Terdapat nilai kosong pada kolom target.")

    if not np.issubdtype(out[ycol].dtype, np.number):
        try:
            out[ycol] = pd.to_numeric(out[ycol])
        except Exception:
            raise ValueError("Kolom target harus bertipe numerik.")

    if (out[ycol] % 1 == 0).all():
        out[ycol] = out[ycol].astype(int)
    if (out[ycol] < 0).any():
        raise ValueError("Nilai target tidak boleh negatif.")

    if out[dcol].duplicated().any():
        out = out.groupby(dcol, as_index=False)[ycol].sum()

    out = out.sort_values(dcol).reset_index(drop=True)
    _check_monthly_contiguity(out[dcol])
    return out


def _holiday_counter(years: List[int], country_code: str):
    if pyholidays is None:
        return set()
    years = sorted(set(years))
    try:
        return pyholidays.country_holidays(country_code=country_code, years=years)
    except Exception:
        return set()


def add_calendar_features(df: pd.DataFrame, prep: PrepConfig) -> pd.DataFrame:
    dcol = prep.date_column
    out = df.copy()
    out["month"] = out[dcol].dt.month
    out["quarter"] = out[dcol].dt.quarter
    out["year"] = out[dcol].dt.year

    if prep.add_sin_cos:
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    years = list(range(out[dcol].dt.year.min(), out[dcol].dt.year.max() + 2))
    hcal = _holiday_counter(years, prep.holiday_country)
    if len(hcal) == 0:
        out["holiday_count"] = 0
    else:
        def count_holidays_in_month(ts: pd.Timestamp) -> int:
            start = ts
            end = ts + pd.offsets.MonthEnd(0)
            days = pd.date_range(start, end, freq="D")
            return sum(1 for d in days if d in hcal)
        out["holiday_count"] = out[dcol].apply(count_holidays_in_month)
    return out


def add_lag_rolling_features(df: pd.DataFrame, prep: PrepConfig) -> pd.DataFrame:
    ycol = prep.target_column
    out = df.copy()

    for lag in prep.lags:
        out[f"{ycol}_lag{lag}"] = out[ycol].shift(lag)

    for win in prep.rollings:
        out[f"{ycol}_roll{win}"] = out[ycol].shift(1).rolling(window=win, min_periods=win).mean()
    return out


def make_supervised(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    prep = PrepConfig(
        date_column=cfg["data"]["date_column"],
        target_column=cfg["data"]["target_column"],
        lags=cfg["data"]["lags"],
        rollings=cfg["data"]["rollings"],
        holiday_country=cfg["data"].get("holiday_country", "ID"),
        add_sin_cos=cfg["data"].get("add_sin_cos", True),
    )

    base = add_calendar_features(df, prep)
    sup = add_lag_rolling_features(base, prep)

    ycol = prep.target_column
    feat_cols = ["month", "quarter", "holiday_count"] \
        + ([ "sin_month", "cos_month"] if prep.add_sin_cos else []) \
        + [f"{ycol}_lag{l}" for l in prep.lags] \
        + [f"{ycol}_roll{r}" for r in prep.rollings]

    sup = sup.dropna(subset=feat_cols).reset_index(drop=True)
    return sup[[prep.date_column, ycol] + feat_cols], feat_cols