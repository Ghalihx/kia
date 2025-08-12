from __future__ import annotations
import numpy as np

def blend_forecast(history_values: np.ndarray,
                   horizon: int,
                   w: float,
                   season_length: int = 12) -> np.ndarray:
    """
    Blend sederhana: w * naive_last + (1-w) * seasonal_naive
    (Dipakai hanya saat model_name == 'blend' di Streamlit v13 style).
    """
    history_values = np.asarray(history_values, dtype=float)
    last_val = history_values[-1]
    seasonal_part = []
    for h in range(1, horizon + 1):
        if len(history_values) >= season_length:
            idx = -season_length + (h - 1)
            if abs(idx) <= len(history_values):
                seasonal_part.append(history_values[idx])
            else:
                seasonal_part.append(last_val)
        else:
            seasonal_part.append(last_val)
    seasonal_part = np.array(seasonal_part, dtype=float)
    naive_part = np.full(horizon, last_val, dtype=float)
    return w * naive_part + (1 - w) * seasonal_part
