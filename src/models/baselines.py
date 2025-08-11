from __future__ import annotations
import numpy as np


def naive_forecast(y_hist, horizon: int):
    """
    Naive: semua langkah = nilai terakhir histori.
    """
    y_hist = np.asarray(y_hist, dtype=float)
    if y_hist.size == 0:
        return np.array([], dtype=float)
    return np.full(horizon, y_hist[-1], dtype=float)


def seasonal_naive_forecast(y_hist, horizon: int, season_length: int = 12):
    """
    Seasonal naive multi-step:
      ŷ_{t+h} = y_{t+h - season_length} jika h <= season_length
                 ŷ_{t+h - season_length} (hasil forecast sebelumnya) jika h > season_length
    Jika panjang y_hist < season_length → fallback ke naive (semua = nilai terakhir).
    Mendukung horizon > season_length secara rekursif.
    """
    y_hist = np.asarray(y_hist, dtype=float)
    n = y_hist.size
    if n == 0 or horizon <= 0:
        return np.array([], dtype=float)
    if season_length <= 0:
        raise ValueError("season_length harus > 0")

    # Jika histori belum cukup untuk satu musim penuh → fallback naive.
    if n < season_length:
        return np.full(horizon, y_hist[-1], dtype=float)

    out = np.empty(horizon, dtype=float)
    # h: 1..horizon (loop 0-based -> i)
    for i in range(horizon):
        h = i + 1
        if h <= season_length:
            # Ambil nilai dari musim lalu: index = n - season_length + (h - 1)
            idx = n - season_length + (h - 1)
            out[i] = y_hist[idx]
        else:
            # Gunakan forecast yang sudah dibuat season_length langkah sebelumnya
            out[i] = out[i - season_length]
    return out


def blend_forecast(y_hist, horizon: int, w: float = 0.5, season_length: int = 12):
    """
    Blend: w * naive + (1-w) * seasonal_naive.
    w di [0,1].
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError("w harus di antara 0 dan 1.")
    y_hist = np.asarray(y_hist, dtype=float)
    naive = naive_forecast(y_hist, horizon)
    seas = seasonal_naive_forecast(y_hist, horizon, season_length=season_length)
    return w * naive + (1.0 - w) * seas