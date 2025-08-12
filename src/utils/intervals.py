from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

# Coba pakai SciPy untuk z-value; jika tidak ada, fallback manual
try:
    from scipy.stats import norm
except ImportError:
    norm = None


def _z_from_alpha(alpha: float) -> float:
    """
    Ambil z (two-sided) untuk alpha (misal alpha=0.05 -> ~1.96).
    Fallback kalau scipy tidak tersedia: gunakan aproksimasi percentile normal.
    """
    if norm is not None:
        return float(norm.ppf(1 - alpha / 2.0))
    # Aproksimasi kasar pakai persentil simetri (cukup baik untuk alpha umum)
    # alpha umum: 0.10 -> 1.645; 0.05 -> 1.96; 0.025 -> 2.24; 0.01 -> 2.575
    common = {
        0.10: 1.6449,
        0.05: 1.96,
        0.025: 2.2414,
        0.01: 2.5758,
        0.001: 3.2905,
    }
    # Cari yang mendekati
    return common.get(round(alpha, 5), 1.96)


def compute_prediction_intervals(point_forecast: np.ndarray,
                                 residuals: np.ndarray,
                                 alpha: float = 0.05,
                                 method: str = "quantile",
                                 model_name: Optional[str] = None,
                                 scale_for_horizon: bool = True
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengembalikan tuple: (point_forecast, lower, upper)

    Parameters
    ----------
    point_forecast : np.ndarray
        Array nilai prediksi titik (horizon urut).
    residuals : np.ndarray
        Residual historis (actual - prediksi) dari holdout.
    alpha : float
        1 - confidence level. alpha=0.05 => 95% interval.
    method : str
        "quantile" (empiris) atau "normal".
    model_name : Optional[str]
        Hanya dummy agar kompatibel dengan pemanggilan lama (tidak dipakai).
    scale_for_horizon : bool
        Jika method="normal" dan True, lebar interval diskalakan sqrt(h).

    Returns
    -------
    (pf, lower, upper)
        pf    = array point_forecast (dipastikan float)
        lower = batas bawah
        upper = batas atas
    """
    pf = np.asarray(point_forecast, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    # Bersihkan NaN
    residuals = residuals[~np.isnan(residuals)]

    # Jika residual minim, tidak bisa bikin interval berarti → pakai pf saja
    if residuals.size < 2:
        return pf, pf.copy(), pf.copy()

    alpha = float(alpha)
    method = (method or "quantile").lower().strip()

    if method == "normal":
        sigma = float(np.std(residuals, ddof=1))
        z = _z_from_alpha(alpha)
        lowers = []
        uppers = []
        for h, yhat in enumerate(pf, start=1):
            scale = np.sqrt(h) if scale_for_horizon else 1.0
            delta = z * sigma * scale
            lowers.append(yhat - delta)
            uppers.append(yhat + delta)
        return pf, np.array(lowers), np.array(uppers)

    # Default: quantile (empiris)
    q_low = float(np.quantile(residuals, alpha / 2.0))
    q_high = float(np.quantile(residuals, 1 - alpha / 2.0))
    lower = pf + q_low
    upper = pf + q_high
    return pf, lower, upper


__all__ = ["compute_prediction_intervals"]
