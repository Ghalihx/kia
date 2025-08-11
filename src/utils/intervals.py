import numpy as np

def compute_prediction_intervals(
    point_forecast: np.ndarray,
    residuals: np.ndarray,
    alpha: float = 0.05,
    method: str = "quantile",
    model_name: str = "",
    scale_for_horizon: bool = True
):
    """
    Hitung interval prediksi sederhana berbasis residual holdout.

    Parameter:
      point_forecast : np.ndarray shape (H,)
      residuals      : np.ndarray residual historis (y_true - y_pred)
      alpha          : 0.05 untuk 95% CI
      method         : 'quantile' atau 'normal'
      model_name     : nama model (bisa dipakai menentukan skala horizon)
      scale_for_horizon : jika True dan model tipe naive/seasonal/blend,
                          varian diperbesar sqrt(h) (random walk-ish).

    Catatan:
      - Interval ini sangat sederhana dan mengasumsikan residual stasioner.
      - Dengan sedikit residual (misal holdout 6), hasil bisa tidak stabil.
    """
    H = len(point_forecast)
    residuals = residuals[~np.isnan(residuals)]
    if residuals.size < 3:
        # fallback: garis tanpa lebar
        return point_forecast, point_forecast, point_forecast

    h_idx = np.arange(1, H + 1)
    if scale_for_horizon and any(k in model_name.lower() for k in ["naive", "seasonal", "blend"]):
        scale = np.sqrt(h_idx)  # asumsi random walk-ish
    else:
        scale = np.ones(H)

    if method == "normal":
        # Approx z untuk alpha (hindari dependency scipy)
        from math import sqrt
        # z 2-sided (alpha=0.05 -> 1.96); untuk general alpha gunakan aproksimasi:
        # z ≈ 1.96 untuk 0.05, 1.645 untuk 0.10, 2.576 untuk 0.01
        if abs(alpha - 0.05) < 1e-6:
            z = 1.96
        elif abs(alpha - 0.10) < 1e-6:
            z = 1.645
        elif abs(alpha - 0.01) < 1e-6:
            z = 2.576
        else:
            # fallback via persentil normal aproks sederhana (Beasley-Springer/Moro rumus disederhanakan)
            # Demi kesederhanaan: gunakan numpy quantile residual untuk substitusi (kurang tepat utk z)
            z = np.quantile(np.abs((residuals - residuals.mean()) / (residuals.std(ddof=1) + 1e-9)), 1 - alpha/2)
            if z <= 0:
                z = 1.96
        sigma = residuals.std(ddof=1)
        half_width = z * sigma * scale
        lower = point_forecast - half_width
        upper = point_forecast + half_width
    else:
        # quantile (default)
        q_low = np.quantile(residuals, alpha / 2)
        q_high = np.quantile(residuals, 1 - alpha / 2)
        # Terapkan scaling ke rentang residu
        center = (q_low + q_high) / 2
        half = (q_high - q_low) / 2
        adj_half = half * scale
        # Gunakan center sebagai shift rata-rata residu
        lower = point_forecast + center - adj_half
        upper = point_forecast + center + adj_half

    # Pastikan lower <= upper
    lower = np.minimum(lower, upper)
    return point_forecast, lower, upper