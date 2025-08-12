import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.models.train import load_config, train_pipeline, save_artifact
from src.models.infer import (
    forecast_iterative_xgb,
    forecast_iterative_seasonal_naive,
    forecast_iterative_naive,
    load_artifact,
)
from src.pipeline.data_prep import load_and_validate

# Interval (opsional)
try:
    from src.utils.intervals import compute_prediction_intervals
    HAS_INTERVALS = True
except Exception:
    HAS_INTERVALS = False

st.set_page_config(page_title="Prediksi Permohonan KIA", layout="wide")
st.title("Sistem Prediksi Permohonan KIA - Disdukcapil Kota Bogor")

cfg = load_config()
forecast_cfg = cfg.get("forecast", {})
default_horizon = int(forecast_cfg.get("horizon", 6))
max_horizon = int(forecast_cfg.get("max_horizon", 24))

with st.sidebar:
    st.header("Pengaturan")
    horizon = st.number_input("Horizon Prediksi (bulan)", min_value=1, max_value=max_horizon, value=default_horizon)
    use_existing_model = st.checkbox("Gunakan model tersimpan (jika ada)", value=False)
    show_details = st.checkbox("Detail skor lengkap", value=False)
    show_holdout_plot = st.checkbox("Plot holdout", value=True)
    advanced_mode = st.checkbox("Mode Advanced", value=False)
    if advanced_mode:
        show_ape_expander = st.checkbox("Tampilkan Analisis APE", value=True)
        show_residual_analysis = st.checkbox("Tampilkan Residual", value=False)
        show_prediction_intervals = st.checkbox("Interval Prediksi", value=False, disabled=not HAS_INTERVALS)
        interval_method = st.selectbox("Metode Interval", ["quantile", "normal"],
                                       disabled=not show_prediction_intervals or not HAS_INTERVALS)
        interval_alpha = st.selectbox("Alpha (1 - confidence)",
                                      [0.01, 0.025, 0.05, 0.10],
                                      index=2,
                                      disabled=not show_prediction_intervals or not HAS_INTERVALS)
    else:
        show_ape_expander = False
        show_residual_analysis = False
        show_prediction_intervals = False
        interval_method = "quantile"
        interval_alpha = 0.05

    train_button = st.button("Latih Model")
    predict_button = st.button("Prediksi ke Depan")

st.subheader("1) Unggah Data Historis")
uploaded = st.file_uploader("Unggah CSV (kolom: periode, permohonan_kia)", type=["csv"])

if uploaded is None:
    st.info("Unggah data untuk memulai.")
    st.stop()

# Baca CSV
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

# Validasi
try:
    df = load_and_validate(df_raw, cfg)
except Exception as e:
    st.error(f"Validasi data gagal: {e}")
    st.stop()

st.success(f"Data OK. Jumlah periode: {df.shape[0]}")

dcol = cfg["data"]["date_column"]
ycol = cfg["data"]["target_column"]

fig_hist = px.line(df, x=dcol, y=ycol, title="Historis Permohonan KIA")
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("2) Pelatihan & Evaluasi")
artifact = None

def _show_scores(scores: dict):
    if not scores:
        st.info("Belum ada skor.")
        return
    if show_details:
        st.json(scores)
    else:
        cols = st.columns(len(scores))
        for i, (m, sc) in enumerate(scores.items()):
            with cols[i]:
                st.metric(m, f"MAPE {sc['MAPE']:.2f}%", help=f"RMSE={sc['RMSE']:.2f}")

# Load existing
if use_existing_model:
    try:
        artifact = load_artifact("models/kia_forecast")
        st.info(f"Model tersimpan: {artifact['model_name']}")
        _show_scores(artifact["scores"])
    except Exception as e:
        st.warning(f"Gagal memuat model tersimpan: {e}")

# Train new
elif train_button:
    with st.spinner("Melatih model..."):
        artifact = train_pipeline(df, cfg)
        save_artifact(artifact, out_dir="models", filename_prefix="kia_forecast")

    st.success(f"Model '{artifact['model_name']}' tersimpan.")
    _show_scores(artifact["scores"])

    # Simpan history (tetap ditulis ke file, tapi TIDAK lagi ditampilkan di UI)
    try:
        from datetime import datetime
        import csv
        history_path = Path("models") / "model_history.csv"
        fields = ["timestamp", "model_name", "blend_weight",
                  "MAPE_naive", "MAPE_seasonal", "MAPE_xgb", "MAPE_blend",
                  "RMSE_naive", "RMSE_seasonal", "RMSE_xgb", "RMSE_blend"]
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": artifact.get("model_name"),
            "blend_weight": artifact.get("blend_weight_final"),
            "MAPE_naive": artifact["scores"].get("naive", {}).get("MAPE"),
            "MAPE_seasonal": artifact["scores"].get("seasonal_naive", {}).get("MAPE"),
            "MAPE_xgb": artifact["scores"].get("xgboost", {}).get("MAPE"),
            "MAPE_blend": artifact["scores"].get("blend", {}).get("MAPE"),
            "RMSE_naive": artifact["scores"].get("naive", {}).get("RMSE"),
            "RMSE_seasonal": artifact["scores"].get("seasonal_naive", {}).get("RMSE"),
            "RMSE_xgb": artifact["scores"].get("xgboost", {}).get("RMSE"),
            "RMSE_blend": artifact["scores"].get("blend", {}).get("RMSE"),
        }
        write_header = not history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        st.warning(f"Gagal tulis model_history: {e}")

    with st.expander("Info Training"):
        st.write({
            "model_name": artifact.get("model_name"),
            "cutoff_date": artifact.get("cutoff_date"),
            "holdout_months": artifact.get("holdout_months"),
            "train_rows": artifact.get("train_rows"),
            "test_rows": artifact.get("test_rows"),
            "blend_weight_final": artifact.get("blend_weight_final"),
            "xgboost_error": artifact.get("xgboost_error"),
        })

# Plot holdout
if artifact and show_holdout_plot and "holdout_preds" in artifact:
    st.markdown("### Holdout: Actual vs Prediksi")
    try:
        holdout_dates = pd.to_datetime(artifact["holdout_dates"])
        y_actual = artifact["holdout_y_actual"]
        plot_df = pd.DataFrame({dcol: holdout_dates, "actual": y_actual})
        for mname, obj in artifact["holdout_preds"].items():
            plot_df[mname] = obj["y_pred"]
        fig_holdout = px.line(plot_df, x=dcol,
                              y=[c for c in plot_df.columns if c != dcol],
                              title="Holdout Comparison")
        st.plotly_chart(fig_holdout, use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal membuat plot holdout: {e}")

# APE analysis
if artifact and advanced_mode and show_ape_expander:
    with st.expander("Analisis APE"):
        try:
            y_true = np.array(artifact["holdout_y_actual"])
            dates = pd.to_datetime(artifact["holdout_dates"])
            data = {"periode": dates, "actual": y_true}
            for mname, obj in artifact["holdout_preds"].items():
                preds = np.array(obj["y_pred"])
                data[mname] = preds
                ape = np.where(y_true != 0, np.abs((y_true - preds) / y_true) * 100, np.nan)
                data[f"APE_{mname}"] = ape
            df_ape = pd.DataFrame(data)
            if {"APE_naive", "APE_blend"}.issubset(df_ape.columns):
                df_ape["APE_improve_blend_vs_naive"] = df_ape["APE_naive"] - df_ape["APE_blend"]
            st.dataframe(df_ape, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal analisis APE: {e}")

# Residual
if artifact and advanced_mode and show_residual_analysis:
    with st.expander("Residual (Holdout)"):
        try:
            if "holdout_residuals" not in artifact:
                st.info("Tidak ada residual.")
            else:
                dates = pd.to_datetime(artifact["holdout_dates"])
                rows = []
                for m, res_list in artifact["holdout_residuals"].items():
                    for d, r in zip(dates, res_list):
                        rows.append({"periode": d, "model": m, "residual": r})
                df_res = pd.DataFrame(rows)
                st.dataframe(df_res.pivot(index="periode", columns="model", values="residual"),
                             use_container_width=True)
        except Exception as e:
            st.error(f"Gagal menampilkan residual: {e}")

st.subheader("3) Prediksi ke Depan")
download_slot = st.empty()

if predict_button:
    if artifact is None:
        try:
            artifact = load_artifact("models/kia_forecast")
        except Exception:
            with st.spinner("Model belum ada, melatih cepat..."):
                artifact = train_pipeline(df, cfg)
                save_artifact(artifact, out_dir="models", filename_prefix="kia_forecast")

    try:
        model_name = artifact["model_name"]
        st.write(f"Model digunakan: {model_name}")
        # Pilih metode forecast
        if model_name == "xgboost":
            fc = forecast_iterative_xgb(df[[dcol, ycol]], artifact, horizon=horizon)
            fc_df = fc.rename(columns={"y_pred": "prediksi"})
        elif model_name == "seasonal_naive":
            fc = forecast_iterative_seasonal_naive(
                df_hist=df[[dcol, ycol]],
                cfg=cfg,
                horizon=horizon,
                season_length=12
            )
            fc_df = fc.rename(columns={"y_pred": "prediksi"})
        elif model_name == "blend":
            from src.models.baselines import blend_forecast
            y_hist = df[ycol].values
            w = float(artifact.get("blend_weight_final", 0.5) or 0.5)
            preds = blend_forecast(y_hist, horizon=horizon, w=w, season_length=12)
            last_date = df[dcol].iloc[-1]
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
            fc_df = pd.DataFrame({dcol: future_dates, "prediksi": preds})
            st.caption(f"Blend weight optimal: w={w:.2f}")
        else:
            fc = forecast_iterative_naive(df_hist=df[[dcol, ycol]], cfg=cfg, horizon=horizon)
            fc_df = fc.rename(columns={"y_pred": "prediksi"})

        fc_df["prediksi"] = fc_df["prediksi"].astype(float)

        # Interval 
        if advanced_mode and show_prediction_intervals and HAS_INTERVALS:
            if "holdout_residuals" in artifact and model_name in artifact["holdout_residuals"]:
                residuals = np.array(artifact["holdout_residuals"][model_name], dtype=float)
                if residuals.size >= 2:
                    _, lower, upper = compute_prediction_intervals(
                        point_forecast=fc_df["prediksi"].values,
                        residuals=residuals,
                        alpha=float(interval_alpha),
                        method=interval_method,
                        model_name=model_name,
                        scale_for_horizon=True
                    )
                    fc_df["lower"] = lower
                    fc_df["upper"] = upper
                else:
                    st.warning("Residual terlalu sedikit untuk interval.")
            else:
                st.warning("Residual model tidak tersedia; interval dilewati.")

        # Format tampilan
        fc_df["prediksi_fmt"] = fc_df["prediksi"].map(lambda v: f"{v:,.2f}")
        if "lower" in fc_df.columns:
            fc_df["lower_fmt"] = fc_df["lower"].map(lambda v: f"{v:,.2f}")
        if "upper" in fc_df.columns:
            fc_df["upper_fmt"] = fc_df["upper"].map(lambda v: f"{v:,.2f}")

        show_cols = [dcol, "prediksi_fmt"]
        if "lower_fmt" in fc_df.columns and "upper_fmt" in fc_df.columns:
            show_cols += ["lower_fmt", "upper_fmt"]
        st.dataframe(fc_df[show_cols], use_container_width=True)

        # Grafik sederhana (tanpa interval shading)
        hist_plot = df[[dcol, ycol]].rename(columns={dcol: "periode", ycol: "aktual"})
        future_plot = fc_df.rename(columns={dcol: "periode"})
        future_plot["aktual"] = np.nan
        plot_df = pd.concat([hist_plot, future_plot], ignore_index=True)

        fig = px.line(plot_df, x="periode", y=["aktual", "prediksi"], title=f"Forecast ({model_name})")
        st.plotly_chart(fig, use_container_width=True)

        # Download CSV
        export_cols = [dcol, "prediksi"]
        if "lower" in fc_df.columns and "upper" in fc_df.columns:
            export_cols += ["lower", "upper"]
        export_df = fc_df[export_cols].rename(columns={dcol: "periode"})
        csv_bytes = export_df.to_csv(index=False, float_format="%.2f")
        download_slot.download_button(
            "Download Forecast CSV",
            data=csv_bytes,
            file_name="forecast_interval.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
