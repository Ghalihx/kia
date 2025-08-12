import sys
from pathlib import Path
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.train import load_config, train_pipeline, save_artifact
from src.models.infer import (
    forecast_iterative_xgb,
    forecast_iterative_seasonal_naive,
    forecast_iterative_naive,
    load_artifact as load_artifact_file,
)
from src.pipeline.data_prep import load_and_validate

try:
    from src.utils.intervals import compute_prediction_intervals
    HAS_INTERVALS = True
except Exception:
    HAS_INTERVALS = False

st.set_page_config(page_title="Prediksi Permohonan KIA", layout="wide")
st.title("Sistem Prediksi Permohonan KIA - Disdukcapil Kota Bogor")

# Session state init
ss = st.session_state
if "artifact" not in ss: ss.artifact = None
if "data_hash" not in ss: ss.data_hash = None

@st.cache_data(show_spinner=False)
def cached_load_config():
    return load_config()

@st.cache_data(show_spinner=False)
def cached_validate(df_raw, cfg):
    return load_and_validate(df_raw, cfg)

cfg = cached_load_config()
forecast_cfg = cfg.get("forecast", {})
default_horizon = int(forecast_cfg.get("horizon", 6))
max_horizon = int(forecast_cfg.get("max_horizon", 24))

with st.sidebar:
    st.header("Pengaturan")
    horizon = st.number_input("Horizon Prediksi (bulan)", min_value=1, max_value=max_horizon, value=default_horizon)
    fast_mode = st.checkbox("Mode Cepat (skip XGBoost jika data kecil)", value=True)
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

    train_button = st.button("Latih / Retrain Model")
    predict_button = st.button("Prediksi ke Depan")

st.subheader("1) Unggah Data Historis")
uploaded = st.file_uploader("Unggah CSV (kolom: periode, permohonan_kia)", type=["csv"])
if uploaded is None:
    st.info("Unggah data untuk memulai.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}")
    st.stop()

try:
    df = cached_validate(df_raw, cfg)
except Exception as e:
    st.error(f"Validasi data gagal: {e}")
    st.stop()

st.success(f"Data OK. Jumlah periode: {df.shape[0]}")

dcol = cfg["data"]["date_column"]
ycol = cfg["data"]["target_column"]

fig_hist = px.line(df, x=dcol, y=ycol, title="Historis Permohonan KIA")
st.plotly_chart(fig_hist, use_container_width=True)

def data_hash(df):
    h = hashlib.md5()
    h.update(df[dcol].astype(str).str.cat(sep="|").encode())
    h.update(b"::")
    h.update(df[ycol].astype(str).str.cat(sep="|").encode())
    return h.hexdigest()

cur_hash = data_hash(df)

st.subheader("2) Pelatihan & Evaluasi")

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
                try:
                    st.metric(m, f"MAPE {sc['MAPE']:.2f}%", help=f"RMSE={sc['RMSE']:.2f}")
                except Exception:
                    st.write(m, sc)

# Load artifact dari disk jika diminta
if use_existing_model and ss.artifact is None:
    try:
        ss.artifact = load_artifact_file("models/kia_forecast")
        st.info(f"Model tersimpan: {ss.artifact['model_name']}")
    except Exception as e:
        st.warning(f"Gagal memuat model tersimpan: {e}")

# Train (retrain) hanya jika user klik tombol & hash beda / belum ada model
if train_button:
    if ss.artifact is not None and ss.data_hash == cur_hash:
        st.info("Dataset sama, model sudah ada. (Jika ingin paksa retrain, ubah sedikit data atau hapus model.)")
    else:
        with st.spinner("Melatih model..."):
            ss.artifact = train_pipeline(df, cfg, fast_mode=fast_mode)
            save_artifact(ss.artifact, out_dir="models", filename_prefix="kia_forecast")
            ss.data_hash = cur_hash
        st.success(f"Model '{ss.artifact['model_name']}' tersimpan.")

if ss.artifact:
    _show_scores(ss.artifact.get("scores", {}))
    with st.expander("Info Training"):
        st.write({
            "model_name": ss.artifact.get("model_name"),
            "cutoff_date": ss.artifact.get("cutoff_date"),
            "train_rows": ss.artifact.get("train_rows"),
            "test_rows": ss.artifact.get("test_rows"),
            "holdout_months": ss.artifact.get("holdout_months"),
            "blend_weight_final": ss.artifact.get("blend_weight_final"),
            "xgboost_error": ss.artifact.get("xgboost_error"),
            "fast_mode": ss.artifact.get("fast_mode"),
            "data_hash": ss.artifact.get("data_hash"),
        })

# Plot holdout
if ss.artifact and show_holdout_plot and "holdout_preds" in ss.artifact:
    try:
        holdout_dates = pd.to_datetime(ss.artifact["holdout_dates"])
        y_actual = ss.artifact["holdout_y_actual"]
        plot_df = pd.DataFrame({dcol: holdout_dates, "actual": y_actual})
        for mname, obj in ss.artifact["holdout_preds"].items():
            plot_df[mname] = obj["y_pred"]
        fig_holdout = px.line(plot_df, x=dcol, y=[c for c in plot_df.columns if c != dcol],
                              title="Holdout Comparison")
        st.plotly_chart(fig_holdout, use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal membuat plot holdout: {e}")

# APE
if ss.artifact and advanced_mode and show_ape_expander:
    with st.expander("Analisis APE"):
        try:
            y_true = np.array(ss.artifact["holdout_y_actual"])
            dates = pd.to_datetime(ss.artifact["holdout_dates"])
            data_ape = {"periode": dates, "actual": y_true}
            for mname, obj in ss.artifact["holdout_preds"].items():
                preds = np.array(obj["y_pred"])
                data_ape[mname] = preds
                ape = np.where(y_true != 0, np.abs((y_true - preds) / y_true) * 100, np.nan)
                data_ape[f"APE_{mname}"] = ape
            df_ape = pd.DataFrame(data_ape)
            st.dataframe(df_ape, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal analisis APE: {e}")

# Residual
if ss.artifact and advanced_mode and show_residual_analysis:
    with st.expander("Residual (Holdout)"):
        try:
            if "holdout_residuals" not in ss.artifact:
                st.info("Tidak ada residual.")
            else:
                dates = pd.to_datetime(ss.artifact["holdout_dates"])
                rows = []
                for m, res_list in ss.artifact["holdout_residuals"].items():
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
    if ss.artifact is None:
        st.error("Belum ada model. Latih model dulu.")
    else:
        try:
            artifact = ss.artifact
            model_name = artifact["model_name"]
            st.write(f"Model digunakan: {model_name}")
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
                st.caption(f"Blend weight: {w:.2f}")
            else:
                fc = forecast_iterative_naive(df_hist=df[[dcol, ycol]], cfg=cfg, horizon=horizon)
                fc_df = fc.rename(columns={"y_pred": "prediksi"})

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
                    st.warning("Residual model tidak tersedia untuk interval.")

            # Format
            fc_df["prediksi_fmt"] = fc_df["prediksi"].map(lambda v: f"{v:,.2f}")
            show_cols = [dcol, "prediksi_fmt"]
            if "lower" in fc_df.columns and "upper" in fc_df.columns:
                fc_df["lower_fmt"] = fc_df["lower"].map(lambda v: f"{v:,.2f}")
                fc_df["upper_fmt"] = fc_df["upper"].map(lambda v: f"{v:,.2f}")
                show_cols += ["lower_fmt", "upper_fmt"]
            st.dataframe(fc_df[show_cols], use_container_width=True)

            # Plot
            hist_plot = df[[dcol, ycol]].rename(columns={dcol: "periode", ycol: "aktual"})
            future_plot = fc_df.rename(columns={dcol: "periode"})
            future_plot["aktual"] = np.nan
            plot_df = pd.concat([hist_plot, future_plot], ignore_index=True)
            fig = px.line(plot_df, x="periode", y=["aktual", "prediksi"], title=f"Forecast ({model_name})")
            st.plotly_chart(fig, use_container_width=True)

            # Download
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
