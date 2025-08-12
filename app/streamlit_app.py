import sys
from pathlib import Path
import hashlib
import threading
import time
from typing import Optional, Callable, Dict, Any

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

# ---------- SESSION STATE INIT ----------
ss = st.session_state
if "artifact" not in ss:            ss.artifact = None
if "is_training" not in ss:         ss.is_training = False
if "train_progress" not in ss:      ss.train_progress = 0.0
if "train_status_text" not in ss:   ss.train_status_text = ""
if "last_train_time" not in ss:     ss.last_train_time = None
if "data_hash" not in ss:           ss.data_hash = None
if "train_thread" not in ss:        ss.train_thread = None
if "train_error" not in ss:         ss.train_error = None

st.set_page_config(page_title="Prediksi Permohonan KIA", layout="wide")
st.title("Sistem Prediksi Permohonan KIA - Disdukcapil Kota Bogor (Refactored)")

# ---------- CACHING FUNGSI RINGAN ----------
@st.cache_data(show_spinner=False)
def cached_load_config():
    return load_config()

@st.cache_data(show_spinner=False)
def cached_validate(df_raw: pd.DataFrame, cfg: dict):
    return load_and_validate(df_raw, cfg)

cfg = cached_load_config()
forecast_cfg = cfg.get("forecast", {})
default_horizon = int(forecast_cfg.get("horizon", 6))
max_horizon = int(forecast_cfg.get("max_horizon", 24))

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Pengaturan")
    horizon = st.number_input("Horizon Prediksi (bulan)", min_value=1, max_value=max_horizon, value=default_horizon)
    use_existing_model = st.checkbox("Gunakan model tersimpan (jika ada)", value=False)
    show_details = st.checkbox("Detail skor lengkap", value=False)
    show_holdout_plot = st.checkbox("Plot holdout", value=True)

    st.markdown("---")
    advanced_mode = st.checkbox("Mode Advanced", value=False)
    if advanced_mode:
        show_ape_expander = st.checkbox("Tampilkan Analisis APE", value=True)
        show_residual_analysis = st.checkbox("Tampilkan Residual", value=False)
        show_prediction_intervals = st.checkbox("Interval Prediksi", value=False, disabled=not HAS_INTERVALS)
        interval_method = st.selectbox(
            "Metode Interval",
            ["quantile", "normal"],
            disabled=not show_prediction_intervals or not HAS_INTERVALS
        )
        interval_alpha = st.selectbox(
            "Alpha (1 - confidence)",
            [0.01, 0.025, 0.05, 0.10],
            index=2,
            disabled=not show_prediction_intervals or not HAS_INTERVALS
        )
    else:
        show_ape_expander = False
        show_residual_analysis = False
        show_prediction_intervals = False
        interval_method = "quantile"
        interval_alpha = 0.05

    st.markdown("---")
    train_button = st.button("Latih / Retrain Model", disabled=ss.is_training)
    predict_button = st.button("Prediksi ke Depan", disabled=ss.is_training)

st.subheader("1) Unggah Data Historis")
uploaded = st.file_uploader("Unggah CSV (kolom: periode, permohonan_kia)", type=["csv"])

if uploaded is None:
    st.info("Unggah data untuk memulai.")
    st.stop()

# ---------- BACA & VALIDASI DATA ----------
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

# ---------- UTIL ----------
def hash_series(series: pd.Series) -> str:
    # Hash sederhana berdasarkan nilai target (bisa diperluas dgn tanggal)
    h = hashlib.md5(series.astype(str).str.cat(sep="|").encode())
    return h.hexdigest()

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

def load_existing_if_requested():
    if use_existing_model:
        try:
            art = load_artifact("models/kia_forecast")
            return art
        except Exception as e:
            st.warning(f"Gagal memuat model tersimpan: {e}")
    return None

# ---------- TRAINING ASINKRON ----------
def training_callback(event: str, info: Dict[str, Any]):
    # Callback dipanggil oleh train_pipeline (per epoch / tahap)
    # event bisa: 'init', 'progress', 'phase', 'done'
    if event == "progress":
        # info: {'current': int, 'total': int, 'label': str}
        cur = info.get("current", 0)
        tot = max(info.get("total", 1), 1)
        ss.train_progress = cur / tot
        ss.train_status_text = info.get("label", f"Epoch {cur}/{tot}")
    elif event == "phase":
        ss.train_status_text = info.get("label", "")
    elif event == "done":
        ss.train_progress = 1.0
        ss.train_status_text = "Selesai"
    elif event == "error":
        ss.train_status_text = f"Error: {info.get('message')}"

def run_training_thread(df_in: pd.DataFrame, cfg_in: dict, data_hash: str):
    try:
        start = time.time()
        artifact_local = train_pipeline(df_in, cfg_in, progress_callback=training_callback)  # <== perlu kamu modifikasi di train_pipeline agar menerima arg progress_callback
        # Simpan metadata tambahan
        artifact_local["data_hash"] = data_hash
        save_artifact(artifact_local, out_dir="models", filename_prefix="kia_forecast")
        ss.artifact = artifact_local
        ss.last_train_time = round(time.time() - start, 2)
        training_callback("done", {})
    except Exception as e:
        ss.train_error = str(e)
        training_callback("error", {"message": str(e)})
    finally:
        ss.is_training = False

# ---------- BAGIAN 2: PELATIHAN & EVALUASI ----------
st.subheader("2) Pelatihan & Evaluasi")

# Load existing jika user minta & belum ada di session
if ss.artifact is None and use_existing_model:
    existing_art = load_existing_if_requested()
    if existing_art:
        ss.artifact = existing_art

# TOMBOL TRAIN
current_hash = hash_series(df[ycol])
need_retrain_reason = None
if ss.artifact is not None:
    # Cek data berubah?
    art_hash = ss.artifact.get("data_hash")
    if art_hash and art_hash != current_hash:
        need_retrain_reason = "Data berubah (hash berbeda)."
else:
    need_retrain_reason = "Belum ada model di sesi."

if train_button:
    if ss.is_training:
        st.warning("Training sedang berjalan.")
    else:
        # Putuskan apakah perlu training ulang (kalau data sama & user hanya iseng menekan tombol bisa skip)
        if ss.artifact and need_retrain_reason is None:
            st.info("Data sama dan model sudah ada. Tidak retrain. (Tekan lagi sambil tahan Shift jika ingin paksa retrain).")
        else:
            ss.is_training = True
            ss.train_progress = 0.0
            ss.train_status_text = "Inisialisasi..."
            ss.train_error = None
            ss.data_hash = current_hash
            # Mulai thread
            th = threading.Thread(target=run_training_thread, args=(df, cfg, current_hash), daemon=True)
            ss.train_thread = th
            th.start()

# STATUS TRAINING
if ss.is_training:
    st.info("Training berjalan di background:")
    st.progress(ss.train_progress)
    st.write(ss.train_status_text)
    st.stop()  # Tahan UI lain dulu agar user tahu sedang jalan

# Jika ada error training
if ss.train_error:
    st.error(f"Training gagal: {ss.train_error}")

# Tampilkan hasil model
if ss.artifact:
    st.success(f"Model aktif: {ss.artifact.get('model_name')} (latency training: {ss.last_train_time}s)" if ss.last_train_time else f"Model aktif: {ss.artifact.get('model_name')}")
    if need_retrain_reason:
        st.caption(f"Catatan: {need_retrain_reason}")
    _show_scores(ss.artifact.get("scores", {}))
    with st.expander("Info Training / Meta"):
        st.write({
            "model_name": ss.artifact.get("model_name"),
            "cutoff_date": ss.artifact.get("cutoff_date"),
            "holdout_months": ss.artifact.get("holdout_months"),
            "train_rows": ss.artifact.get("train_rows"),
            "test_rows": ss.artifact.get("test_rows"),
            "blend_weight_final": ss.artifact.get("blend_weight_final"),
            "xgboost_error": ss.artifact.get("xgboost_error"),
            "data_hash": ss.artifact.get("data_hash"),
        })

# Plot holdout
if ss.artifact and show_holdout_plot and "holdout_preds" in ss.artifact:
    st.markdown("### Holdout: Actual vs Prediksi")
    try:
        holdout_dates = pd.to_datetime(ss.artifact["holdout_dates"])
        y_actual = ss.artifact["holdout_y_actual"]
        plot_df = pd.DataFrame({dcol: holdout_dates, "actual": y_actual})
        for mname, obj in ss.artifact["holdout_preds"].items():
            plot_df[mname] = obj["y_pred"]
        fig_holdout = px.line(plot_df, x=dcol,
                              y=[c for c in plot_df.columns if c != dcol],
                              title="Holdout Comparison")
        st.plotly_chart(fig_holdout, use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal membuat plot holdout: {e}")

# APE analysis
if ss.artifact and advanced_mode and show_ape_expander:
    with st.expander("Analisis APE"):
        try:
            y_true = np.array(ss.artifact["holdout_y_actual"])
            dates = pd.to_datetime(ss.artifact["holdout_dates"])
            data = {"periode": dates, "actual": y_true}
            for mname, obj in ss.artifact["holdout_preds"].items():
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

# ---------- PREDIKSI KE DEPAN ----------
st.subheader("3) Prediksi ke Depan")
download_slot = st.empty()

if predict_button:
    if ss.artifact is None:
        st.error("Belum ada model. Latih model terlebih dahulu.")
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

            hist_plot = df[[dcol, ycol]].rename(columns={dcol: "periode", ycol: "aktual"})
            future_plot = fc_df.rename(columns={dcol: "periode"})
            future_plot["aktual"] = np.nan
            plot_df = pd.concat([hist_plot, future_plot], ignore_index=True)
            fig = px.line(plot_df, x="periode", y=["aktual", "prediksi"], title=f"Forecast ({model_name})")
            st.plotly_chart(fig, use_container_width=True)

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
