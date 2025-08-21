# Sistem Prediksi Permohonan Kartu Identitas Anak (KIA) 
Disdukcapil Kota Bogor dan Universitas Djuanda Bogor

Aplikasi ini membantu memprediksi jumlah Permohonan Kartu Identitas Anak (KIA) beberapa bulan ke depan menggunakan kombinasi model time series sederhana (naive, seasonal naive), model XGBoost, dan model blend (kombinasi teroptimasi dari dua model terbaik). Antarmuka disajikan melalui Streamlit agar mudah digunakan oleh pengguna non-teknis.

---

## Fitur Utama
- Upload data historis (format sederhana: periode, permohonan_kia)
- Validasi dasar (format tanggal, nilai numerik)
- Pelatihan cepat dengan holdout (default 6 bulan terakhir)
- Evaluasi multi-model: naive, seasonal_naive, xgboost, blend
- Pemilihan otomatis model terbaik berbasis MAPE terendah
- Opsional: Analisis APE dan residual (Mode Advanced)
- Interval prediksi (opsional – muncul di tabel, bukan grafik)
- Penyimpanan artefak model (versi timestamp + latest)
- Pencatatan riwayat pelatihan ke file model_history.csv
- Ekspor hasil prediksi ke CSV

---

## Struktur Proyek

```
.
├─ app/
│  └─ streamlit_app.py
├─ config.toml
├─ data/
│  ├─ README.md
│  └─ sample_kia.csv
├─ models/                # Artefak model (*.pkl) & model_history.csv (setelah training)
├─ src/
│  ├─ models/
│  │  ├─ baselines.py
│  │  ├─ infer.py
│  │  └─ train.py
│  ├─ pipeline/
│  │  └─ data_prep.py
│  └─ utils/
│     └─ intervals.py     # (opsional; hanya jika ingin interval)
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## Persyaratan
- Python 3.10+ (disarankan)
- Paket pada requirements.txt (xgboost, streamlit, pandas, numpy, plotly, toml, dll.)

### Pastikan Masuk Ke Folder Kia ( C:\Users\USER\kia> )

Install:
```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
## Menjalankan Aplikasi

```bash
streamlit run app/streamlit_app.py 
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```
---

## Format Data Input

CSV minimal berisi dua kolom:
| Kolom | Deskripsi | Contoh |
|-------|-----------|--------|
| periode | Tanggal awal bulan (YYYY-MM atau YYYY-MM-DD) | 2023-01-01 |
| permohonan_kia | Jumlah permohonan (integer / float) | 1250 |

Contoh (sample_kia.csv):
```csv
periode,permohonan_kia
2021-01-01,980
2021-02-01,1005
2021-03-01,995
...
```

Catatan:
- Tidak boleh ada duplikat periode
- Tidak boleh ada bulan hilang di tengah (gap)
- Nilai harus numerik (akan divalidasi)

---

## Konfigurasi (config.toml)

Contoh ringkas:
```toml
[data]
date_column = "periode"
target_column = "permohonan_kia"
lags = [1,2,3,12]
rollings = [3,6,12]
add_sin_cos = true

[training]
holdout_months = 6
season_length = 12

[forecast]
horizon = 6
max_horizon = 24
```

Penjelasan:
- holdout_months: jumlah bulan terakhir dipakai evaluasi
- season_length: panjang musim (12 = tahunan)
- horizon: default jumlah bulan ke depan untuk prediksi

---

Langkah di UI:
1. Unggah CSV historis
2. Klik "Latih Model" (atau gunakan model tersimpan)
3. Lihat skor model (MAPE & RMSE)
4. Klik "Prediksi ke Depan" untuk forecast horizon tertentu
5. (Opsional) Aktifkan Mode Advanced untuk lihat APE / residual / interval
6. Unduh hasil CSV

---

## Metodologi Pelatihan

1. Data diurutkan kronologis
2. Split: (Total - holdout_months) sebagai train, sisanya holdout
3. Model yang dihitung:
   - naive: ŷ_t = y_{t-1}
   - seasonal_naive: ŷ_t = y_{t-12} (fallback ke y_{t-1} bila tidak ada)
   - xgboost: regresi fitur (lag, rolling, sin/cos bulan, diff)
   - blend: kombinasi w * xgboost + (1 - w) * seasonal_naive (w dicari grid 0..1 step 0.1)
4. Skor dihitung (MAPE & RMSE) pada holdout
5. Model terbaik = MAPE terendah (dengan logika override kecil bila naive menang tipis)
6. Artefak disimpan sebagai:
   - models/kia_forecast_<timestamp>.pkl
   - models/kia_forecast_latest.pkl

---

## Metrik

- MAPE (%): Rata-rata kesalahan relatif. Cocok untuk interpretasi persentase.
- RMSE: Akar rata-rata kuadrat error (memberi penalti tinggi pada kesalahan besar).

Interpretasi: semakin kecil kedua nilai → semakin baik.

---

## Artefak Model (Isi Utama)
Contoh kunci dalam artefak .pkl:
```text
model_name            # nama model terpilih ('blend', 'xgboost', dll.)
scores                # dict: {model: {MAPE, RMSE}}
holdout_dates         # daftar tanggal holdout
holdout_y_actual      # nilai aktual holdout
holdout_preds         # prediksi tiap model
holdout_residuals     # residual tiap model
blend_weight_final    # bobot w hasil optimasi (jika model blend)
xgb_model_raw         # model XGBoost terserialisasi (string)
feature_columns       # kolom fitur yang dipakai XGBoost
cutoff_date           # tanggal terakhir data train
schema_version        # versi struktur artefak
```

---

## Riwayat Pelatihan
File: models/model_history.csv  
Tiap pelatihan append baris:
```
timestamp,model_name,blend_weight,MAPE_naive,MAPE_seasonal,MAPE_xgb,MAPE_blend,RMSE_naive,RMSE_seasonal,RMSE_xgb,RMSE_blend
```
Gunakan untuk memantau drift performa antar re-train.

---

## Interval Prediksi (Opsional)
- Menggunakan residual holdout model terpilih
- Metode: quantile (empirical) atau normal (mean ± z * std)
- Tidak selalu stabil jika residual sedikit (< 6–8 data)
- Ditampilkan hanya di tabel (lower, upper)

---

## Mode Advanced
Aktif bila dicentang:
- Analisis APE (Absolute Percentage Error) per bulan holdout
- Residual holdout semua model
- Interval prediksi (jika residual tersedia)

---

## Prosedur Re-Train Rutin
1. Tambahkan baris bulan terbaru ke data historis
2. Jalankan ulang aplikasi
3. Upload data terbaru
4. Klik Latih Model
5. Bandingkan skor baru dengan model_history.csv
6. Jika MAPE naik drastis → investigasi (data anomali / pola baru)

---

## Ekspor Forecast
Tombol "Download Forecast CSV" menghasilkan file (dua desimal), contoh:
```csv
periode,prediksi,lower,upper
2025-07-01,1340.22,1295.10,1389.55
...
```
Kolom lower/upper hanya ada jika interval diaktifkan.

---

## Menambahkan Fitur Baru (Opsional)
- Tambah lag lain di config.toml (misal 6, 9)
- Tambah rolling lain (24) jika historis cukup panjang
- Tambah fitur event manual (misal bulan_ramadhan=1/0) di pipeline sebelum train
- Regenerasi model → cek apakah MAPE turun

---

## Troubleshooting

| Gejala | Penyebab Mungkin | Solusi |
|--------|------------------|--------|
| Error "Data terlalu sedikit" | Histori < holdout + kebutuhan lag | Kurangi holdout atau kumpulkan data tambahan |
| MAPE tiba-tiba tinggi | Outlier / data salah input | Validasi ulang nilai bulan baru |
| XGBoost gagal | Paket tidak terinstal / versi tak cocok | pip install xgboost, cek log artifact["xgboost_error"] |
| Interval kosong | Residual kurang | Tambah siklus re-train dulu |
| Semua forecast flat | Model naive terpilih | Pastikan seasonal dengan cukup data 12+ bulan atau gunakan blend |

---

## Kontak / Pemelihara
- ghalihputra8@gmail.com

---

Selamat menggunakan!



