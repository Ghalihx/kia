# Sistem Prediksi Permohonan KIA  
Disdukcapil Kota Bogor – Internal

Dashboard Streamlit untuk:
- Memuat data historis permohonan KIA (bulanan)
- Melatih beberapa model baseline (naive, seasonal_naive) + XGBoost + (opsional) blend
- Mengevaluasi (MAPE, RMSE, holdout plot)
- Menghasilkan forecast ke depan (1–24 bulan)
- Mengunduh hasil prediksi (CSV)

> Catatan: Aplikasi ini untuk perencanaan internal. Tidak untuk publikasi eksternal tanpa verifikasi.

---

## 1. Struktur Folder (Saran)

```
.
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ models/
│  │  ├─ train.py
│  │  ├─ infer.py
│  │  └─ baselines.py
│  ├─ pipeline/
│  │  └─ data_prep.py
│  └─ utils/ (opsional)
├─ models/               # artifact tersimpan (output training)
├─ data/                 # contoh / staging CSV (jangan commit data sensitif)
├─ assets/               # logo / css (jika dipakai)
├─ requirements.txt
├─ config.yml / config.toml (jika ada)
└─ README.md
```

Jika saat ini struktur sedikit berbeda, sesuaikan bagian import di `streamlit_app.py`.

---

## 2. Persyaratan Lingkungan

- Python 3.9–3.11 (disarankan 3.10)
- Lihat paket di `requirements.txt`
- Untuk XGBoost di beberapa OS perlu compiler / wheel yang sesuai (di Streamlit Cloud biasanya langsung jalan)

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Format Data Input

File CSV minimal berisi kolom:

| Kolom          | Tipe        | Keterangan                          |
|----------------|-------------|-------------------------------------|
| periode        | string/tanggal (YYYY-MM) | Periode bulanan (disarankan format YYYY-MM atau tanggal awal bulan) |
| permohonan_kia | integer/float | Jumlah permohonan KIA pada periode itu |

Contoh (CSV):

```csv
periode,permohonan_kia
2021-01,512
2021-02,498
2021-03,530
...
```

Pastikan:
- Tidak ada duplikat periode
- Urutan kronologis (jika tidak, pipeline akan mengurutkan)
- Nilai negatif tidak diperbolehkan

---

## 4. Menjalankan Aplikasi

```bash
streamlit run app/streamlit_app.py
```

Buka URL yang ditampilkan (biasanya http://localhost:8501).

---

## 5. Alur Penggunaan

1. Jalankan aplikasi.
2. Unggah `CSV` historis.
3. (Otomatis / tombol) latih model → muncul metrik & holdout.
4. Tentukan horizon (misal 6 bulan) → klik "Prediksi ke Depan".
5. Unduh hasil forecast (CSV).

---

## 6. Model yang Didukung (Saat Ini)

| Model             | Deskripsi                                                                 | Catatan |
|-------------------|---------------------------------------------------------------------------|---------|
| naive             | Prediksi = nilai terakhir                                                 | Baseline 1 |
| seasonal_naive    | Prediksi = nilai periode sama musim sebelumnya (lag 12)                   | Perlu data ≥ 13 bulan |
| xgboost           | Tree boosting dengan fitur lag/kalender sederhana                         | Opsional (hapus paket jika tidak dipakai) |
| blend             | Kombinasi naive & seasonal (atau naive & xgboost) dengan bobot tertentu   | Bobot dapat disimpan di artifact |

> Jika XGBoost gagal diload (misal environment terbatas), pipeline fallback ke model baseline.

---

## 7. File Artifact

Direktori `models/` akan menyimpan file artifact (misal `.pkl` atau `.json`) berisi:
- model_name terpilih
- skor setiap model
- holdout preds
- parameter (misal blend weight)

Jangan commit artifact sensitif jika mengandung data asli.

---

## 8. Evaluasi & Metrik

- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Holdout: bagian akhir data (misal n bulan) dipisahkan untuk evaluasi

Rumus MAPE aman:

```
MAPE = mean( |y_true - y_pred| / y_true ) * 100
(dengan y_true == 0 → NaN / di-skip)
```

---

## 9. Konfigurasi

Jika ada `config.yml` atau `config.toml`, contoh isi minimal:

```toml
[data]
date_column = "periode"
target_column = "permohonan_kia"

[forecast]
horizon = 6
max_horizon = 24

[training]
holdout_months = 6
blend_weight = 0.5
```

Modul `load_config()` harus membaca file ini. Jika tidak ada file, sediakan default di kode.

---

## 10. Penanganan Error Umum

| Gejala | Penyebab Umum | Solusi |
|--------|---------------|--------|
| Tombol prediksi tidak aktif | Artifact belum terbentuk / state belum rerun | Latih model ulang atau refresh |
| MAPE NaN | Ada y_true = 0 | Filter / ganti 0 sementara (misal 0→eps) |
| XGBoost ImportError | Paket tidak terinstall / platform tidak mendukung | Hapus dependensi XGBoost di pipeline |
| Forecast melompat ekstrem | Outlier di ujung historis | Lakukan smoothing / median filter lokal |

---

## 11. Roadmap Peningkatan (Opsional)

- Interval kepercayaan sederhana (naive stdev)
- Fitur kalender (libur nasional)
- Logging metrik historis (JSON timeline)
- Auto retrain (GitHub Actions / cron)
- Ekspor PDF ringkasan
- Validasi tambahan (gap periode otomatis deteksi)

---

## 12. Kontribusi

1. Fork atau buat branch feature
2. Commit terpisah (satu fitur satu PR)
3. Sertakan deskripsi perubahan + screenshot (jika UI)
4. Pastikan lint dasar (opsional: jalankan `black`)

---

## 13. Lisensi

Tambahkan file `LICENSE` jika proyek akan dibuka lintas unit.  
Jika purely internal → bisa ditulis “Hak Cipta Disdukcapil Kota Bogor (Internal Use Only)”.

---

## 14. Kontak Internal

| Peran | Nama / Unit | Catatan |
|-------|-------------|---------|
| Pemilik Produk | (isi) | Prioritas fitur |
| Analis Data | (isi) | Retrain & evaluasi |
| Admin Teknis | (isi) | Deployment / server |

---

## 15. FAQ (Singkat)

**Q:** Kenapa setiap upload retrain otomatis?  
**A:** Saat artifact belum ada atau tombol latih ditekan, pipeline berjalan. Bisa dimodifikasi agar hanya manual.

**Q:** Kalau data kurang dari 12 bulan?  
**A:** Model seasonal_naive mungkin tidak valid; fallback ke naive.

**Q:** Bisa tambah model lain?  
**A:** Ya; tambahkan di `src/models/train.py` (buat fungsi fit & predict) dan integrasikan di pemilihan skor.

---

## 16. Changelog (Contoh)

| Versi | Tanggal | Perubahan Ringkas |
|-------|---------|-------------------|
| v13   | 2025-08 | Layout tabs + hero |
| v13.1 | (isi)   | (Misal) tambah toggles |
| v14   | (isi)   | (Eksperimen step-by-step) |

---

Selamat menggunakan & mengembangkan lebih lanjut.