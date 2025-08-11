# Format Data KIA

Gunakan file CSV dengan dua kolom wajib:
- periode: tanggal awal bulan. Contoh valid:
  - 2024-01
  - 2024-01-01
- permohonan_kia: integer ≥ 0

Contoh minimal isi:
```csv
periode,permohonan_kia
2024-01-01,1200
2024-02-01,1150
```

Tips:
- Pastikan tidak ada duplikasi bulan.
- Nilai kosong/NaN tidak diperbolehkan.
- Jika ada outlier ekstrem, pertimbangkan verifikasi sumber data.
