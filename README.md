# Prediksi Kombinasi Angka — Markov2 + Jawa + Cina

Aplikasi Streamlit untuk memprediksi kombinasi angka (6-digit / 4-digit) menggunakan:
- Markov orde-2 built from historical 6-digit records
- Penyesuaian dengan Angka Samaran (ALIAS)
- Faktor Hari & Pasaran (neptu Jawa)
- Integrasi numerik dari sistem Tiongkok: Heavenly Stems / Earthly Branches, Lo Shu, I Ching (hexagram)
- Beam search untuk menghasilkan Top-K prediksi

## Cara pakai (singkat)
1. Siapkan `a.csv`, `b.csv`, `c.csv` di root repository (format bebas, tiap baris bisa multi-kolom; app akan mencari kolom paling kanan yang berisi angka di baris terbawah).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
