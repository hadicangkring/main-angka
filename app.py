# app.py
# ===============================================
# 🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar
# Model Markov Orde-2 dengan integrasi Hari, Pasaran,
# dan dukungan pembacaan File C (dua blok mingguan)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import io, re
from datetime import datetime
from collections import defaultdict, Counter

# === SETUP PAGE ===
st.set_page_config(page_title="Fusion China & Jawa Calendar", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan Kalender Cina (rencana pengembangan).")

# === PARAMETER ===
st.sidebar.header("⚙️ Pengaturan")
beam_width = st.sidebar.slider("Beam Width", 3, 50, 10)
topk = st.sidebar.slider("Top-K Prediksi", 1, 10, 5)
laplace_alpha = st.sidebar.number_input("Laplace α", 0.1, 5.0, 1.0, 0.1)

# === KALENDER JAWA SEDERHANA ===
def hari_pasaran(date: datetime):
    hari_list = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]
    pasaran_list = ["Legi","Pahing","Pon","Wage","Kliwon"]
    ref = datetime(2020,1,1)  # acuan Rabu Legi
    delta = (date - ref).days
    hari = hari_list[date.weekday()]
    pasaran = pasaran_list[delta % 5]
    neptu = (date.weekday()+1) + [5,9,7,4,8][delta % 5]
    return hari, pasaran, neptu

today = datetime.now()
hari, pasaran, neptu = hari_pasaran(today)
st.markdown(f"📅 **{hari} {pasaran}** (Neptu {neptu})")

# === Fungsi bantu File C ===
def process_file_c(uploaded_file):
    """Membaca file C berformat dua blok mingguan (lokal & Data HK Lotto)."""
    if uploaded_file is None:
        return None, "Tidak ada file diunggah."

    try:
        text = uploaded_file.getvalue().decode("utf-8")
    except Exception:
        try:
            text = uploaded_file.read().decode("utf-8")
        except:
            return None, "Gagal membaca file."

    parts = re.split(r"Data\s+HK\s+Lotto", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None, "Format file tidak sesuai. Tidak ditemukan pemisah 'Data HK Lotto'."

    def parse_block(block_text):
        block_text = block_text.strip()
        lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
        # buang baris header yang berawalan huruf
        lines = [ln for ln in lines if not re.match(r"^[A-Z]", ln)]
        data = []
        for ln in lines:
            row = re.split(r"\s+", ln)
            if len(row) == 7:
                data.append([x if x != "–" else None for x in row])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["SENIN","SELASA","RABU","KAMIS","JUMAT","SABTU","MINGGU"])
        for col in df.columns:
            df[col] = (df[col].astype(str)
                                .str.replace("–","")
                                .replace({"nan":"","None":""})
                                .replace("", pd.NA))
        return df

    df_local = parse_block(parts[0])
    df_hk = parse_block(parts[1])
    df_all = pd.concat([df_local, df_hk], ignore_index=True)
    return df_all, f"Berhasil membaca data: {len(df_all)} baris (lokal {len(df_local)}, HK {len(df_hk)})"

# === Fungsi bantu umum (File A/B biasa) ===
def load_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep="\t")
            return df
        except Exception:
            return None

# === UPLOAD FILE ===
st.subheader("📘 File A")
file_a = st.file_uploader("Unggah File A (.csv atau .tsv)", type=["csv","txt"])
df_a = load_csv(file_a)
if df_a is not None:
    st.success(f"Data A berhasil dibaca ({len(df_a)} baris)")
else:
    st.info("Tidak ada file A valid.")

st.subheader("📗 File B")
file_b = st.file_uploader("Unggah File B (.csv atau .tsv)", type=["csv","txt"])
df_b = load_csv(file_b)
if df_b is not None:
    st.success(f"Data B berhasil dibaca ({len(df_b)} baris)")
else:
    st.info("Tidak ada file B valid.")

st.subheader("📙 File C")
file_c = st.file_uploader("Unggah File C (format dua blok mingguan)", type=["txt","csv"])
if file_c:
    df_c, info_c = process_file_c(file_c)
    if df_c is not None:
        st.success(info_c)
        st.dataframe(df_c.head())
    else:
        st.error(info_c)
else:
    st.info("Belum ada file C diunggah.")

# === GABUNGKAN SEMUA DATA ===
data_sources = [d for d in [df_a, df_b, 'df_c' in locals() and df_c] if isinstance(d, pd.DataFrame)]
if not data_sources:
    st.warning("Belum ada data valid dari file A/B/C.")
    st.stop()

data_all = pd.concat(data_sources, ignore_index=True)

# === MODEL MARKOV ORDE-2 ===
def build_markov(data_series):
    transitions = defaultdict(Counter)
    prev1, prev2 = None, None
    for num in data_series:
        if num and len(num) >= 2:
            if prev1 and prev2:
                transitions[(prev1[-1], prev2[-1])][num[-1]] += 1
            prev1, prev2 = prev2, num
    return transitions

flat_series = []
for _, row in data_all.iterrows():
    flat_series.extend([x for x in row.values if pd.notna(x)])
model = build_markov(flat_series)

# === PREDIKSI ===
def predict_next(model, last_two):
    key = (last_two[-2], last_two[-1]) if len(last_two) >= 2 else None
    if not key or key not in model:
        return []
    probs = model[key]
    total = sum(probs.values()) + laplace_alpha * 10
    dist = {k:(v+laplace_alpha)/total for k,v in probs.items()}
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:topk]

if flat_series:
    last_two = flat_series[-2:]
    preds = predict_next(model, last_two)
    st.subheader("🔮 Prediksi Angka Berikutnya")
    if preds:
        for p,v in preds:
            st.write(f"**...{p}** → Probabilitas ≈ {v:.2%}")
    else:
        st.info("Model belum punya riwayat cukup untuk prediksi.")
else:
    st.info("Tidak ada data angka valid untuk model.")

st.success("✅ Analisis selesai — Model Markov Orde-2 siap.")
