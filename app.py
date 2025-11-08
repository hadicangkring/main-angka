# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import product

# === SETUP PAGE ===
st.set_page_config(page_title="🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan Kalender Cina")

# === PENGATURAN ===
st.sidebar.header("⚙️ Pengaturan")
beam_width = st.sidebar.number_input("Beam Width", min_value=1, max_value=50, value=3)
top_k = st.sidebar.number_input("Top-K Prediksi", min_value=1, max_value=20, value=10)
laplace_alpha = st.sidebar.number_input("Laplace α", min_value=0.1, max_value=5.0, value=1.0)

# === INFORMASI HARI & PASARAN ===
hari = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
pasaran = st.selectbox("Pasaran", ["Legi", "Pahing", "Pon", "Wage", "Kliwon"])
st.markdown(f"📅 Pilihan: {hari} {pasaran}")

# === LOAD FILE DATA ===
def load_file(file_name):
    try:
        df = pd.read_csv(file_name, header=None)
        return df
    except Exception as e:
        st.warning(f"Gagal memuat {file_name}: {e}")
        return pd.DataFrame()

file_a = load_file("a.csv")
file_b = load_file("b.csv")
file_c = load_file("c.csv")

# === CEK DATA ===
if file_a.empty and file_b.empty and file_c.empty:
    st.error("Belum ada data valid dari File A/B/C.")
else:
    st.success("File data berhasil dimuat!")

# === BANGUN MODEL MARKOV ORDE-2 ===
def build_markov_model(df_list, alpha=1.0):
    model = defaultdict(lambda: defaultdict(float))
    for df in df_list:
        if df.empty:
            continue
        for row in df.values:
            for i in range(len(row)-2):
                prev_pair = (row[i], row[i+1])
                next_num = row[i+2]
                model[prev_pair][next_num] += 1
    # Laplace smoothing
    for prev_pair, counter in model.items():
        total = sum(counter.values()) + alpha * len(counter)
        for k in counter:
            counter[k] = (counter[k] + alpha) / total
    return model

markov_model = build_markov_model([file_a, file_b, file_c], alpha=laplace_alpha)

# === FUNGSIONAL PREDIKSI ===
def predict_next(prev1, prev2, model, top_k=5):
    key = (prev1, prev2)
    if key in model:
        counter = model[key]
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    else:
        return []

# === INPUT DARI USER ===
st.subheader("Masukkan dua angka terakhir (untuk prediksi)")
col1, col2 = st.columns(2)
with col1:
    last1 = st.number_input("Angka terakhir ke-1", min_value=0, max_value=99, value=0)
with col2:
    last2 = st.number_input("Angka terakhir ke-2", min_value=0, max_value=99, value=0)

if st.button("Prediksi"):
    preds = predict_next(last1, last2, markov_model, top_k=top_k)
    if preds:
        st.subheader("🔮 Prediksi Top-{}".format(top_k))
        for i, (num, prob) in enumerate(preds, 1):
            st.write(f"{i}. {num} → Probabilitas: {prob*100:.2f}%")
    else:
        st.info("Tidak ada prediksi untuk kombinasi ini.")

# === INFO TAMBAHAN ===
st.markdown("---")
st.markdown("ℹ️ Sistem ini menggunakan **Markov Orde-2** untuk memprediksi angka berikutnya berdasarkan dua angka terakhir. Integrasi dengan Hari, Pasaran, dan Kalender Cina sedang dalam pengembangan.")
