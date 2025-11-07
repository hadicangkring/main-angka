# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# === SETUP PAGE ===
st.set_page_config(page_title="🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan rencana Kalender Cina.")

# === PARAMETER MANUAL ===
alpha = st.slider("Laplace α", 0.0, 2.0, 1.0, 0.1)
beam_width = st.slider("Beam Width", 3, 50, 10, 1)
top_k = st.slider("Top-K Prediksi", 1, 10, 5, 1)

# === KONVERSI HARI JAWA ===
def hari_jawa(tanggal):
    hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    pasaran = ["Legi", "Pahing", "Pon", "Wage", "Kliwon"]
    neptu_hari = [4, 3, 7, 8, 6, 9, 5]
    neptu_pasaran = [5, 9, 7, 4, 8]

    idx_hari = tanggal.weekday()  # 0=Senin
    idx_pasaran = (tanggal.toordinal() + 3) % 5
    return f"{hari[idx_hari]} {pasaran[idx_pasaran]}", neptu_hari[idx_hari] + neptu_pasaran[idx_pasaran]

today = datetime.now()
hari_pasaran, neptu = hari_jawa(today)
st.markdown(f"📅 **{hari_pasaran} (Neptu {neptu})**")

# === BACA DATA ===
def baca_data(file_name):
    try:
        df = pd.read_csv(file_name, header=None)
        df = df.dropna(how="all")
        if df.empty:
            return None
        data = []
        for val in df.values.flatten():
            if isinstance(val, str) and val.strip() != "":
                val = val.strip().replace(",", "")
                for part in val.split():
                    if part.isdigit():
                        data.append(part.zfill(6))
        return data
    except Exception:
        return None

# === MODEL MARKOV ORDE 2 ===
def markov_order2_predict(data, top_k=5, alpha=1.0, beam_width=10):
    if not data or len(data) < 3:
        return []

    # pecah ke dalam digit
    sequences = [list(x) for x in data]
    transitions = {}

    for seq in sequences:
        for i in range(len(seq) - 2):
            key = (seq[i], seq[i+1])
            next_digit = seq[i+2]
            if key not in transitions:
                transitions[key] = {}
            transitions[key][next_digit] = transitions[key].get(next_digit, 0) + 1

    for k in transitions:
        total = sum(transitions[k].values()) + 10 * alpha
        for d in map(str, range(10)):
            transitions[k][d] = (transitions[k].get(d, 0) + alpha) / total

    last = list(data[-1])
    state = (last[-2], last[-1])
    preds = []

    for _ in range(top_k):
        seq = last[-4:]  # ambil 4 digit terakhir
        for _ in range(2):  # prediksi 2 langkah
            next_probs = transitions.get(state, None)
            if not next_probs:
                break
            next_digit = max(next_probs, key=next_probs.get)
            seq.append(next_digit)
            state = (state[1], next_digit)
        preds.append("".join(seq[-4:]))

    return list(dict.fromkeys(preds))[:top_k]

# === TAMPILKAN HASIL ===
def tampilkan_prediksi(file_name, label, emoji):
    data = baca_data(file_name)
    st.subheader(f"{emoji} {label}")

    if not data:
        st.text("Tidak ada data valid.")
        return

    last_num = data[-1]
    st.write(f"Angka terakhir sebelum prediksi adalah: **{last_num}**")

    pred4 = markov_order2_predict(data, top_k=top_k, alpha=alpha, beam_width=beam_width)
    pred2 = [x[-2:] for x in pred4]

    st.markdown("**Prediksi 4 Digit (Top 5):**")
    st.write(", ".join(pred4))

    st.markdown("**Prediksi 2 Digit (Top 5):**")
    st.write(", ".join(pred2))


# === JALANKAN UNTUK SETIAP FILE ===
tampilkan_prediksi("a.csv", "File A", "📘")
tampilkan_prediksi("b.csv", "File B", "📗")
tampilkan_prediksi("c.csv", "File C", "📙")

# === PREDIKSI GABUNGAN ===
st.subheader("🧩 Gabungan Semua Data")

data_a = baca_data("a.csv") or []
data_b = baca_data("b.csv") or []
data_c = baca_data("c.csv") or []

gabungan = data_a + data_b + data_c

if gabungan:
    st.write(f"Angka terakhir sebelum prediksi gabungan: **{gabungan[-1]}**")
    pred4_gab = markov_order2_predict(gabungan, top_k=top_k, alpha=alpha, beam_width=beam_width)
    pred2_gab = [x[-2:] for x in pred4_gab]

    st.markdown("**Prediksi 4 Digit (Top 5 Gabungan):**")
    st.write(", ".join(pred4_gab))

    st.markdown("**Prediksi 2 Digit (Top 5 Gabungan):**")
    st.write(", ".join(pred2_gab))
else:
    st.text("Belum ada data valid dari file A/B/C.")
