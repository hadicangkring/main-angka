# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import json
import math

# === SETUP PAGE ===
st.set_page_config(page_title="🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan rencana Kalender Cina.")

# === DATA SOURCE (GitHub raw folder) ===
BASE_URL = "https://raw.githubusercontent.com/hadicangkring/akurat/main/data/"
FILE_MAP = {
    "a": ("a.csv", "📘 File A"),
    "b": ("b.csv", "📗 File B"),
    "c": ("c.csv", "📙 File C"),
}

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
    # agar pasaran bergeser semirip kalender Jawa, pakai ordinal + offset (offset dipilih agar sinkron)
    idx_pasaran = (tanggal.toordinal() + 3) % 5
    return f"{hari[idx_hari]} {pasaran[idx_pasaran]}", neptu_hari[idx_hari] + neptu_pasaran[idx_pasaran]

today = datetime.now()
hari_pasaran, neptu = hari_jawa(today)
st.markdown(f"📅 **{hari_pasaran} (Neptu {neptu})**")

# === UTIL: baca data (mendukung URL dari GitHub raw) ===
def baca_data(file_key_or_path):
    """
    file_key_or_path: bisa 'a'/'b'/'c' atau nama file lokal/URL.
    Mengembalikan list of 6-digit strings (zfill).
    """
    # jika user memberi key 'a','b','c', ubah ke URL
    if isinstance(file_key_or_path, str) and file_key_or_path in FILE_MAP:
        filename = FILE_MAP[file_key_or_path][0]
        path = BASE_URL + filename
    else:
        path = file_key_or_path

    try:
        df = pd.read_csv(path, header=None, dtype=str)
        df = df.dropna(how="all")
        if df.empty:
            return []
        data = []
        for val in df.values.flatten():
            if isinstance(val, str) and val.strip() != "":
                v = val.strip().replace(",", "")
                # bagi berdasarkan whitespace, ambil bagian yang sepenuhnya digit
                for part in v.split():
                    # ambil digit saja (jika ada gabungan, ambil angka di dalamnya)
                    digits = "".join(ch for ch in part if ch.isdigit())
                    if digits != "":
                        data.append(digits.zfill(6)[-6:])
        return data
    except Exception:
        # jika gagal baca (mis. 404), kembalikan list kosong
        return []

# === MODEL MARKOV ORDE 2 (sejalan dengan skrip lama) ===
def markov_order2_predict(data, top_k=5, alpha=1.0, beam_width=10):
    """
    Menghasilkan daftar prediksi 4-digit (sebagai string) berdasarkan model Markov orde-2.
    Implementasi mempertahankan prinsip sederhana dari skrip asli.
    """
    if not data or len(data) < 3:
        return []

    # build counts for (a,b) -> c
    counts = defaultdict(Counter)
    for s in data:
        s6 = str(s).zfill(6)
        digits = list(s6)
        for i in range(len(digits) - 2):
            a,b,c = digits[i], digits[i+1], digits[i+2]
            counts[(a,b)][c] += 1

    # convert to conditional probs with Laplace smoothing
    cond_probs = {}
    for key, counter in counts.items():
        total = sum(counter.values()) + 10 * alpha
        probs = {}
        for d in map(str, range(10)):
            probs[d] = (counter.get(d, 0) + alpha) / total
        cond_probs[key] = probs

    # fallback uniform unigram if needed
    total_counts = Counter()
    for c in counts.values():
        total_counts.update(c)
    total_all = sum(total_counts.values()) or 1
    unigram = {str(d): (total_counts.get(str(d), 0) / total_all) for d in range(10)}
    if sum(unigram.values()) == 0:
        unigram = {str(d): 1.0/10.0 for d in range(10)}

    # beam-like greedy generation (preserve spirit skrip asli)
    last = list(data[-1].zfill(6))
    start_pair = (last[-2], last[-1])
    beams = [( "".join(start_pair), 0.0 )]  # seq of last two, score as log-prob (relative)
    steps = 2  # menghasilkan 2 langkah (untuk 4-digit prediksi terakhir)

    for _ in range(steps):
        new_beams = []
        for seq, logscore in beams:
            a,b = seq[-2], seq[-1]
            key = (a,b)
            cand_probs = cond_probs.get(key, unigram)
            # compute scores with probs (no extra multiplier here to keep simple)
            items = [(c, p) for c,p in cand_probs.items() if p > 0]
            # normalize for this candidate set
            total_p = sum(p for _,p in items) or 1.0
            for c,p in items:
                new_log = logscore + math.log(p/total_p)
                new_beams.append((seq + c, new_log))
        if not new_beams:
            break
        # select top beam_width beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:max(1, int(beam_width))]

    # get top_k unique 4-digit results (last 4 digits of the built seq)
    beams.sort(key=lambda x: x[1], reverse=True)
    results = []
    seen = set()
    for seq, score in beams:
        # seq currently contains starting two digits plus produced digits; we need last 4 digits
        # to match original behavior, ensure we take from the last full 6-digit context if possible
        full = seq  # e.g. "56..." (we only have tail)
        # construct a candidate final string: take the last 4 chars of seq (if shorter, pad)
        candidate_4 = full[-4:].rjust(4, "0")
        if candidate_4 not in seen:
            seen.add(candidate_4)
            results.append(candidate_4)
        if len(results) >= top_k:
            break

    return results

# === TAMPILAN HASIL ===
def tampilkan_prediksi(file_key, label, emoji):
    data = baca_data(file_key)
    st.subheader(f"{emoji} {label}")

    if not data:
        st.text("Tidak ada data valid.")
        return

    last_num = data[-1]
    st.write(f"Angka terakhir sebelum prediksi adalah: **{last_num}**")

    pred4 = markov_order2_predict(data, top_k=top_k, alpha=alpha, beam_width=beam_width)
    pred2 = [x[-2:] for x in pred4]

    st.markdown("**Prediksi 4 Digit (Top):**")
    st.write(", ".join(pred4) if pred4 else "—")

    st.markdown("**Prediksi 2 Digit (Top):**")
    st.write(", ".join(pred2) if pred2 else "—")

# === FUNGSI HELP: generate JSON response untuk API ===
def api_response_for_key(key, only_pred=None):
    # key: 'a'/'b'/'c' or 'all'
    if key in ("a","b","c"):
        data = baca_data(key)
        last = data[-1] if data else None
        pred4 = markov_order2_predict(data, top_k=top_k, alpha=alpha, beam_width=beam_width)
        pred2 = [p[-2:] for p in pred4]
        resp = {
            "file": key,
            "last": last,
            "series_count": len(data),
            "series": data,
            "pred4": pred4,
            "pred2": pred2,
        }
        if only_pred == "4":
            return {"file": key, "last": last, "pred4": pred4}
        if only_pred == "2":
            return {"file": key, "last": last, "pred2": pred2}
        return resp

    if key == "all":
        da = baca_data("a") or []
        db = baca_data("b") or []
        dc = baca_data("c") or []
        all_series = da + db + dc
        last = all_series[-1] if all_series else None
        pred4 = markov_order2_predict(all_series, top_k=top_k, alpha=alpha, beam_width=beam_width)
        pred2 = [p[-2:] for p in pred4]
        resp = {
            "file": "all",
            "last": last,
            "series_count": len(all_series),
            "series": all_series,
            "pred4": pred4,
            "pred2": pred2,
        }
        if only_pred == "4":
            return {"file": "all", "last": last, "pred4": pred4}
        if only_pred == "2":
            return {"file": "all", "last": last, "pred2": pred2}
        return resp

    return {"error": "invalid key"}

# === HANDLE API via query params ===
query = st.experimental_get_query_params()
api_param = query.get("api", [None])[0]  # a/b/c/all
pred_param = query.get("pred", [None])[0]  # "4" or "2" or None

if api_param:
    api_param = str(api_param).lower()
    if api_param in ("a","b","c","all"):
        only = pred_param if pred_param in ("4","2") else None
        resp = api_response_for_key(api_param, only_pred=only)
        # tampilkan JSON murni dan hentikan UI
        st.header("📡 API Response (JSON)")
        st.code(json.dumps(resp, indent=2), language="json")
        st.stop()
    else:
        st.error("API param tidak dikenali. Gunakan ?api=a|b|c|all")

# === JALANKAN UNTUK SETIAP FILE (UI) ===
tampilkan_prediksi("a", "File A", "📘")
tampilkan_prediksi("b", "File B", "📗")
tampilkan_prediksi("c", "File C", "📙")

# === PREDIKSI GABUNGAN ===
st.subheader("🧩 Gabungan Semua Data")

data_a = baca_data("a") or []
data_b = baca_data("b") or []
data_c = baca_data("c") or []

gabungan = data_a + data_b + data_c

if gabungan:
    st.write(f"Angka terakhir sebelum prediksi gabungan: **{gabungan[-1]}**")
    pred4_gab = markov_order2_predict(gabungan, top_k=top_k, alpha=alpha, beam_width=beam_width)
    pred2_gab = [x[-2:] for x in pred4_gab]

    st.markdown("**Prediksi 4 Digit (Top Gabungan):**")
    st.write(", ".join(pred4_gab) if pred4_gab else "—")

    st.markdown("**Prediksi 2 Digit (Top Gabungan):**")
    st.write(", ".join(pred2_gab) if pred2_gab else "—")
else:
    st.text("Belum ada data valid dari file A/B/C.")
