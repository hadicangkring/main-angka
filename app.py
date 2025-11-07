
---

## ⚙️ `app.py`
```python
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import os, math

# === SETUP
st.set_page_config(page_title="🔢 Prediksi Kombinasi Angka — Fusion China & Jawa", layout="centered")
st.title("🔢 Sistem Prediksi Angka — Fusion China & Jawa Calendar")
st.caption("Model Markov Orde-2 dengan integrasi Hari, Pasaran, dan rencana Kalender Cina.")

# === MAP & KONFIGURASI
HARI_MAP = {"senin":4,"selasa":3,"rabu":7,"kamis":8,"jumat":6,"sabtu":9,"minggu":5}
PASARAN_LIST = ["legi","pahing","pon","wage","kliwon"]
PASARAN_VAL = {"legi":5,"pahing":9,"pon":7,"wage":4,"kliwon":8}

ALIAS = {
    0:[1,8],1:[0,7],2:[5,6],3:[8,9],4:[7,5],5:[2,4],6:[9,2],7:[4,1],8:[3,0],9:[6,3]
}

FILES = [("a.csv","📘 File A"),("b.csv","📗 File B"),("c.csv","📙 File C")]

# === Fungsi bantu
def get_hari_pasaran():
    now = datetime.now()
    eng = now.strftime("%A").lower()
    eng_map = {
        "monday":"senin","tuesday":"selasa","wednesday":"rabu",
        "thursday":"kamis","friday":"jumat","saturday":"sabtu","sunday":"minggu"
    }
    hari = eng_map.get(eng, "kamis")
    pasaran = PASARAN_LIST[now.toordinal() % 5]
    return hari, pasaran, HARI_MAP[hari], PASARAN_VAL[pasaran]

def read_and_normalize(path):
    if not os.path.exists(path): return pd.DataFrame(columns=["6digit"])
    df_raw = pd.read_csv(path, header=None, dtype=str)
    df_clean = df_raw.applymap(lambda x: "".join(ch for ch in str(x) if ch.isdigit()) if pd.notna(x) else "")
    vals = []
    for _, row in df_clean.iterrows():
        first = [c for c in row if c != ""]
        if first:
            vals.append(first[-1])  # ambil paling kanan (terakhir)
    if not vals:
        return pd.DataFrame(columns=["6digit"])
    s6 = pd.Series(vals).astype(str).str[-6:].str.zfill(6)
    return pd.DataFrame({"6digit": s6})

def ambil_angka_terakhir(df):
    if df.empty: return "-"
    val = df["6digit"].iloc[-1]
    return str(val).zfill(6) if val else "-"

# === MARKOV MODEL
def build_markov2_counts(series):
    counts = defaultdict(Counter)
    for s in series:
        s6 = str(s).zfill(6)
        for i in range(len(s6)-2):
            counts[(s6[i], s6[i+1])][s6[i+2]] += 1
    return counts

def cond_probs(counts, alpha=1.0):
    probs = {}
    for key, counter in counts.items():
        total = sum(counter.values()) + alpha*10
        probs[key] = {str(d): (counter.get(str(d),0)+alpha)/total for d in range(10)}
    return probs

def unigram_probs(counts):
    total = Counter()
    for c in counts.values(): total.update(c)
    t = sum(total.values()) or 1
    return {str(d): total[str(d)]/t for d in range(10)}

def multiplier(prev, cand, hari_v, pasar_v):
    a,b = map(int, prev)
    c = int(cand)
    m = 1.0
    if c in ALIAS.get(a, []): m *= 1.12
    if c in ALIAS.get(b, []): m *= 1.10
    if c == hari_v: m *= 1.08
    if c == pasar_v: m *= 1.08
    return m

def generate_markov2(start, probs, uni, hari_v, pasar_v, steps=4, beam=10, topk=5):
    beams = [("".join(start), 0.0)]
    for _ in range(steps):
        new = []
        for seq, logp in beams:
            key = (seq[-2], seq[-1])
            cand = probs.get(key, uni)
            for c,p in cand.items():
                score = p * multiplier(key,c,hari_v,pasar_v)
                new.append((seq+c, logp+math.log(score)))
        new.sort(key=lambda x:x[1], reverse=True)
        beams = new[:beam]
    res = [(s,math.exp(sc)) for s,sc in beams[:topk]]
    s = sum(p for _,p in res) or 1
    return [(s,p/s) for s,p in res]

# === UI
st.sidebar.header("⚙️ Pengaturan")
beam = st.sidebar.slider("Beam Width",3,50,10)
topk = st.sidebar.slider("Top-K Prediksi",1,10,5)
alpha = st.sidebar.number_input("Laplace α",0.1,5.0,1.0)

hari, pasaran, hv, pv = get_hari_pasaran()
st.markdown(f"📅 **{hari.capitalize()} {pasaran.capitalize()}** (Neptu `{hv+pv}`)")
st.write("---")

def show_file(path,title):
    st.subheader(title)
    df = read_and_normalize(path)
    if df.empty:
        st.warning("Tidak ada data valid.")
        return []
    last = ambil_angka_terakhir(df)
    st.caption(f"Angka terakhir: **{last}**")

    series = df["6digit"].tolist()
    counts = build_markov2_counts(series)
    probs = cond_probs(counts, alpha)
    uni = unigram_probs(counts)

    if len(last) >= 2:
        start = [last[-2], last[-1]]
        preds = generate_markov2(start, probs, uni, hv, pv, beam=beam, topk=topk)
        st.table(pd.DataFrame([{"Rank":i+1,"Prediksi 6D":s,"Prediksi 4D":s[-4:],"Score":round(p,4)} for i,(s,p) in enumerate(preds)]))
    else:
        st.info("Data tidak cukup untuk memulai prediksi.")
    return series

all_series = []
for p,t in FILES:
    s = show_file(p,t)
    all_series += s
st.write("---")

if all_series:
    st.subheader("📦 Prediksi Gabungan")
    counts = build_markov2_counts(all_series)
    probs = cond_probs(counts, alpha)
    uni = unigram_probs(counts)
    last = all_series[-1]
    st.caption(f"Angka terakhir gabungan: **{last}**")
    if len(last)>=2:
        start = [last[-2], last[-1]]
        preds = generate_markov2(start, probs, uni, hv, pv, beam=beam, topk=topk)
        st.table(pd.DataFrame([{"Rank":i+1,"Prediksi 6D":s,"Prediksi 4D":s[-4:],"Score":round(p,4)} for i,(s,p) in enumerate(preds)]))
    else:
        st.info("Tidak cukup data gabungan.")
else:
    st.info("Belum ada data valid dari file A/B/C.")
