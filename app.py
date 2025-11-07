
---

# 5) `app.py` (kode lengkap)

Berikut `app.py` lengkap yang mengimplementasikan rancangan di atas. Salin seluruh isi ke file `app.py` di repo-mu.

> Catatan: aku mengutamakan kestabilan dan kontrol di UI. Fungsi-fungsi Cina (stem/branch, Lo Shu, I Ching) dibuat sederhana dan deterministik sehingga bisa langsung dipakai sebagai fitur pembobot.

```python
# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
import os, math

# -----------------------
# Config & mappings
# -----------------------
st.set_page_config(page_title="Prediksi Kombinasi (Markov2 + Jawa + Cina)", layout="centered")
st.title("🔢 Prediksi Kombinasi Angka — Markov2 + Jawa + Cina")
st.caption("Markov ordo-2 diperkuat dengan angka samaran, neptu Jawa, and beberapa fitur numerik Tiongkok.")

# alias (angka samaran)
ALIAS = {
    0: [1, 8],
    1: [0, 7],
    2: [5, 6],
    3: [8, 9],
    4: [7, 5],
    5: [2, 4],
    6: [9, 2],
    7: [4, 1],
    8: [3, 0],
    9: [6, 3],
}
HARI_MAP = {"senin":4,"selasa":3,"rabu":7,"kamis":8,"jumat":6,"sabtu":9,"minggu":5}
PASARAN_LIST = ["legi","pahing","pon","wage","kliwon"]
PASARAN_VAL = {"legi":5,"pahing":9,"pon":7,"wage":4,"kliwon":8}
STEMS = ["Jia","Yi","Bing","Ding","Wu","Ji","Geng","Xin","Ren","Gui"]   # indices 0..9
BRANCHES = ["Zi","Chou","Yin","Mao","Chen","Si","Wu","Wei","Shen","You","Xu","Hai"]  # 0..11

FILES = [("a.csv","📘 File A"), ("b.csv","📗 File B"), ("c.csv","📙 File C")]

# -----------------------
# Utility: Chinese & LoShu helpers
# -----------------------
def get_hari_pasaran():
    now = datetime.now()
    eng = now.strftime("%A").lower()
    eng_map = {"monday":"senin","tuesday":"selasa","wednesday":"rabu",
               "thursday":"kamis","friday":"jumat","saturday":"sabtu","sunday":"minggu"}
    hari = eng_map.get(eng, "kamis")
    pasaran = PASARAN_LIST[now.toordinal() % 5]
    return hari, pasaran, HARI_MAP[hari], PASARAN_VAL[pasaran]

def year_stem_branch(year):
    # stem and branch indices from year; classical formula uses offset 4 (so 1984 -> 0)
    idx_stem = (int(year) - 4) % 10
    idx_branch = (int(year) - 4) % 12
    return idx_stem, idx_branch

def loshu_balance_score(number):
    """
    Simple Lo Shu-inspired balance:
    - compute digit counts for digits 1..9 from the number's digits
    - compute variance or imbalance; map to multiplier around [0.9..1.1]
    """
    s = "".join(ch for ch in str(number) if ch.isdigit())
    if not s:
        return 1.0
    digits = [int(ch) for ch in s]
    # count 1..9 (map 0->9)
    counts = [0]*9
    for d in digits:
        if d == 0:
            idx = 8
        else:
            idx = (d-1) % 9
        counts[idx] += 1
    # imbalance measure: std dev normalized
    std = np.std(counts)
    # map std to multiplier: lower std -> closer to 1.08 (more balanced), higher std -> 0.92
    # choose mapping heuristic
    mul = max(0.85, min(1.15, 1.05 - (std-0.5)*0.1))
    return float(mul)

def iching_hex_index_from_number(number):
    """Map a number to hexagram index 0..63 by summing digits and mod 64."""
    s = "".join(ch for ch in str(number) if ch.isdigit())
    if not s:
        return 0
    val = sum(int(ch) for ch in s) % 64
    return int(val)

# -----------------------
# Data reading & normalization
# -----------------------
def read_raw_df(path):
    """Read CSV as raw (no header), keep strings (for searching right-most cell)."""
    try:
        return pd.read_csv(path, header=None, dtype=str)
    except Exception:
        return pd.DataFrame()

def read_and_normalize(path):
    """Return DataFrame with columns '6digit' and '4digit' extracted from first non-empty cell per row."""
    df_raw = read_raw_df(path)
    if df_raw.empty:
        return pd.DataFrame(columns=["6digit","4digit"])
    df_clean = df_raw.applymap(lambda x: "" if pd.isna(x) else "".join(ch for ch in str(x) if ch.isdigit()))
    vals = []
    for _, row in df_clean.iterrows():
        first = ""
        for cell in row:
            if cell != "":
                first = cell
                break
        if first != "":
            vals.append(first)
    if not vals:
        return pd.DataFrame(columns=["6digit","4digit"])
    s = pd.Series(vals).astype(str)
    s6 = s.str[-6:].str.zfill(6)
    return pd.DataFrame({"6digit": s6, "4digit": s6.str[-4:]})

def get_last_rightmost_in_bottom_row(df_raw):
    """
    Scan rows from bottom to top, and columns right->left. Return the first numeric string found (zfilled 6).
    If none found, return '-'.
    """
    if df_raw.empty:
        return "-"
    try:
        for _, row in df_raw[::-1].iterrows():
            for cell in reversed(row):
                if pd.isna(cell):
                    continue
                s = "".join(ch for ch in str(cell) if ch.isdigit())
                if s != "":
                    return s[-6:].zfill(6)
        return "-"
    except Exception:
        return "-"

# -----------------------
# Markov2 functions
# -----------------------
def build_markov2_counts(series6):
    counts = defaultdict(Counter)
    for s in series6:
        if not isinstance(s, str) or len(s) < 3:
            continue
        s6 = str(s).zfill(6)
        digits = list(s6)
        for i in range(len(digits)-2):
            a,b,c = digits[i], digits[i+1], digits[i+2]
            counts[(a,b)][c] += 1
    return counts

def cond_probs_from_counts(counts, alpha=1.0):
    probs = {}
    for key, counter in counts.items():
        total = sum(counter.values()) + alpha*10
        probs[key] = {}
        for d in map(str, range(10)):
            probs[key][d] = (counter.get(d,0) + alpha) / total
    return probs

def unigram_probs_from_counts(counts):
    total_counts = Counter()
    for counter in counts.values():
        total_counts.update(counter)
    total = sum(total_counts.values()) or 1
    probs = {str(d): (total_counts.get(str(d),0)/total) for d in range(10)}
    if sum(probs.values()) == 0:
        probs = {str(d): 1.0/10.0 for d in range(10)}
    return probs

# -----------------------
# multiplier function combining all features
# -----------------------
def composite_multiplier(prev_pair, candidate, features, weights):
    """
    prev_pair: tuple (a,b) as chars
    candidate: c as char
    features: dict with computed values (hari_val, pasaran_val, stem_idx, branch_idx, loshu_score, iching_idx)
    weights: dict of weights per feature (w_samaran, w_hari, w_pasaran, w_stem, w_branch, w_loshu, w_iching)
    Returns multiplier float >= 0
    """
    a = int(prev_pair[0]); b = int(prev_pair[1]); c = int(candidate)
    mul = 1.0

    # samaran
    if weights.get("w_samaran",0) > 0:
        w = weights["w_samaran"]
        score = 1.0
        if c in ALIAS.get(a, []):
            score *= 1.15
        if c in ALIAS.get(b, []):
            score *= 1.12
        # apply as exponent of weight: score^w
        mul *= (score ** w)

    # hari
    if weights.get("w_hari",0) > 0:
        w = weights["w_hari"]
        if c == features.get("hari_val"):
            mul *= (1.10 ** w)

    # pasaran
    if weights.get("w_pasaran",0) > 0:
        w = weights["w_pasaran"]
        if c == features.get("pasaran_val"):
            mul *= (1.10 ** w)

    # stem/branch influence (if exact match to stem_index or branch_index -> boost; else small distance effect)
    if weights.get("w_stem",0) > 0 or weights.get("w_branch",0) > 0:
        stem_idx = features.get("stem_idx", 0)
        branch_idx = features.get("branch_idx", 0)
        # map to small numeric proxies (1..10 and 1..12)
        # direct equality gives boost
        if weights.get("w_stem",0) > 0:
            w = weights["w_stem"]
            if c == (stem_idx % 10):
                mul *= (1.12 ** w)
            else:
                # small proximity boost (closer to stem -> slight)
                mul *= (1.0 + 0.01 * (5 - abs(c - (stem_idx % 10))) ) ** w
        if weights.get("w_branch",0) > 0:
            w = weights["w_branch"]
            if c == (branch_idx % 10):
                mul *= (1.10 ** w)
            else:
                mul *= (1.0 + 0.008 * (6 - abs(c - (branch_idx % 10)))) ** w

    # loshu
    if weights.get("w_loshu",0) > 0:
        w = weights["w_loshu"]
        loshu = features.get("loshu", 1.0)
        # loshu already around ~0.9..1.15, raise to w
        mul *= (loshu ** w)

    # iching
    if weights.get("w_iching",0) > 0:
        w = weights["w_iching"]
        hexidx = features.get("iching_idx", 0)
        # if candidate digit matches some simple pattern derived from hexidx, boost slightly
        if c == (hexidx % 10):
            mul *= (1.08 ** w)
    return float(mul)

# -----------------------
# Beam search (Markov2) with composite multiplier
# -----------------------
def generate_top_k_markov2_with_features(start_pair, cond_probs, unigram_probs, features, weights,
                                         steps=4, beam_width=12, top_k=5, alpha=1.0):
    beams = [( "".join(start_pair), 0.0 )]  # seq, logscore
    for step in range(steps):
        new_beams = []
        for seq, logscore in beams:
            a,b = seq[-2], seq[-1]
            key = (a,b)
            cand_probs = cond_probs.get(key, unigram_probs)
            scored = []
            for c, p in cand_probs.items():
                if p <= 0: continue
                mult = composite_multiplier((a,b), c, features, weights)
                score = p * mult
                if score <= 0: continue
                scored.append((c, score))
            total = sum(s for _,s in scored) or 1.0
            for c, s_prob in scored:
                new_log = logscore + math.log(s_prob / total)
                new_beams.append((seq + c, new_log))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    beams.sort(key=lambda x: x[1], reverse=True)
    top = beams[:top_k]
    if not top:
        return []
    exps = [math.exp(b[1]) for b in top]
    s = sum(exps) or 1.0
    results = [(top[i][0], exps[i]/s) for i in range(len(top))]
    return results

# -----------------------
# Helper UI tables
# -----------------------
def compute_position_top5(series6):
    counters = {"ribuan":Counter(), "ratusan":Counter(), "puluhan":Counter(), "satuan":Counter()}
    for s in series6:
        try:
            s6 = str(s).zfill(6)[-4:]
            counters["ribuan"][s6[0]] += 1
            counters["ratusan"][s6[1]] += 1
            counters["puluhan"][s6[2]] += 1
            counters["satuan"][s6[3]] += 1
        except Exception:
            continue
    top5 = {}
    for pos in ["ribuan","ratusan","puluhan","satuan"]:
        most = [k for k,_ in counters[pos].most_common(5)]
        most = (most + ["-"]*5)[:5]
        top5[pos] = most
    return top5

def top10_combinations(series6):
    ctr = Counter()
    for s in series6:
        try:
            ctr[str(s).zfill(6)[-4:]] += 1
        except Exception:
            continue
    return pd.DataFrame(ctr.most_common(10), columns=["Kombinasi 4 Digit","Frekuensi"])

# -----------------------
# Streamlit Controls (sidebar)
# -----------------------
st.sidebar.header("Model controls")
use_markov = st.sidebar.checkbox("Gunakan Markov Ordo-2", value=True)
use_samaran = st.sidebar.checkbox("Gunakan Angka Samaran", value=True)
use_hari = st.sidebar.checkbox("Gunakan Hari (neptu) sebagai faktor", value=True)
use_pasaran = st.sidebar.checkbox("Gunakan Pasaran (neptu) sebagai faktor", value=True)
use_stembranch = st.sidebar.checkbox("Gunakan Stem/Branch (Cina)", value=True)
use_loshu = st.sidebar.checkbox("Gunakan Lo Shu (balance)", value=False)
use_iching = st.sidebar.checkbox("Gunakan I Ching (hexagram)", value=False)

beam_width = st.sidebar.slider("Beam width (lebar pencarian)", min_value=3, max_value=60, value=12, step=1)
top_k = st.sidebar.slider("Jumlah hasil (Top K)", 1, 10, 5)
alpha = st.sidebar.number_input("Laplace smoothing alpha", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

st.sidebar.write("---")
st.sidebar.subheader("Feature weights (scale)")
w_samaran = st.sidebar.slider("w_samaran", 0.0, 2.0, 1.0, 0.05)
w_hari = st.sidebar.slider("w_hari", 0.0, 2.0, 1.0, 0.05)
w_pasaran = st.sidebar.slider("w_pasaran", 0.0, 2.0, 1.0, 0.05)
w_stem = st.sidebar.slider("w_stem", 0.0, 2.0, 0.8, 0.05)
w_branch = st.sidebar.slider("w_branch", 0.0, 2.0, 0.8, 0.05)
w_loshu = st.sidebar.slider("w_loshu", 0.0, 2.0, 0.6, 0.05)
w_iching = st.sidebar.slider("w_iching", 0.0, 2.0, 0.5, 0.05)

hari_name, pasaran_name, hari_val, pasaran_val = get_hari_pasaran()
st.write(f"📅 Hari ini: **{hari_name.capitalize()} {pasaran_name.capitalize()} (Neptu {hari_val+pasaran_val})**")
st.write("---")

# -----------------------
# Processing per-file & combined
# -----------------------
def process_and_show(path, title):
    st.subheader(title)
    if not os.path.exists(path):
        st.warning(f"File {path} tidak ditemukan.")
        return None, None
    df_raw = read_raw_df(path)
    df_norm = read_and_normalize(path)
    if df_norm.empty:
        st.warning("Tidak ada data valid pada file.")
        return None, None

    # last (rightmost in bottom row)
    last6 = get_last_rightmost_in_bottom_row(df_raw)
    st.caption(f"Angka terakhir sebelum prediksi (baris terbawah, kolom paling kanan): **{last6}**")

    series6 = df_norm["6digit"].tolist()
    counts = build_markov2_counts(series6)
    cond_probs = cond_probs_from_counts(counts, alpha=float(alpha))
    unigram = unigram_probs_from_counts(counts)

    pos_top5 = compute_position_top5(series6)
    table_df = pd.DataFrame({
        "ribuan": pos_top5["ribuan"],
        "ratusan": pos_top5["ratusan"],
        "puluhan": pos_top5["puluhan"],
        "satuan": pos_top5["satuan"]
    }, index=[1,2,3,4,5])
    st.write("📊 Frekuensi Angka per Posisi (Top-5)")
    st.dataframe(table_df, use_container_width=True)

    # prepare features for multipliers
    year = datetime.now().year
    stem_idx, branch_idx = year_stem_branch(year)
    loshu = loshu_balance_score(last6)
    iching_idx = iching_hex_index_from_number(last6)
    features = {
        "hari_val": hari_val,
        "pasaran_val": pasaran_val,
        "stem_idx": stem_idx,
        "branch_idx": branch_idx,
        "loshu": loshu,
        "iching_idx": iching_idx
    }
    weights = {
        "w_samaran": (w_samaran if use_samaran else 0.0),
        "w_hari": (w_hari if use_hari else 0.0),
        "w_pasaran": (w_pasaran if use_pasaran else 0.0),
        "w_stem": (w_stem if use_stembranch else 0.0),
        "w_branch": (w_branch if use_stembranch else 0.0),
        "w_loshu": (w_loshu if use_loshu else 0.0),
        "w_iching": (w_iching if use_iching else 0.0)
    }

    preds = []
    if use_markov:
        if last6 == "-" or len(last6) < 2:
            st.info("Tidak cukup data untuk pasangan awal (d5,d6).")
        else:
            start_pair = [last6[-2], last6[-1]]
            preds = generate_top_k_markov2_with_features(start_pair, cond_probs, unigram,
                                                        features, weights,
                                                        steps=4, beam_width=int(beam_width), top_k=int(top_k),
                                                        alpha=float(alpha))
    else:
        st.info("Markov disabled — tidak ada prediksi (gunakan Markov untuk prediksi).")

    st.write(f"🧠 Prediksi (Top-{top_k}) 6-digit — model: {'Markov2' if use_markov else 'None'}")
    if not preds:
        st.info("Tidak ada prediksi.")
    else:
        df_preds = pd.DataFrame([{"rank": i+1, "prediksi_6d": seq, "prediksi_4d": seq[-4:], "score": round(score,4)}
                                 for i,(seq,score) in enumerate(preds)])
        st.table(df_preds.set_index("rank"))

    st.write("🔥 Angka Dominan (Top-10 Kombinasi 4 Digit)")
    st.dataframe(top10_combinations(series6), use_container_width=True)

    # logs
    os.makedirs("logs", exist_ok=True)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = f"logs/markov2_{os.path.basename(path).replace('.csv','')}_{tstamp}.txt"
    with open(logfile, "w", encoding="utf-8") as f:
        f.write(f"File: {path}\nTanggal: {datetime.now()}\n")
        f.write(f"Angka terakhir (baris terbawah): {last6}\nFeatures: {features}\nWeights: {weights}\n")
        f.write("Prediksi top-k (6-digit, score):\n")
        for seq,score in preds:
            f.write(f"{seq}  {score:.6f}\n")
    st.caption(f"📁 Log tersimpan di: `{logfile}`")
    return series6, preds

# -----------------------
# Run all files
# -----------------------
st.header("Hasil per file")
all_series = []
for path, title in FILES:
    series6, preds = process_and_show(path, title)
    if series6:
        all_series.extend(series6)

st.write("---")
st.header("📦 Prediksi Gabungan (A + B + C)")
if all_series:
    counts_all = build_markov2_counts(all_series)
    cond_probs_all = cond_probs_from_counts(counts_all, alpha=float(alpha))
    unigram_all = unigram_probs_from_counts(counts_all)
    last6_combined = all_series[-1] if all_series else "-"
    st.caption(f"Angka terakhir gabungan (ambil terakhir dari gabungan): **{last6_combined}**")
    if last6_combined == "-" or len(last6_combined) < 2:
        st.info("Tidak cukup data untuk prediksi gabungan.")
    else:
        start_pair = [last6_combined[-2], last6_combined[-1]]
        # reuse features computed from last6_combined
        stem_idx, branch_idx = year_stem_branch(datetime.now().year)
        features_all = {
            "hari_val": hari_val,
            "pasaran_val": pasaran_val,
            "stem_idx": stem_idx,
            "branch_idx": branch_idx,
            "loshu": loshu_balance_score(last6_combined),
            "iching_idx": iching_hex_index_from_number(last6_combined)
        }
        preds_comb = generate_top_k_markov2_with_features(start_pair, cond_probs_all, unigram_all,
                                                           features_all, {
                                                               "w_samaran": w_samaran if use_samaran else 0.0,
                                                               "w_hari": w_hari if use_hari else 0.0,
                                                               "w_pasaran": w_pasaran if use_pasaran else 0.0,
                                                               "w_stem": w_stem if use_stembranch else 0.0,
                                                               "w_branch": w_branch if use_stembranch else 0.0,
                                                               "w_loshu": w_loshu if use_loshu else 0.0,
                                                               "w_iching": w_iching if use_iching else 0.0
                                                           },
                                                           steps=4, beam_width=int(beam_width), top_k=int(top_k),
                                                           alpha=float(alpha))
        if not preds_comb:
            st.info("Tidak ada prediksi gabungan.")
        else:
            df_preds_c = pd.DataFrame([{"rank":i+1,"prediksi_6d":seq,"prediksi_4d":seq[-4:],"score":round(score,4)}
                                       for i,(seq,score) in enumerate(preds_comb)])
            st.table(df_preds_c.set_index("rank"))
else:
    st.info("Gabungan kosong (tidak ada data valid di file A/B/C).")

st.write("---")
st.write("Tips: adjust feature toggles & weights in the sidebar to experiment with how cultural features affect predictions.")
