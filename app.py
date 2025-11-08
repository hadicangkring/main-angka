# === PROCESS FILE C ===
import pandas as pd
import io
import re

def process_file_c(uploaded_file):
    """Membaca file C dengan dua blok data mingguan (lokal & HK Lotto)"""
    if uploaded_file is None:
        return None, "Tidak ada file diunggah."

    try:
        text = uploaded_file.getvalue().decode('utf-8')
    except Exception:
        try:
            text = uploaded_file.read().decode('utf-8')
        except:
            return None, "Gagal membaca file."

    # Pisahkan blok berdasarkan baris 'Data HK Lotto'
    parts = re.split(r'Data\s+HK\s+Lotto', text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None, "Format file tidak sesuai. Tidak ditemukan pemisah 'Data HK Lotto'."

    # === Fungsi bantu untuk parse tiap blok ===
    def parse_block(block_text):
        block_text = block_text.strip()
        lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
        # Hapus header non-data
        lines = [ln for ln in lines if not re.match(r'^[A-Z]', ln)]
        data = []
        for ln in lines:
            row = re.split(r'\s+', ln)
            if len(row) == 7:
                data.append([x if x != '–' else None for x in row])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["SENIN","SELASA","RABU","KAMIS","JUMAT","SABTU","MINGGU"])
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace('–', '').replace('nan','').replace('None','')
            df[col] = df[col].replace('', pd.NA)
        return df

    # Parse dua blok
    df_local = parse_block(parts[0])
    df_hk = parse_block(parts[1])

    # Gabungkan
    df_all = pd.concat([df_local, df_hk], ignore_index=True)

    return df_all, f"Berhasil membaca data: {len(df_all)} baris (lokal={len(df_local)}, HK={len(df_hk)})"
