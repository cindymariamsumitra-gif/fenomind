"""Dataset linguistik: DatasetIndikasiDepresi (Budiman, 2021).

Sumber: https://github.com/andrebudiman/DatasetIndikasiDepresi (tweet berbahasa Indonesia,
18 Nov–7 Des 2020 menurut README; kolom tanggal memuat 9 Nov–7 Des 2020).
Repositori sumber TIDAK mencantumkan lisensi, jadi data ini tidak didistribusikan ulang
di repo FENOMIND; unduh dengan scripts/download_text_data.py.

Langkah pembersihan (alasannya di DATASET_AUDIT.md):
1. Deduplikasi teks persis (6.194 dari 10.082 baris adalah salinan).
2. Buang teks dengan label bertentangan antar-salinan.
3. Grup = komponen terhubung pengguna yang berbagi teks yang sama, sehingga satu
   pengguna tidak pernah muncul di train dan test sekaligus.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..text.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "indikasi_depresi" / "Data10k_Csv.csv"
LABEL_MAP = {"Terindikasi Depresi": 1, "Tidak Terindikasi": 0}


def _user_components(df: pd.DataFrame) -> dict[str, int]:
    parent: dict[str, str] = {}

    def find(u):
        while parent.setdefault(u, u) != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    for _, users in df.groupby("dedup_key")["username"]:
        users = list(users)
        for u in users[1:]:
            ru, r0 = find(u), find(users[0])
            if ru != r0:
                parent[ru] = r0
    roots = {u: find(u) for u in df["username"].unique()}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {u: ids[r] for u, r in roots.items()}


def load_indikasi_depresi(path: Path = RAW) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    stats = {"n_raw_rows": int(len(raw))}
    df = raw[["username", "deEmoji", "cleanTweet", "sentiment"]].copy()
    df["label"] = df["sentiment"].map(LABEL_MAP)
    stats["n_unmapped_labels"] = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"])
    # Kunci dedup dari teks yang sudah dinormalisasi manual oleh penyusun dataset
    df["dedup_key"] = df["cleanTweet"].astype(str).str.lower().str.split().str.join(" ")
    df["text"] = df["deEmoji"].astype(str).map(clean_text)  # praproses FENOMIND (sama dengan inference)
    groups = _user_components(df)
    conflict = df.groupby("dedup_key")["label"].nunique()
    conflict_keys = set(conflict[conflict > 1].index)
    stats["n_conflicting_label_texts"] = len(conflict_keys)
    df = df[~df["dedup_key"].isin(conflict_keys)]
    stats["n_duplicate_rows_removed"] = int(df.duplicated("dedup_key").sum())
    dedup = df.drop_duplicates("dedup_key", keep="first").copy()
    dedup["group"] = dedup["username"].map(groups)
    dedup = dedup[dedup["text"].str.len() > 0].reset_index(drop=True)
    dedup["label"] = dedup["label"].astype(int)
    stats.update({
        "n_unique_texts": int(len(dedup)),
        "n_groups": int(dedup["group"].nunique()),
        "positive_rate": float(dedup["label"].mean()),
        "label_counts": {str(k): int(v) for k, v in dedup["label"].value_counts().items()},
    })
    return dedup[["text", "label", "group"]], stats
