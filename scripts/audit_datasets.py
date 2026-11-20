"""Audit dataset. Semua angka di DATASET_AUDIT.md berasal dari skrip ini.

    python scripts/audit_datasets.py   ->  artifacts/audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.data.indikasi_depresi import RAW, load_indikasi_depresi  # noqa: E402
from fenomind.data.lmvd import feature_columns, load_lmvd, quality_filter  # noqa: E402


def _desc(s: pd.Series) -> dict:
    d = s.describe(percentiles=[0.25, 0.5, 0.75])
    return {k: round(float(v), 3) for k, v in d.items()}


def audit_lmvd() -> dict:
    df = load_lmvd()
    n_raw = len(pd.read_csv(ROOT / "data/processed/lmvd/labels.csv"))
    feats = feature_columns(df)
    filt = quality_filter(df)
    ids = df.index.to_numpy()
    first_normal = int(ids[df["label"].to_numpy() == 0].min())
    last_dep = int(ids[df["label"].to_numpy() == 1].max())
    # Duplikasi: vektor fitur identik / hampir identik (setelah standardisasi)
    Z = filt[feats].to_numpy(float)
    Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-9)
    Z = np.nan_to_num(Z)
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    dist = nn.kneighbors(Z)[0][:, 1]
    by_label = df.groupby("label")
    return {
        "source": "LMVD (He, 2024), figshare doi:10.6084/m9.figshare.25698351.v1, Apache-2.0; label dari GitHub helang818/LMVD",
        "n_samples_raw": int(n_raw),
        "deduplication": df.attrs.get("dedup", {}),
        "n_samples_after_dedup": int(len(df)),
        "n_participants_reported_by_authors": 1475,
        "participant_ids_available": False,
        "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "label_order": {"last_depression_id": last_dep, "first_normal_id": first_normal,
                        "positive_rate_per_100_ids": [round(float(v), 2) for v in
                                                      df.groupby((df.index.to_numpy() - 1) // 100)["label"].mean()],
                        "note": "ID tersusun dalam blok berlabel sama (bukan acak)"},
        "quality_by_class": {
            str(k): {"valid_ratio": _desc(g["q_valid_ratio"]), "duration_s": _desc(g["q_duration_s"]),
                     "confidence_mean": _desc(g["q_confidence_mean"])} for k, g in by_label
        },
        "low_quality": {
            str(t): {"n": int((df["q_valid_ratio"] < t).sum()),
                     "positive_rate": round(float(df.loc[df["q_valid_ratio"] < t, "label"].mean()), 3)}
            for t in (0.1, 0.3, 0.5)
        },
        "too_few_valid_frames": int(df["q_too_few_valid_frames"].sum()),
        "after_quality_filter": {"n": int(len(filt)), "class_counts": {str(k): int(v) for k, v in filt["label"].value_counts().items()}},
        "n_features": len(feats),
        "missing_values_after_filter": {"cells": int(filt[feats].isna().sum().sum()),
                                        "features_with_missing": int((filt[feats].isna().sum() > 0).sum())},
        "duplicates_remaining_after_dedup": {"exact_feature_duplicates": int((dist < 1e-9).sum()),
                       "near_duplicates_dist_lt_0.5": int((dist < 0.5).sum()),
                       "median_nn_distance": round(float(np.median(dist)), 3)},
        "modalities": {"visual": "OpenFace 2.0 per frame (AU intensitas & kehadiran, gaze, 68 landmark 2D, pose)",
                       "audio": "VGGish (tidak dipakai; FENOMIND tidak memakai suara sebagai modality)",
                       "text": "TIDAK ADA (tidak ada transkrip)"},
    }


def audit_text() -> dict:
    raw = pd.read_csv(RAW)
    df, stats = load_indikasi_depresi()
    words = df["text"].str.split().str.len()
    vc = raw["username"].value_counts()
    return {
        "source": "DatasetIndikasiDepresi (Budiman, 2021), github.com/andrebudiman/DatasetIndikasiDepresi, commit c3cbd20; tanpa lisensi eksplisit",
        "n_raw_rows": int(len(raw)),
        "readme_claims_rows": 10801,
        "columns": list(raw.columns),
        "missing_values_raw": {c: int(v) for c, v in raw.isna().sum().items()},
        "raw_class_counts": {str(k): int(v) for k, v in raw["sentiment"].value_counts().items()},
        "exact_duplicate_rows_cleanTweet": int(raw["cleanTweet"].duplicated().sum()),
        "tweet_id_precision_lost": bool(raw["tweet_id"].astype(str).str.contains("e\\+", case=False).any()
                                        or raw["tweet_id"].duplicated().mean() > 0.5),
        "n_users_raw": int(raw["username"].nunique()),
        "users_with_multiple_rows": int((vc > 1).sum()),
        "date_range": [str(pd.to_datetime(raw["date"], format="%m/%d/%Y %H:%M").min()),
                       str(pd.to_datetime(raw["date"], format="%m/%d/%Y %H:%M").max())],
        "location_counts": {k: int(v) for k, v in raw["location"].value_counts().items()},
        "after_cleaning": stats,
        "words_per_text": _desc(words),
        "keyword_bias": {kw: round(float(df["text"].str.contains(rf"\b{kw}\b").mean()), 3)
                         for kw in ("pernah", "sendiri", "aku", "tidak")},
        "modalities": {"text": "tweet (bukan jawaban wawancara/refleksi mahasiswa)", "visual": "TIDAK ADA"},
    }


def main() -> None:
    out = {"lmvd": audit_lmvd(), "indikasi_depresi": audit_text(),
           "paired_multimodal": {"available": False,
                                 "reason": "Tidak ada participant/session yang sama antara LMVD (vlog Tiongkok/global, "
                                           "tanpa teks) dan DatasetIndikasiDepresi (tweet Indonesia, tanpa video)."}}
    path = ROOT / "artifacts" / "audit.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
