"""Pemuat fitur LMVD hasil scripts/download_lmvd.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LMVD_DIR = ROOT / "data" / "processed" / "lmvd"

# ID LMVD tersusun per label (1..908 depresi, sisanya normal) dan 1.823 sampel berasal
# dari 1.475 partisipan tanpa ID partisipan. Proksi grup: blok ID berurutan, dengan
# asumsi vlog dari orang yang sama dikumpulkan berurutan. Ini BUKAN ID partisipan asli.
ID_BLOCK_SIZE = 10
MIN_VALID_RATIO = 0.5


def load_lmvd(directory: Path = LMVD_DIR) -> pd.DataFrame:
    rows = {}
    for line in (directory / "features.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "features" in r:
            rows[int(r["id"])] = r["features"]  # entri terakhir yang sukses dipakai
    X = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    labels = pd.read_csv(directory / "labels.csv")
    labels.index = labels["id"].astype(int)
    df = X.join(labels["label"], how="inner")
    df.index.name = "id"
    df["group"] = (df.index.to_numpy() - 1) // ID_BLOCK_SIZE
    return deduplicate(df)


def duplicate_key(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c not in ("label", "group")]
    return df[cols].round(6).fillna(-999.0).astype(str).agg("|".join, axis=1)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """LMVD memuat rekaman identik (vektor fitur sama persis, durasi sama). Sebagian
    berlabel berbeda. Rekaman berlabel bertentangan dibuang semua; yang berlabel sama
    disisakan satu. Hanya rekaman dengan frame valid yang dibandingkan."""
    has_face = df["q_too_few_valid_frames"] == 0
    key = duplicate_key(df).where(has_face, other=df.index.astype(str))
    n_labels = df.groupby(key)["label"].transform("nunique")
    conflict = n_labels > 1
    keep = ~conflict & ~key.duplicated(keep="first")
    out = df[keep].copy()
    out.attrs["dedup"] = {"conflicting_label_duplicates_removed": int(conflict.sum()),
                          "same_label_duplicates_removed": int((~conflict & key.duplicated(keep="first")).sum())}
    return out


def quality_filter(df: pd.DataFrame, min_valid_ratio: float = MIN_VALID_RATIO) -> pd.DataFrame:
    keep = (df["q_too_few_valid_frames"] == 0) & (df["q_valid_ratio"] >= min_valid_ratio)
    return df[keep]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Fitur perilaku wajah saja. Fitur kualitas (q_*) sengaja DIKELUARKAN karena
    berkorelasi dengan label di LMVD (artefak pengumpulan data, bukan gejala)."""
    return [c for c in df.columns if not c.startswith("q_") and c not in ("label", "group")]


def quality_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("q_") and c != "q_too_few_valid_frames"]


def load_sequence(sample_id: int, directory: Path = LMVD_DIR) -> np.ndarray:
    for name in (f"{sample_id:03d}.npy", f"{sample_id:04d}.npy", f"{sample_id}.npy"):
        p = directory / "seq" / name
        if p.exists():
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(sample_id)
