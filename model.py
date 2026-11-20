"""Multimodal fusion FENOMIND (visual + linguistik).

STATUS: NOT_TRAINED / NOT_VALIDATED.
Belum ada dataset yang memasangkan rekaman wajah dan teks dari partisipan DAN sesi yang
sama dengan label depresi. LMVD (visual) tidak memuat transkrip; DatasetIndikasiDepresi
(teks) tidak memuat video. Memasangkan keduanya secara artifisial tidak valid, sehingga
modul ini menolak data yang tidak punya ``participant_id`` dan ``session_id`` yang cocok.

Dua arsitektur disediakan:
- ``LateFusion``: stacking regresi logistik atas logit model unimodal + indikator modality
  yang hilang. Bisa dipakai saat salah satu modality tidak ada.
- ``FeatureLevelFusion``: gabungan fitur visual + representasi teks (direduksi PCA) +
  indikator missing modality, lalu regresi logistik.

Keduanya hanya boleh dilatih pada pasangan valid (lihat ``validate_pairs``).
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "artifacts" / "fusion" / "status.json"
EPS = 1e-6


class FusionStatus(str, enum.Enum):
    NOT_TRAINED = "NOT_TRAINED"
    TRAINED_NOT_VALIDATED = "TRAINED_NOT_VALIDATED"
    VALIDATED = "VALIDATED"


def current_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {"status": FusionStatus.NOT_TRAINED.value,
            "reason": "Belum ada dataset berpasangan (wajah + teks, partisipan & sesi sama) berlabel."}


class InvalidPairingError(ValueError):
    pass


REQUIRED_PAIR_COLUMNS = ("participant_id", "session_id", "label")


def validate_pairs(visual: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan modality HANYA berdasarkan participant_id + session_id yang sama.

    - Kedua tabel wajib punya participant_id dan session_id.
    - Label harus sama di kedua sisi bila keduanya punya label.
    - Baris tanpa pasangan dipertahankan dengan modality yang hilang ditandai (bukan diisi dari orang lain).
    """
    for name, d in (("visual", visual), ("text", text)):
        missing = [c for c in ("participant_id", "session_id") if c not in d.columns]
        if missing:
            raise InvalidPairingError(f"Tabel {name} tidak punya kolom {missing}; pasangan tidak bisa diverifikasi.")
        if d.duplicated(["participant_id", "session_id"]).any():
            raise InvalidPairingError(f"Tabel {name} punya participant_id+session_id ganda.")
    merged = visual.merge(text, on=["participant_id", "session_id"], how="outer", suffixes=("_v", "_t"),
                          indicator=True)
    if "label_v" in merged and "label_t" in merged:
        both = merged["_merge"] == "both"
        clash = both & (merged["label_v"] != merged["label_t"])
        if clash.any():
            raise InvalidPairingError(f"{int(clash.sum())} pasangan punya label berbeda antar-modality.")
        merged["label"] = merged["label_v"].fillna(merged["label_t"])
    merged["has_visual"] = merged["_merge"].isin(["both", "left_only"]).astype(int)
    merged["has_text"] = merged["_merge"].isin(["both", "right_only"]).astype(int)
    return merged.drop(columns="_merge")


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class LateFusion:
    """Stacking atas probabilitas unimodal. NaN = modality tidak tersedia."""

    model: LogisticRegression | None = None
    status: FusionStatus = FusionStatus.NOT_TRAINED

    @staticmethod
    def design(p_visual, p_text) -> np.ndarray:
        pv, pt = np.asarray(p_visual, float), np.asarray(p_text, float)
        mv, mt = ~np.isnan(pv), ~np.isnan(pt)
        lv = np.where(mv, _logit(np.where(mv, pv, 0.5)), 0.0)
        lt = np.where(mt, _logit(np.where(mt, pt, 0.5)), 0.0)
        return np.column_stack([lv, lt, mv.astype(float), mt.astype(float)])

    def fit(self, p_visual, p_text, y):
        """p_visual/p_text WAJIB prediksi out-of-fold dari model unimodal pada pasangan yang sama."""
        self.model = LogisticRegression(max_iter=1000).fit(self.design(p_visual, p_text), y)
        self.status = FusionStatus.TRAINED_NOT_VALIDATED
        return self

    def predict_proba(self, p_visual, p_text) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LateFusion belum dilatih (status NOT_TRAINED).")
        return self.model.predict_proba(self.design(p_visual, p_text))[:, 1]

    def weights(self) -> dict | None:
        if self.model is None:
            return None
        c = self.model.coef_[0]
        return {"visual_logit": float(c[0]), "text_logit": float(c[1]), "has_visual": float(c[2]),
                "has_text": float(c[3]), "intercept": float(self.model.intercept_[0])}


class FeatureLevelFusion:
    """Gabungan fitur: [fitur visual | PCA(embedding teks) | indikator missing]."""

    def __init__(self, n_text_components: int = 16, C: float = 0.1):
        self.n_text_components = n_text_components
        self.C = C
        self.status = FusionStatus.NOT_TRAINED
        self._pca: PCA | None = None
        self._clf = None

    def _design(self, Xv, Et, fit: bool):
        Xv, Et = np.asarray(Xv, float), np.asarray(Et, float)
        has_v = (~np.isnan(Xv).all(axis=1)).astype(float)
        has_t = (~np.isnan(Et).all(axis=1)).astype(float)
        Et0 = np.where(np.isnan(Et), 0.0, Et)
        if fit:
            k = min(self.n_text_components, Et0.shape[1], max(int(has_t.sum()) - 1, 1))
            self._pca = PCA(n_components=k, random_state=0).fit(Et0[has_t == 1])
        Zt = self._pca.transform(Et0) * has_t[:, None]
        return np.column_stack([Xv, Zt, has_v, has_t])

    def fit(self, Xv, Et, y):
        D = self._design(Xv, Et, fit=True)
        self._clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced")).fit(D, y)
        self.status = FusionStatus.TRAINED_NOT_VALIDATED
        return self

    def predict_proba(self, Xv, Et) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("FeatureLevelFusion belum dilatih (status NOT_TRAINED).")
        return self._clf.predict_proba(self._design(Xv, Et, fit=False))[:, 1]
