"""Eksperimen fusion yang siap dijalankan begitu dataset berpasangan tersedia.

Format dataset berpasangan (CSV manifest), satu baris per sesi:
    participant_id, session_id, label (0/1), openface_csv (path), text
Pembagian data per participant_id (StratifiedGroupKFold). Model unimodal dilatih ulang
di setiap fold, lalu fusion dilatih pada prediksi out-of-fold dari fold latih (nested),
sehingga tidak ada kebocoran.

Membandingkan: visual saja, teks saja, late fusion, feature-level fusion; uji tambahan
daya prediksi dengan bootstrap berpasangan per partisipan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from ..evaluation.metrics import binary_metrics
from .model import FeatureLevelFusion, LateFusion


def _inner_oof(make_model, X, y, groups, n_splits=3, seed=0):
    p = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups):
        p[te] = clone(make_model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def run_fusion_cv(Xv: np.ndarray, texts: np.ndarray, Et: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  visual_model, text_model, n_splits: int = 5, seed: int = 0) -> dict:
    """Xv: fitur visual (NaN baris = tidak ada); texts/Et: teks & embedding (NaN = tidak ada)."""
    preds = {k: np.full(len(y), np.nan) for k in ("visual", "text", "late_fusion", "feature_fusion")}
    has_v = ~np.isnan(Xv).all(axis=1)
    has_t = ~np.isnan(Et).all(axis=1)
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(Xv, y, groups):
        trv, trt = tr[has_v[tr]], tr[has_t[tr]]
        vm = clone(visual_model).fit(Xv[trv], y[trv])
        tm = clone(text_model).fit(texts[trt], y[trt])
        pv_te = np.where(has_v[te], vm.predict_proba(np.nan_to_num(Xv[te]))[:, 1], np.nan)
        pt_te = np.where(has_t[te], tm.predict_proba(texts[te])[:, 1], np.nan)
        preds["visual"][te], preds["text"][te] = pv_te, pt_te
        # OOF di dalam fold latih untuk melatih fusion tanpa kebocoran
        pv_tr = np.full(len(tr), np.nan)
        pt_tr = np.full(len(tr), np.nan)
        pv_tr[has_v[tr]] = _inner_oof(visual_model, Xv[trv], y[trv], groups[trv], seed=seed)
        pt_tr[has_t[tr]] = _inner_oof(text_model, texts[trt], y[trt], groups[trt], seed=seed)
        lf = LateFusion().fit(pv_tr, pt_tr, y[tr])
        preds["late_fusion"][te] = lf.predict_proba(pv_te, pt_te)
        ff = FeatureLevelFusion().fit(Xv[tr], Et[tr], y[tr])
        preds["feature_fusion"][te] = ff.predict_proba(Xv[te], Et[te])
    out = {}
    for k, p in preds.items():
        ok = ~np.isnan(p)
        out[k] = binary_metrics(y[ok], p[ok]) | {"coverage": float(ok.mean())}
    out["paired_auc_gain_vs_best_unimodal"] = paired_bootstrap_gain(y, preds, groups, seed=seed)
    return out


def paired_bootstrap_gain(y, preds: dict, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Selisih AUC fusion - unimodal terbaik pada sampel yang punya KEDUA modality."""
    from sklearn.metrics import roc_auc_score

    both = ~np.isnan(preds["visual"]) & ~np.isnan(preds["text"])
    y, g = np.asarray(y)[both], np.asarray(groups)[both]
    P = {k: v[both] for k, v in preds.items()}
    if len(np.unique(y)) < 2:
        return {"note": "tidak cukup pasangan lengkap"}
    best_uni = max(("visual", "text"), key=lambda k: roc_auc_score(y, P[k]))
    rng = np.random.default_rng(seed)
    units = [np.flatnonzero(g == u) for u in np.unique(g)]
    res = {}
    for fk in ("late_fusion", "feature_fusion"):
        diffs = []
        for _ in range(n_boot):
            idx = np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))])
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y[idx], P[fk][idx]) - roc_auc_score(y[idx], P[best_uni][idx]))
        res[fk] = {"vs": best_uni, "auc_gain": float(roc_auc_score(y, P[fk]) - roc_auc_score(y, P[best_uni])),
                   "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return res


def load_manifest(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)
    need = {"participant_id", "session_id", "label"}
    if not need <= set(m.columns):
        raise ValueError(f"Manifest wajib punya kolom {sorted(need)}")
    if m.duplicated(["participant_id", "session_id"]).any():
        raise ValueError("participant_id+session_id ganda di manifest")
    return m
