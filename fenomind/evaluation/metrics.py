"""Metrik klasifikasi biner dan interval kepercayaan bootstrap.

Semua angka di laporan FENOMIND dihitung dari prediksi out-of-fold (data yang tidak
dipakai melatih model yang memprediksinya).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return out


def bootstrap_ci(y_true, prob, keys=("roc_auc", "balanced_accuracy", "f1", "recall_sensitivity", "specificity"),
                 n_boot: int = 1000, seed: int = 0, threshold: float = 0.5, groups=None) -> dict:
    """CI persentil 95%. Bila ``groups`` diberikan, resampling dilakukan per grup (cluster bootstrap)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        units = [np.array([i]) for i in range(len(y))]
    else:
        g = np.asarray(groups)
        units = [np.flatnonzero(g == u) for u in np.unique(g)]
    stats = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, len(units), len(units))
        idx = np.concatenate([units[i] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], p[idx], threshold)
        for k in keys:
            stats[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}


def summarize(y_true, prob, groups=None, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0) -> dict:
    m = binary_metrics(y_true, prob, threshold)
    m["ci95"] = bootstrap_ci(y_true, prob, groups=groups, threshold=threshold, n_boot=n_boot, seed=seed)
    return m
