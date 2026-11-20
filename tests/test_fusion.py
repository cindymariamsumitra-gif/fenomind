"""Fusion: hanya menguji KODE (data sintetis). Tidak ada klaim performa."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from fenomind.fusion.experiment import run_fusion_cv
from fenomind.fusion.model import (
    FeatureLevelFusion, FusionStatus, InvalidPairingError, LateFusion, current_status, validate_pairs,
)


def test_status_is_not_trained_by_default():
    assert current_status()["status"] == FusionStatus.NOT_TRAINED.value


def test_validate_pairs_requires_ids_and_matching_labels():
    v = pd.DataFrame({"participant_id": ["a", "b"], "session_id": ["1", "1"], "label": [1, 0]})
    t = pd.DataFrame({"participant_id": ["a", "c"], "session_id": ["1", "1"], "label": [1, 1]})
    m = validate_pairs(v, t)
    row = m.set_index("participant_id")
    assert row.loc["a", "has_visual"] == 1 and row.loc["a", "has_text"] == 1
    assert row.loc["b", "has_text"] == 0 and row.loc["c", "has_visual"] == 0
    with pytest.raises(InvalidPairingError):
        validate_pairs(v.drop(columns="session_id"), t)
    with pytest.raises(InvalidPairingError, match="label berbeda"):
        validate_pairs(v, t.assign(label=[0, 1]))


def test_late_fusion_untrained_refuses_and_handles_missing_modality():
    with pytest.raises(RuntimeError, match="NOT_TRAINED"):
        LateFusion().predict_proba([0.5], [0.5])
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    pv = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, 200), 0.01, 0.99)
    pt = np.clip(0.3 + 0.2 * y + rng.normal(0, 0.2, 200), 0.01, 0.99)
    pt[:50] = np.nan  # teks hilang
    lf = LateFusion().fit(pv, pt, y)
    assert lf.status == FusionStatus.TRAINED_NOT_VALIDATED
    out = lf.predict_proba([0.9, np.nan], [np.nan, 0.9])
    assert out.shape == (2,) and not np.isnan(out).any()


def test_fusion_cv_runs_end_to_end_on_synthetic_pairs():
    rng = np.random.default_rng(1)
    n = 120
    y = rng.integers(0, 2, n)
    groups = np.arange(n) // 2
    Xv = rng.normal(0, 1, (n, 5)) + y[:, None] * 0.8
    words = np.array(["capek sedih", "senang main", "biasa saja"])
    texts = np.array([f"{words[y_i]} hari {i}" for i, y_i in enumerate(y)], dtype=object)
    Et = rng.normal(0, 1, (n, 8)) + y[:, None] * 0.5
    Xv[:10] = np.nan
    texts_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    from sklearn.impute import SimpleImputer
    vis_model = make_pipeline(SimpleImputer(), LogisticRegression())
    res = run_fusion_cv(Xv, texts, Et, y, groups, vis_model, texts_model, n_splits=3)
    assert set(res) >= {"visual", "text", "late_fusion", "feature_fusion", "paired_auc_gain_vs_best_unimodal"}
    assert res["visual"]["coverage"] < 1.0
    ff = FeatureLevelFusion().fit(Xv, Et, y)
    assert ff.predict_proba(Xv[:3], Et[:3]).shape == (3,)
