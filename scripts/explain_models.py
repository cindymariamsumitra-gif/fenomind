"""Explainability global, dihitung pada data UJI di setiap fold CV berbasis grup.

    python scripts/explain_models.py

- Visual: permutation importance (penurunan AUC) per fitur dan per kelompok fitur,
  rata-rata atas fold uji. Model = model terpilih di artifacts/visual/model_meta.json.
- Teks: koefisien terbesar fitur kata (unigram/bigram) model TF-IDF + regresi logistik final.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.data.lmvd import feature_columns, load_lmvd, quality_filter  # noqa: E402
from fenomind.evaluation.cv import outer_splits  # noqa: E402
from fenomind.explain.visual import FAMILIES, family_index  # noqa: E402


def visual_importance(seed: int = 42, repeats: int = 5) -> dict:
    meta = json.loads((ROOT / "artifacts/visual/model_meta.json").read_text())
    final = joblib.load(ROOT / "artifacts/visual/model.joblib")
    df = quality_filter(load_lmvd())
    feats = feature_columns(df)
    X, y, g = df[feats].to_numpy(float), df["label"].to_numpy(), df["group"].to_numpy()
    fam = family_index(feats)
    rng = np.random.default_rng(seed)
    per_feat = np.zeros(len(feats))
    per_fam = {k: 0.0 for k in fam}
    splits = outer_splits(y, g, 5, seed)
    for tr, te in splits:
        m = clone(final).fit(X[tr], y[tr])
        base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        for j in range(len(feats)):
            drops = []
            for _ in range(repeats):
                Z = X[te].copy()
                Z[:, j] = Z[rng.permutation(len(te)), j]
                drops.append(base - roc_auc_score(y[te], m.predict_proba(Z)[:, 1]))
            per_feat[j] += np.mean(drops) / len(splits)
        for k, idx in fam.items():
            drops = []
            for _ in range(repeats):
                Z = X[te].copy()
                perm = rng.permutation(len(te))
                Z[:, idx] = Z[perm][:, idx]
                drops.append(base - roc_auc_score(y[te], m.predict_proba(Z)[:, 1]))
            per_fam[k] += float(np.mean(drops)) / len(splits)
    order = np.argsort(-per_feat)
    return {
        "model": meta["model"],
        "method": "permutation importance pada fold uji (CV berbasis grup), penurunan ROC-AUC",
        "families": sorted(({"family": k, "label": FAMILIES.get(k, ("Lainnya",))[0], "auc_drop": round(v, 4)}
                            for k, v in per_fam.items()), key=lambda d: -d["auc_drop"]),
        "top_features": [{"feature": feats[i], "auc_drop": round(float(per_feat[i]), 4)} for i in order[:25]],
    }


def text_coefficients(top_k: int = 30) -> dict:
    pipe = joblib.load(ROOT / "artifacts/text/model.joblib")
    union, clf = pipe.steps[0][1], pipe.steps[-1][1]
    word_vec = dict(union.transformer_list)["word"]
    names = word_vec.get_feature_names_out()
    coef = clf.coef_[0][: len(names)]
    order = np.argsort(coef)
    return {
        "method": "koefisien regresi logistik untuk fitur kata (TF-IDF unigram/bigram); fitur karakter tidak ditampilkan",
        "toward_indicated": [{"term": names[i], "coef": round(float(coef[i]), 3)} for i in order[::-1][:top_k]],
        "toward_not_indicated": [{"term": names[i], "coef": round(float(coef[i]), 3)} for i in order[:top_k]],
    }


def main() -> None:
    v = visual_importance()
    (ROOT / "artifacts/visual/global_importance.json").write_text(json.dumps(v, indent=2))
    print("Visual — kelompok fitur:")
    for f in v["families"]:
        print(f"  {f['auc_drop']:+.4f}  {f['label']}")
    t = text_coefficients()
    (ROOT / "artifacts/text/global_importance.json").write_text(json.dumps(t, indent=2, ensure_ascii=False))
    print("Teks — arah 'terindikasi':", [d["term"] for d in t["toward_indicated"][:15]])
    print("Teks — arah 'tidak':", [d["term"] for d in t["toward_not_indicated"][:15]])


if __name__ == "__main__":
    main()
