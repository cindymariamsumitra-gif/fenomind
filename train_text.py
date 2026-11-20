"""Eksperimen dan pelatihan model linguistik (DatasetIndikasiDepresi).

    python scripts/train_text.py                  # baseline + TF-IDF + sentence embedding
    python scripts/train_text.py --indobert-only  # fine-tune IndoBERT (lama di CPU), digabung ke results.json

Evaluasi: 5-fold StratifiedGroupKFold, grup = pengguna Twitter (setelah deduplikasi).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.data.indikasi_depresi import load_indikasi_depresi  # noqa: E402
from fenomind.evaluation.cv import assert_no_group_leak, oof_predict, outer_splits  # noqa: E402
from fenomind.evaluation.metrics import summarize  # noqa: E402

OUT = ROOT / "artifacts" / "text"


def tfidf():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                 max_features=50000)),
    ])


def text_models(seed: int):
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), None),
        "tfidf_logreg": (
            make_pipeline(tfidf(), LogisticRegression(max_iter=5000, class_weight="balanced")),
            {"logisticregression__C": [0.3, 1.0, 3.0, 10.0]},
        ),
        "tfidf_linearsvm": (
            make_pipeline(tfidf(), CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
            {"calibratedclassifiercv__estimator__C": [0.03, 0.1, 0.3]},
        ),
        "tfidf_complement_nb": (
            make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), ComplementNB()),
            {"complementnb__alpha": [0.1, 0.3, 1.0]},
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--indobert-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, stats = load_indikasi_depresi()
    texts, y, g = df["text"].to_numpy(), df["label"].to_numpy(), df["group"].to_numpy()
    splits = outer_splits(y, g, args.folds, args.seed, grouped=True)
    assert_no_group_leak(g, splits)
    res_path, oof_path = OUT / "results.json", OUT / "oof_predictions.csv"

    if args.indobert_only:
        from fenomind.text.finetune import fit_predict

        results = json.loads(res_path.read_text())
        oof = pd.read_csv(oof_path)
        prob = np.full(len(y), np.nan)
        t0 = time.time()
        for k, (tr, te) in enumerate(splits):
            prob[te] = fit_predict(texts[tr], y[tr], texts[te], seed=args.seed + k, epochs=args.epochs)
            print(f"fold {k + 1}/{len(splits)} selesai ({time.time() - t0:.0f}s)", flush=True)
        m = summarize(y, prob, groups=g)
        m["seconds"] = round(time.time() - t0, 1)
        m["config"] = {"model": "indobenchmark/indobert-base-p1", "epochs": args.epochs, "lr": 2e-5, "max_len": 96}
        results["protocols"]["grouped_cv"]["indobert_finetune"] = m
        oof["grouped_cv__indobert_finetune"] = prob
        res_path.write_text(json.dumps(results, indent=2))
        oof.to_csv(oof_path, index=False)
        print(f"IndoBERT AUC={m['roc_auc']:.3f} BA={m['balanced_accuracy']:.3f} F1={m['f1']:.3f}")
        return

    results = {"dataset": stats, "protocols": {"grouped_cv": {}}}
    res = results["protocols"]["grouped_cv"]
    oof = pd.DataFrame({"label": y, "group": g})
    for name, (est, grid) in text_models(args.seed).items():
        t0 = time.time()
        Xin = texts if name != "dummy_prior" else np.zeros((len(y), 1))
        prob, chosen = oof_predict(est, grid, Xin, y, g, splits, seed=args.seed)
        res[name] = summarize(y, prob, groups=g)
        res[name]["chosen_params"] = [{k: str(v) for k, v in c.items()} for c in chosen]
        res[name]["seconds"] = round(time.time() - t0, 1)
        oof[f"grouped_cv__{name}"] = prob
        print(f"{name:24s} AUC={res[name]['roc_auc']:.3f} BA={res[name]['balanced_accuracy']:.3f} "
              f"F1={res[name]['f1']:.3f} ({res[name]['seconds']}s)", flush=True)

    # Sentence embedding (encoder dibekukan) + regresi logistik
    from fenomind.text.embeddings import ENCODERS, encode

    for enc_name in ENCODERS:
        t0 = time.time()
        E = encode(list(texts), enc_name)
        est = LogisticRegression(max_iter=5000, class_weight="balanced")
        prob, chosen = oof_predict(est, {"C": [0.1, 1.0, 10.0]}, E, y, g, splits, seed=args.seed)
        name = f"emb_{enc_name}_logreg"
        res[name] = summarize(y, prob, groups=g)
        res[name]["chosen_params"] = [{k: str(v) for k, v in c.items()} for c in chosen]
        res[name]["encoder"] = ENCODERS[enc_name][0]
        res[name]["seconds"] = round(time.time() - t0, 1)
        oof[f"grouped_cv__{name}"] = prob
        print(f"{name:24s} AUC={res[name]['roc_auc']:.3f} BA={res[name]['balanced_accuracy']:.3f} "
              f"F1={res[name]['f1']:.3f} ({res[name]['seconds']}s)", flush=True)

    # Model yang diekspor untuk API: TF-IDF + regresi logistik, karena kontribusi tiap kata
    # bisa dijelaskan langsung. Perbandingannya dengan model lain dilaporkan apa adanya.
    est, grid = text_models(args.seed)["tfidf_logreg"]
    inner = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=-1).fit(texts, y, groups=g)
    joblib.dump(gs.best_estimator_, OUT / "model.joblib")
    meta = {
        "modality": "linguistic",
        "model": "tfidf_logreg",
        "params": {k: str(v) for k, v in gs.best_params_.items()},
        "preprocess": "fenomind.text.preprocess.clean_text",
        "threshold": 0.5,
        "training_data": "DatasetIndikasiDepresi (Budiman, 2021): tweet Indonesia 2020, label oleh penyusun dataset "
                         "dengan keterlibatan psikolog; tanpa lisensi eksplisit",
        "metrics_grouped_cv": res["tfidf_logreg"],
        "validation_status": "VALIDATED_ON_TWEETS_ONLY — domain tweet, belum divalidasi pada jawaban mahasiswa",
        "min_words": 5,
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    res_path.write_text(json.dumps(results, indent=2))
    oof.to_csv(oof_path, index=False)
    print("tersimpan di", OUT)


if __name__ == "__main__":
    main()
