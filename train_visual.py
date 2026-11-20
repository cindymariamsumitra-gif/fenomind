"""Eksperimen dan pelatihan model visual (LMVD, fitur OpenFace).

    python scripts/train_visual.py [--seed 42] [--skip-temporal]

Keluaran:
- artifacts/visual/results.json      semua metrik (out-of-fold) + CI 95%
- artifacts/visual/oof_predictions.csv
- artifacts/visual/model.joblib       model terpilih, dilatih ulang pada seluruh data tersaring
- artifacts/visual/model_meta.json    fitur, ambang, metrik, status validasi
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.data.lmvd import feature_columns, load_lmvd, load_sequence, quality_columns, quality_filter  # noqa: E402
from fenomind.evaluation.cv import assert_no_group_leak, oof_predict, outer_splits  # noqa: E402
from fenomind.evaluation.metrics import summarize  # noqa: E402

OUT = ROOT / "artifacts" / "visual"


def models(seed: int):
    imp = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), None),
        "logistic_regression": (
            make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
            {"logisticregression__C": [0.01, 0.1, 1.0]},
        ),
        "svm_rbf": (
            make_pipeline(imp(), StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=seed)),
            {"svc__C": [0.3, 1.0, 3.0], "svc__gamma": ["scale"]},
        ),
        "random_forest": (
            make_pipeline(imp(), RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                        random_state=seed, n_jobs=-1)),
            {"randomforestclassifier__max_depth": [6, None], "randomforestclassifier__min_samples_leaf": [1, 5]},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                          eval_metric="logloss", random_state=seed, n_jobs=1),
            {"max_depth": [2, 4], "min_child_weight": [1, 5]},
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-temporal", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    full = load_lmvd()
    df = quality_filter(full)
    feats = feature_columns(df)
    X, y, g = df[feats].to_numpy(np.float64), df["label"].to_numpy(), df["group"].to_numpy()
    results = {
        "dataset": {
            "n_total": int(len(full)), "n_after_quality_filter": int(len(df)),
            "positive_rate_total": float(full["label"].mean()), "positive_rate_filtered": float(df["label"].mean()),
            "n_features": len(feats), "n_groups": int(df["group"].nunique()),
            "grouping": "proksi: blok 10 ID berurutan (LMVD tidak menyediakan ID partisipan)",
            "deduplication": full.attrs.get("dedup", {}),
        },
        "protocols": {},
    }
    splits = {
        "grouped_cv": outer_splits(y, g, args.folds, args.seed, grouped=True),
        "random_cv_leaky_reference": outer_splits(y, g, args.folds, args.seed, grouped=False),
    }
    assert_no_group_leak(g, splits["grouped_cv"])
    oof = pd.DataFrame({"id": df.index, "label": y, "group": g})

    for proto, sp in splits.items():
        res = results["protocols"].setdefault(proto, {})
        for name, (est, grid) in models(args.seed).items():
            t0 = time.time()
            prob, chosen = oof_predict(est, grid, X, y, g, sp, seed=args.seed)
            res[name] = summarize(y, prob, groups=g)
            res[name]["chosen_params"] = [{k: str(v) for k, v in c.items()} for c in chosen]
            res[name]["seconds"] = round(time.time() - t0, 1)
            oof[f"{proto}__{name}"] = prob
            print(f"{proto:28s} {name:20s} AUC={res[name]['roc_auc']:.3f} "
                  f"BA={res[name]['balanced_accuracy']:.3f} F1={res[name]['f1']:.3f} ({res[name]['seconds']}s)",
                  flush=True)

        if not args.skip_temporal:
            seqs = [load_sequence(int(i)) for i in df.index]
            from fenomind.visual.temporal import fit_predict

            prob = np.full(len(y), np.nan)
            t0 = time.time()
            for k, (tr, te) in enumerate(sp):
                prob[te] = fit_predict([seqs[i] for i in tr], y[tr], [seqs[i] for i in te], seed=args.seed + k)
            res["temporal_cnn"] = summarize(y, prob, groups=g)
            res["temporal_cnn"]["seconds"] = round(time.time() - t0, 1)
            oof[f"{proto}__temporal_cnn"] = prob
            print(f"{proto:28s} {'temporal_cnn':20s} AUC={res['temporal_cnn']['roc_auc']:.3f} "
                  f"BA={res['temporal_cnn']['balanced_accuracy']:.3f}", flush=True)

    # Kontrol konfon: seberapa jauh label bisa ditebak HANYA dari kualitas rekaman?
    qcols = quality_columns(full)
    ctrl = {}
    for label, data in (("all_samples", full), ("after_quality_filter", df)):
        sp = outer_splits(data["label"].to_numpy(), data["group"].to_numpy(), args.folds, args.seed)
        est = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=5000))
        prob, _ = oof_predict(est, None, data[qcols].to_numpy(), data["label"].to_numpy(),
                              data["group"].to_numpy(), sp)
        ctrl[label] = summarize(data["label"].to_numpy(), prob, groups=data["group"].to_numpy())
        print(f"quality-only control ({label}): AUC={ctrl[label]['roc_auc']:.3f}")
    results["quality_only_control"] = {"features": qcols, **ctrl}

    # Pilih model terbaik berdasarkan AUC CV berbasis grup (bukan protokol bocor)
    grouped = results["protocols"]["grouped_cv"]
    candidates = {k: v for k, v in grouped.items() if k not in ("dummy_prior", "temporal_cnn")}
    best = max(candidates, key=lambda k: candidates[k]["roc_auc"])
    results["selected_model"] = best
    print("model terpilih:", best)

    est, grid = models(args.seed)[best]
    inner = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc", n_jobs=-1).fit(X, y, groups=g)
    joblib.dump(gs.best_estimator_, OUT / "model.joblib")
    meta = {
        "modality": "visual",
        "model": best,
        "params": {k: str(v) for k, v in gs.best_params_.items()},
        "features": feats,
        "feature_extractor": "fenomind.visual.features.session_features (OpenFace 2.0 per-frame CSV)",
        "threshold": 0.5,
        "quality_requirements": {"min_valid_ratio": 0.5, "min_valid_frames": 30},
        "training_data": "LMVD (He, 2024; Apache-2.0), vlog 4 platform, label depresi/normal dari pengumpul dataset",
        "metrics_grouped_cv": grouped[best],
        "validation_status": "VALIDATED_ON_LMVD_ONLY — belum divalidasi pada mahasiswa Indonesia",
        "train_feature_stats": {"median": df[feats].median().to_dict()},
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("tersimpan di", OUT)


if __name__ == "__main__":
    main()
