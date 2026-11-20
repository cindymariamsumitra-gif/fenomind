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


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def run_indobert(args, texts, y, g, splits, res_path: Path, oof_path: Path) -> None:
    """Fine-tune IndoBERT per fold dengan penyimpanan incremental.

    Setelah SETIAP fold: prediksi + metrik fold (fold_k.json), checkpoint model fp16,
    metrik gabungan sementara di results.json, log, commit git lokal, dan ZIP diperbarui.
    Setiap epoch: state resume (resume_fold_k.pt). Menjalankan ulang perintah yang sama
    melanjutkan dari fold/epoch terakhir yang tersimpan.
    """
    import subprocess

    from fenomind.evaluation.metrics import binary_metrics
    from fenomind.text.finetune import fit_predict

    sys.path.insert(0, str(ROOT / "scripts"))
    from package_results import package

    d = Path(args.indobert_dir) if args.indobert_dir else OUT / "indobert"
    (d / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = d / "train_log.txt"
    if args.indobert_dir:  # mode uji: jangan sentuh hasil resmi
        res_path, oof_path = d / "results.json", d / "oof_predictions.csv"
        if not res_path.exists():
            _write_json(res_path, {"protocols": {"grouped_cv": {}}})
            pd.DataFrame({"label": y, "group": g}).to_csv(oof_path, index=False)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    config = {"model": args.model_name, "epochs": args.epochs, "lr": 2e-5, "max_len": 96,
              "batch_size": 16, "seed": args.seed, "folds": len(splits), "split": "StratifiedGroupKFold per pengguna"}
    _write_json(d / "config.json", config)
    log(f"mulai IndoBERT: {config}")
    prob = np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(splits, start=1):
        fold_file = d / f"fold_{k}.json"
        if fold_file.exists():
            saved = json.loads(fold_file.read_text())
            prob[np.array(saved["test_index"])] = saved["probabilities"]
            log(f"fold {k}/{len(splits)} sudah ada, dilewati")
            continue
        t0 = time.time()
        log(f"fold {k}/{len(splits)} mulai (train {len(tr)}, test {len(te)})")
        p, model, tok = fit_predict(texts[tr], y[tr], texts[te], seed=args.seed + k - 1, epochs=args.epochs,
                                    return_model=True, log=log, resume_path=d / f"resume_fold_{k}.pt",
                                    model_name=args.model_name)
        prob[te] = p
        ckpt = d / "checkpoints" / f"fold_{k}"
        model.half().save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
        m = binary_metrics(y[te], p)
        _write_json(fold_file, {"fold": k, "seconds": round(time.time() - t0, 1), "metrics": m,
                                "test_index": te.tolist(), "probabilities": [round(float(v), 6) for v in p],
                                "checkpoint": str(ckpt.relative_to(ROOT)) if ROOT in ckpt.parents else str(ckpt)})
        (d / f"resume_fold_{k}.pt").unlink(missing_ok=True)
        done = ~np.isnan(prob)
        partial = summarize(y[done], prob[done], groups=g[done])
        partial.update({"folds_completed": int(sum((d / f"fold_{i}.json").exists() for i in range(1, len(splits) + 1))),
                        "folds_total": len(splits), "complete": bool(done.all()), "config": config})
        results = json.loads(res_path.read_text())
        results["protocols"]["grouped_cv"]["indobert_finetune"] = partial
        _write_json(res_path, results)
        oof = pd.read_csv(oof_path)
        oof["grouped_cv__indobert_finetune"] = prob
        oof.to_csv(oof_path, index=False)
        log(f"fold {k}/{len(splits)} selesai ({time.time() - t0:.0f}s): AUC fold={m['roc_auc']:.3f}; "
            f"AUC gabungan {partial['folds_completed']} fold={partial['roc_auc']:.3f}")
        del model
        try:
            if args.indobert_dir:
                raise RuntimeError("mode uji")
            subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True, capture_output=True)
            subprocess.run(["git", "-c", "user.name=FENOMIND", "-c", "user.email=noreply@anthropic.com", "commit",
                            "-q", "-m", f"IndoBERT fold {k}/{len(splits)} selesai"], cwd=ROOT, capture_output=True)
        except Exception as e:  # noqa: BLE001 - commit lokal opsional
            log(f"commit git dilewati: {e}")
        for z in (package() if not args.no_package else []):
            log(f"ZIP diperbarui: {z.name} ({z.stat().st_size / 1e6:.0f} MB)")
    final = json.loads(res_path.read_text())["protocols"]["grouped_cv"]["indobert_finetune"]
    log(f"IndoBERT AUC={final['roc_auc']:.3f} BA={final['balanced_accuracy']:.3f} F1={final['f1']:.3f} "
        f"({final['folds_completed']}/{final['folds_total']} fold)")


def run_indobert_final(args, texts, y, res_path: Path) -> None:
    """Latih IndoBERT pada SELURUH data (konfigurasi sama dengan CV) untuk dipakai API.
    Performa yang dilaporkan untuknya adalah hasil CV 5-fold (bukan dari data latih ini)."""
    from fenomind.text.finetune import fit_predict

    sys.path.insert(0, str(ROOT / "scripts"))
    from package_results import package

    d = OUT / "indobert"
    final = d / "final"
    d.mkdir(parents=True, exist_ok=True)
    cv = json.loads(res_path.read_text())["protocols"]["grouped_cv"].get("indobert_finetune")
    if not cv or not cv.get("complete"):
        raise SystemExit("Jalankan dulu --indobert-only sampai 5/5 fold; metrik CV dibutuhkan untuk metadata.")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [final] {msg}"
        print(line, flush=True)
        with open(d / "train_log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    t0 = time.time()
    log(f"mulai pelatihan final pada {len(y)} teks")
    _, model, tok = fit_predict(texts, y, [], seed=args.seed, epochs=args.epochs, return_model=True, log=log,
                                resume_path=d / "resume_final.pt", model_name=args.model_name)
    model.half().save_pretrained(final, safe_serialization=True)
    tok.save_pretrained(final)
    (d / "resume_final.pt").unlink(missing_ok=True)
    meta = {
        "modality": "linguistic",
        "model": "indobert_finetune",
        "base_model": args.model_name,
        "checkpoint": "artifacts/text/indobert/final",
        "params": cv["config"],
        "preprocess": "fenomind.text.preprocess.clean_text",
        "threshold": 0.5,
        "training_data": "DatasetIndikasiDepresi (Budiman, 2021): tweet Indonesia 2020, label oleh penyusun dataset "
                         "dengan keterlibatan psikolog; tanpa lisensi eksplisit",
        "metrics_grouped_cv": {k: v for k, v in cv.items() if k != "config"},
        "metrics_note": "Metrik = CV 5-fold berbasis pengguna dengan konfigurasi identik; model final dilatih pada semua data.",
        "validation_status": "VALIDATED_ON_TWEETS_ONLY — domain tweet, belum divalidasi pada jawaban mahasiswa",
        "min_words": 5,
        "train_seconds": round(time.time() - t0, 1),
    }
    _write_json(d / "model_meta.json", meta)
    log(f"model final tersimpan di {final} ({time.time() - t0:.0f}s)")
    for z in package():
        log(f"ZIP diperbarui: {z.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--indobert-only", action="store_true")
    ap.add_argument("--indobert-final", action="store_true", help="latih IndoBERT pada seluruh data untuk API")
    ap.add_argument("--epochs", type=int, default=3)
    # Opsi uji cepat alur penyimpanan (bukan untuk hasil): model mini, subset data, folder terpisah
    ap.add_argument("--model-name", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--indobert-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-package", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, stats = load_indikasi_depresi()
    if args.limit:
        df = df.sample(args.limit, random_state=args.seed).reset_index(drop=True)
    texts, y, g = df["text"].to_numpy(), df["label"].to_numpy(), df["group"].to_numpy()
    splits = outer_splits(y, g, args.folds, args.seed, grouped=True)
    assert_no_group_leak(g, splits)
    res_path, oof_path = OUT / "results.json", OUT / "oof_predictions.csv"

    if args.indobert_only:
        run_indobert(args, texts, y, g, splits, res_path, oof_path)
        return
    if args.indobert_final:
        run_indobert_final(args, texts, y, res_path)
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
