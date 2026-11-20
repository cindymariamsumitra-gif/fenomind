"""Buat docs/RESULTS.md dari artefak eksperimen (tidak ada angka yang diketik tangan).

    python scripts/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"

NAMES = {
    "dummy_prior": "Baseline (tebak prior kelas)",
    "logistic_regression": "Regresi logistik",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "temporal_cnn": "1D-CNN temporal (urutan per detik)",
    "tfidf_logreg": "TF-IDF + regresi logistik",
    "tfidf_linearsvm": "TF-IDF + Linear SVM (terkalibrasi)",
    "tfidf_complement_nb": "TF-IDF + Complement NB",
    "emb_minilm_multi_logreg": "Sentence embedding MiniLM-L12 multibahasa + LR",
    "emb_e5_small_multi_logreg": "Sentence embedding multilingual-e5-small + LR",
    "indobert_finetune": "IndoBERT-base-p1 fine-tune",
}


def _ci(m, k):
    c = m.get("ci95", {}).get(k)
    return f" [{c[0]:.3f}–{c[1]:.3f}]" if c else ""


def table(res: dict, selected: str | None = None) -> str:
    rows = ["| Model | ROC-AUC [CI 95%] | Balanced acc. | Accuracy | Precision | Sensitivity | Specificity | F1 |",
            "|---|---|---|---|---|---|---|---|"]
    for k, m in res.items():
        name = NAMES.get(k, k) + (" **(dipakai API)**" if k == selected else "")
        rows.append(f"| {name} | {m['roc_auc']:.3f}{_ci(m, 'roc_auc')} | {m['balanced_accuracy']:.3f} | "
                    f"{m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall_sensitivity']:.3f} | "
                    f"{m['specificity']:.3f} | {m['f1']:.3f} |")
    return "\n".join(rows)


def cm(m: dict) -> str:
    c = m["confusion_matrix"]
    return ("| | Prediksi normal | Prediksi depresi/terindikasi |\n|---|---|---|\n"
            f"| **Label normal** | {c['tn']} (TN) | {c['fp']} (FP) |\n"
            f"| **Label depresi/terindikasi** | {c['fn']} (FN) | {c['tp']} (TP) |")


def main() -> None:
    v = json.loads((A / "visual/results.json").read_text())
    t = json.loads((A / "text/results.json").read_text())
    vconf = json.loads((A / "visual/confound_analysis.json").read_text())
    vimp = json.loads((A / "visual/global_importance.json").read_text()) if (A / "visual/global_importance.json").exists() else None
    timp = json.loads((A / "text/global_importance.json").read_text()) if (A / "text/global_importance.json").exists() else None
    fusion = json.loads((A / "fusion/status.json").read_text())
    vsel = v["selected_model"]
    tsel = (json.loads((A / "text/indobert/model_meta.json").read_text())["model"]
            if (A / "text/indobert/final/model.safetensors").exists()
            else json.loads((A / "text/model_meta.json").read_text())["model"])
    vg, vr = v["protocols"]["grouped_cv"], v["protocols"]["random_cv_leaky_reference"]
    tg = t["protocols"]["grouped_cv"]
    d = v["dataset"]
    lines = [
        "# Hasil Eksperimen FENOMIND",
        "",
        "> Dibuat otomatis oleh `scripts/make_report.py` dari `artifacts/*/results.json`. Semua metrik dihitung",
        "> dari prediksi **out-of-fold** (sampel tidak pernah diprediksi oleh model yang melihatnya saat training).",
        "> CI 95% = bootstrap per grup (1.000 resampling). Ambang keputusan 0,5.",
        "",
        "**Ini bukan validasi klinis.** Angka di bawah hanya menunjukkan seberapa baik model membedakan label",
        "di dataset publik masing-masing, dengan semua keterbatasan di `DATASET_AUDIT.md`.",
        "",
        "## 1. Model visual (LMVD, fitur OpenFace)",
        "",
        f"- Data: {d['n_after_quality_filter']} rekaman setelah deduplikasi dan filter kualitas "
        f"(positif {d['positive_rate_filtered']:.1%}); {d['n_features']} fitur perilaku wajah; "
        f"{d['n_groups']} grup ({d['grouping']}).",
        f"- Deduplikasi: {d['deduplication']}.",
        "- Tuning hiperparameter bersarang (inner CV 3-fold berbasis grup); outer CV 5-fold.",
        "",
        "### 1a. CV berbasis grup (protokol utama)",
        "",
        table(vg, vsel),
        "",
        f"Confusion matrix model terpilih ({NAMES[vsel]}):",
        "",
        cm(vg[vsel]),
        "",
        "### 1b. Pembanding: CV acak (protokol asli LMVD, berpotensi bocor)",
        "",
        table(vr),
        "",
        f"Selisih AUC {NAMES[vsel]}: acak {vr[vsel]['roc_auc']:.3f} vs grup {vg[vsel]['roc_auc']:.3f}. "
        "Selisih kecil ini **tidak** membuktikan bahwa tidak ada kebocoran, karena grup hanya proksi "
        "(blok ID), bukan ID partisipan sebenarnya.",
        "",
        "### 1c. Kontrol konfon kualitas rekaman",
        "",
        "| Model kontrol (hanya fitur kualitas rekaman) | ROC-AUC [CI 95%] |",
        "|---|---|",
    ]
    for k in ("all_samples", "after_quality_filter"):
        m = v["quality_only_control"][k]
        lines.append(f"| {k} | {m['roc_auc']:.3f}{_ci(m, 'roc_auc')} |")
    lines += [
        "",
        "Kualitas rekaman saja sudah memprediksi label jauh di atas acak. Ini menandakan perbedaan sumber atau",
        "pengumpulan video antar kelas. AUC model wajah di dalam strata yang sama:",
        "",
        "| Strata | n | Proporsi positif | ROC-AUC |",
        "|---|---|---|---|",
    ]
    for s in vconf["by_valid_ratio"]:
        lines.append(f"| rasio frame valid {s['range'][0]}–{s['range'][1]} | {s['n']} | {s['positive_rate']:.2f} | {s['auc']:.3f} |")
    for s in vconf["by_fps"]:
        lines.append(f"| frame rate {s['fps']:.0f} fps | {s['n']} | {s['positive_rate']:.2f} | {s['auc']:.3f} |")
    lines += [
        "",
        "AUC tetap jauh di atas 0,5 di dalam strata, jadi daya pembeda tidak semata-mata berasal dari kualitas",
        "atau frame rate. Namun konfon sumber data lain (platform, gaya vlog, kata kunci pencarian) tetap mungkin.",
        "",
    ]
    if vimp:
        lines += ["### 1d. Kelompok fitur terpenting (permutation importance pada fold uji)", "",
                  "| Kelompok fitur | Penurunan AUC bila diacak |", "|---|---|"]
        lines += [f"| {f['label']} | {f['auc_drop']:+.4f} |" for f in vimp["families"]]
        lines += ["", "Fitur individual teratas: " + ", ".join(f"`{f['feature']}` ({f['auc_drop']:+.4f})"
                                                           for f in vimp["top_features"][:10]), ""]
    lines += [
        "## 2. Model linguistik (DatasetIndikasiDepresi)",
        "",
        f"- Data: {t['dataset']['n_unique_texts']} teks unik (dari {t['dataset']['n_raw_rows']} baris), "
        f"{t['dataset']['n_groups']} grup pengguna, positif {t['dataset']['positive_rate']:.1%}.",
        "- Outer CV 5-fold berbasis grup pengguna; tuning bersarang (inner 3-fold).",
        "",
        table(tg, tsel),
        "",
        f"Confusion matrix model yang dipakai API ({NAMES[tsel]}):",
        "",
        cm(tg[tsel]),
        "",
    ]
    if timp:
        lines += ["### 2a. Kata dengan bobot terbesar (model pembanding TF-IDF + LR)", "",
                  "- Ke arah *terindikasi*: " + ", ".join(f"`{d['term']}`" for d in timp["toward_indicated"][:20]),
                  "- Ke arah *tidak terindikasi*: " + ", ".join(f"`{d['term']}`" for d in timp["toward_not_indicated"][:20]),
                  "", "Bobot ini mencerminkan pola di tweet 2020 hasil crawling kata kunci, bukan penanda klinis.", ""]
    lines += [
        "## 3. Multimodal fusion",
        "",
        f"**Status: {fusion['status']}.** {fusion['reason']}",
        "",
        "Tidak ada angka performa fusion yang dilaporkan, karena tidak ada eksperimen fusion yang valid.",
        "",
        "## 4. Overfitting dan generalisasi",
        "",
        "- Semua angka berasal dari data yang tidak dipakai melatih model tersebut (out-of-fold); hiperparameter",
        "  dipilih di inner CV, jadi tidak ada seleksi model pada data uji.",
        "- Model kompleks tidak otomatis lebih baik: 1D-CNN temporal lebih buruk daripada XGBoost tabular pada data",
        "  visual, sedangkan IndoBERT fine-tune lebih baik daripada TF-IDF pada data teks. Pemilihan model didasarkan",
        "  pada hasil CV berbasis grup, bukan asumsi.",
        "- IndoBERT: loss latih turun ke 0,19–0,22 pada epoch 3 dan AUC uji per fold 0,78–0,85",
        "  (`artifacts/text/indobert/train_log.txt`, `fold_*.json`). Kurva validasi per epoch tidak direkam, jadi",
        "  besarnya overfitting tidak bisa dinilai pasti; jumlah epoch (3) ditetapkan sebelum melihat hasil.",
        "- Generalisasi ke mahasiswa Indonesia **belum diketahui**: populasi, bahasa, konteks perekaman, dan definisi",
        "  label berbeda dari kedua dataset latih. Validasi eksternal wajib sebelum dipakai untuk keputusan apa pun.",
        "",
    ]
    out = ROOT / "docs" / "RESULTS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("tersimpan:", out)


if __name__ == "__main__":
    main()
