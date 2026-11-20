"""Lapisan inference yang dipakai API (juga bisa dipanggil langsung dari Python)."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from ..explain.text import explain_text
from ..explain.visual import explain_visual
from ..fusion.model import current_status
from ..text.preprocess import clean_text
from ..visual.features import session_features

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
RESOURCES = ROOT / "fenomind" / "resources"

DISCLAIMER = ("FENOMIND adalah alat skrining awal, bukan alat diagnosis. Hasil ini menunjukkan kemiripan pola "
              "dengan data berlabel di dataset latih, bukan kondisi klinis seseorang. Bicarakan dengan "
              "psikolog atau konselor untuk penilaian yang sebenarnya.")


class ModelUnavailable(RuntimeError):
    pass


class IndoBertClassifier:
    """Pembungkus checkpoint IndoBERT final dengan antarmuka predict_proba(list[str]) seperti sklearn."""

    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        # Disimpan fp16 untuk menghemat ruang; dijalankan fp32 di CPU.
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()

    def predict_proba(self, texts) -> np.ndarray:
        from ..text.finetune import predict_proba

        p = predict_proba(self.model, self.tok, list(texts))
        return np.column_stack([1 - p, p])


def text_backend() -> str:
    """'indobert' bila checkpoint final ada (bisa dipaksa 'tfidf' lewat FENOMIND_TEXT_MODEL)."""
    choice = os.environ.get("FENOMIND_TEXT_MODEL", "auto").lower()
    has_bert = (ARTIFACTS / "text/indobert/final/model.safetensors").exists() and \
        (ARTIFACTS / "text/indobert/model_meta.json").exists()
    if choice == "tfidf" or (choice == "auto" and not has_bert):
        return "tfidf"
    if not has_bert:
        raise ModelUnavailable("FENOMIND_TEXT_MODEL=indobert tetapi checkpoint final belum ada "
                               "(python scripts/train_text.py --indobert-final).")
    return "indobert"


@lru_cache
def _load(modality: str):
    if modality == "text" and text_backend() == "indobert":
        d = ARTIFACTS / "text/indobert"
        return IndoBertClassifier(d / "final"), json.loads((d / "model_meta.json").read_text())
    d = ARTIFACTS / modality
    if not (d / "model.joblib").exists():
        raise ModelUnavailable(f"Model {modality} belum dilatih. Jalankan scripts/train_{modality}.py.")
    return joblib.load(d / "model.joblib"), json.loads((d / "model_meta.json").read_text())


@lru_cache
def safety_phrases() -> list[str]:
    return yaml.safe_load((RESOURCES / "safety_phrases.yaml").read_text(encoding="utf-8"))["phrases"]


@lru_cache
def crisis_resources() -> dict:
    return yaml.safe_load((RESOURCES / "crisis_resources.yaml").read_text(encoding="utf-8"))


def check_safety(clean: str) -> dict:
    padded = f" {clean} "
    hit = any(f" {clean_text(p)} " in padded for p in safety_phrases())
    out = {"triggered": hit}
    if hit:
        c = crisis_resources()
        out["help_resources"] = {"verified": bool(c.get("verified")), "resources": c.get("resources") or [],
                                 "fallback_text": c.get("fallback_text", "")}
    return out


def _model_info(meta: dict) -> dict:
    m = meta["metrics_grouped_cv"]
    return {
        "name": meta["model"],
        "metrics_note": meta.get("metrics_note", "Metrik = CV 5-fold berbasis grup (out-of-fold)."),
        "validation_status": meta["validation_status"],
        "training_data": meta["training_data"],
        "cv_metrics": {k: m[k] for k in ("roc_auc", "balanced_accuracy", "recall_sensitivity", "specificity", "f1")},
        "cv_ci95": m.get("ci95", {}),
    }


def _interpret(prob: float | None, threshold: float, modality: str) -> str:
    if prob is None:
        return "Tidak ada skor: kualitas input tidak memenuhi syarat."
    src = {"visual": "perilaku wajah", "linguistic": "bahasa"}[modality]
    if prob >= threshold:
        return (f"Pola {src} pada input ini lebih mirip dengan sampel berlabel depresi di data latih. "
                "Ini bukan diagnosis; pertimbangkan untuk berbicara dengan profesional.")
    return (f"Pola {src} pada input ini tidak menonjol mirip dengan sampel berlabel depresi di data latih. "
            "Ini tidak menjamin tidak ada masalah; bila kamu merasa tidak baik-baik saja, tetap cari bantuan.")


def screen_text(text: str) -> dict:
    model, meta = _load("text")
    clean = clean_text(text)
    n_words = len(clean.split())
    issues = []
    if n_words < meta.get("min_words", 5):
        issues.append(f"teks terlalu pendek ({n_words} kata; minimal {meta.get('min_words', 5)})")
    prob, expl = None, None
    if not issues:
        expl = explain_text(model, clean)
        prob = expl["probability"]
    thr = meta["threshold"]
    return {
        "modality": "linguistic",
        "probability": prob,
        "screening_flag": None if prob is None else bool(prob >= thr),
        "threshold": thr,
        "interpretation": _interpret(prob, thr, "linguistic"),
        "quality": {"ok": not issues, "issues": issues, "n_words": n_words},
        "explanation": expl,
        "safety": check_safety(clean),
        "model": _model_info(meta),
        "disclaimer": DISCLAIMER,
    }


def screen_openface(frames: pd.DataFrame) -> dict:
    model, meta = _load("visual")
    feats = session_features(frames)
    q = meta["quality_requirements"]
    issues = []
    if feats.get("q_too_few_valid_frames", 1.0) >= 1:
        issues.append("wajah terdeteksi dengan yakin di terlalu sedikit frame")
    elif feats["q_valid_ratio"] < q["min_valid_ratio"]:
        issues.append(f"hanya {feats['q_valid_ratio']:.0%} frame dengan wajah terdeteksi yakin "
                      f"(minimal {q['min_valid_ratio']:.0%})")
    prob, expl = None, None
    names = meta["features"]
    if not issues:
        x = np.array([feats.get(f, np.nan) for f in names], float)
        expl = explain_visual(model, x, names, meta["train_feature_stats"]["median"])
        prob = expl["probability"]
    thr = meta["threshold"]
    return {
        "modality": "visual",
        "probability": prob,
        "screening_flag": None if prob is None else bool(prob >= thr),
        "threshold": thr,
        "interpretation": _interpret(prob, thr, "visual"),
        "quality": {"ok": not issues, "issues": issues,
                    "valid_frame_ratio": round(feats["q_valid_ratio"], 3),
                    "duration_s": round(feats["q_duration_s"], 1)},
        "explanation": expl,
        "model": _model_info(meta),
        "disclaimer": DISCLAIMER,
    }


HEURISTIC_THRESHOLD = 0.5


def fusion_block(visual: dict | None = None, linguistic: dict | None = None) -> dict:
    """Blok fusion. ``combined_probability`` (model terlatih) tetap null selama fusion NOT_TRAINED.

    ``heuristic`` adalah gabungan SEDERHANA tanpa pelatihan untuk keperluan tampilan/demo:
    rata-rata probabilitas modality yang tersedia dan berkualitas cukup. Belum divalidasi:
    kedua model dilatih pada populasi berbeda, sehingga skalanya belum tentu sebanding.
    """
    st = current_status()
    probs = {k: r["probability"] for k, r in (("visual", visual), ("linguistic", linguistic))
             if r is not None and r.get("probability") is not None}
    heuristic = None
    if probs:
        p = float(np.mean(list(probs.values())))
        heuristic = {
            "status": "HEURISTIC_NOT_VALIDATED",
            "method": "rata-rata probabilitas modality yang tersedia (tanpa bobot terlatih)",
            "probability": round(p, 4),
            "screening_flag": bool(p >= HEURISTIC_THRESHOLD),
            "threshold": HEURISTIC_THRESHOLD,
            "modalities_used": sorted(probs),
            "any_modality_flagged": any(v >= HEURISTIC_THRESHOLD for v in probs.values()),
            "note": "Gabungan sederhana untuk tampilan, bukan model fusion terlatih. Tampilkan bersama hasil "
                    "per modality dan disclaimer.",
        }
    return {"status": st["status"], "combined_probability": None, "reason": st.get("reason"),
            "heuristic": heuristic,
            "note": "Model fusion terlatih belum ada (NOT_TRAINED), jadi combined_probability = null. "
                    "Lihat 'heuristic' untuk gabungan sederhana yang belum divalidasi."}


def concat_sessions(frames_list: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Gabungkan beberapa rekaman (mis. satu video per prompt) jadi satu sesi dengan timestamp bersambung.
    Jeda 1 detik disisipkan agar kecepatan perubahan tidak dihitung melintasi batas klip."""
    frames_list = [f for f in frames_list if f is not None and len(f)]
    if not frames_list:
        return None
    out, offset = [], 0.0
    for f in frames_list:
        f = f.copy()
        t0 = float(f["timestamp"].min())
        f["timestamp"] = f["timestamp"] - t0 + offset
        offset = float(f["timestamp"].max()) + 1.0
        out.append(f)
    return pd.concat(out, ignore_index=True)


def screen_session(answers: list[str], frames: pd.DataFrame | None = None, visual_note: str | None = None) -> dict:
    """Satu sesi interview multi-prompt (seperti di website FENOMIND).

    - Linguistik: setiap jawaban diskor terpisah (model dilatih pada teks pendek), lalu
      dirangkum dengan RATA-RATA probabilitas jawaban yang memenuhi syarat kualitas.
    - Keselamatan: dipicu bila SALAH SATU jawaban memuat frasa keselamatan.
    - Visual: semua klip digabung jadi satu sesi (bila ada fitur OpenFace).
    - Fusion: heuristik rata-rata (belum divalidasi), lihat ``fusion_block``.
    """
    per_prompt = []
    for i, a in enumerate(answers, start=1):
        if not a or not a.strip():
            per_prompt.append({"prompt": i, "answered": False})
            continue
        r = screen_text(a)
        per_prompt.append({"prompt": i, "answered": True, **r})
    scored = [p for p in per_prompt if p.get("probability") is not None]
    linguistic = None
    if scored:
        _, meta = _load("text")
        prob = float(np.mean([p["probability"] for p in scored]))
        thr = meta["threshold"]
        linguistic = {
            "modality": "linguistic",
            "aggregation": "rata-rata probabilitas jawaban per prompt yang memenuhi syarat kualitas",
            "n_prompts_scored": len(scored),
            "probability": round(prob, 4),
            "screening_flag": bool(prob >= thr),
            "threshold": thr,
            "interpretation": _interpret(prob, thr, "linguistic"),
            "model": scored[0]["model"],
        }
    safety_hit = any(p.get("safety", {}).get("triggered") for p in per_prompt)
    safety = {"triggered": safety_hit}
    if safety_hit:
        safety = next(p["safety"] for p in per_prompt if p.get("safety", {}).get("triggered"))
    visual = screen_openface(frames) if frames is not None else None
    return {
        "linguistic": linguistic,
        "per_prompt": per_prompt,
        "visual": visual,
        "visual_note": visual_note if visual is None else None,
        "missing_modalities": [k for k, v in (("visual", visual), ("linguistic", linguistic)) if v is None],
        "fusion": fusion_block(visual, linguistic),
        "safety": safety,
        "disclaimer": DISCLAIMER,
    }


def models_status() -> dict:
    out = {}
    for mod in ("visual", "text"):
        try:
            _, meta = _load(mod)
            out[mod] = {"available": True, **_model_info(meta)}
        except ModelUnavailable as e:
            out[mod] = {"available": False, "reason": str(e)}
    out["fusion"] = current_status()
    return out


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def safe_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not _SAFE_ID.match(value):
        raise ValueError("ID hanya boleh huruf, angka, titik, garis bawah, atau strip (maks 64).")
    return value
