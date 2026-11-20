"""Lapisan inference yang dipakai API (juga bisa dipanggil langsung dari Python)."""

from __future__ import annotations

import json
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


@lru_cache
def _load(modality: str):
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
        "validation_status": meta["validation_status"],
        "training_data": meta["training_data"],
        "cv_metrics": {k: m[k] for k in ("roc_auc", "balanced_accuracy", "recall_sensitivity", "specificity", "f1")},
        "cv_ci95": m.get("ci95", {}),
    }


def _interpret(prob: float | None, threshold: float, modality: str) -> str:
    if prob is None:
        return "Tidak ada skor: kualitas input tidak memenuhi syarat."
    src = {"visual": "perilaku wajah", "linguistic": "pola bahasa"}[modality]
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


def fusion_block() -> dict:
    st = current_status()
    return {"status": st["status"], "combined_probability": None, "reason": st.get("reason"),
            "note": "Hasil per modality ditampilkan terpisah. Skor gabungan tidak dihitung karena model fusion "
                    "belum dilatih pada data berpasangan yang valid."}


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
