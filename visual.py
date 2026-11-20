"""Explainability model visual.

Atribusi per prediksi = oklusi per kelompok fitur: nilai satu kelompok diganti median
data latih, lalu dicatat perubahan probabilitasnya. Positif = kelompok itu menaikkan
skor skrining dibanding nilai tipikal di data latih.

Catatan: atribusi menjelaskan perilaku MODEL, bukan sebab klinis. Kelompok AU adalah
gerak otot wajah, bukan label emosi.
"""

from __future__ import annotations

import numpy as np

FAMILIES = {
    "au_intensity": ("Intensitas action unit (rata-rata, p90)", ("au_", ("_mean", "_p90"))),
    "au_variability": ("Variabilitas action unit", ("au_", ("_std",))),
    "au_dynamics": ("Kecepatan perubahan action unit", ("au_", ("_vel",))),
    "au_presence": ("Frekuensi kemunculan action unit", ("auc_", ("_rate",))),
    "composites": ("Komposit otot (senyum AU6+12, alis AU4/AU1, sudut bibir AU15)", ("comp_", ("",))),
    "expressivity": ("Ekspresivitas keseluruhan", ("expr_", ("",))),
    "head_pose": ("Gerak dan posisi kepala", ("head_", ("",))),
    "gaze": ("Arah dan gerak pandangan", ("gaze_", ("",))),
    "landmarks": ("Bukaan mulut/mata, gerak landmark, kedipan", ("lmk_", ("",))),
    "blink": ("Laju kedipan", ("blink_", ("",))),
}


def family_of(feature: str) -> str:
    for key, (_label, (prefix, suffixes)) in FAMILIES.items():
        if feature.startswith(prefix) and any(feature.endswith(s) for s in suffixes):
            return key
    return "other"


def family_index(features: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, f in enumerate(features):
        out.setdefault(family_of(f), []).append(i)
    return out


def explain_visual(model, x: np.ndarray, features: list[str], medians: dict[str, float], top_k: int = 5) -> dict:
    x = np.asarray(x, float).reshape(1, -1)
    base = float(model.predict_proba(x)[0, 1])
    med = np.array([medians.get(f, np.nan) for f in features], float)
    fam = family_index(features)
    contrib = []
    for key, idx in fam.items():
        z = x.copy()
        z[0, idx] = med[idx]
        delta = base - float(model.predict_proba(z)[0, 1])
        label = FAMILIES.get(key, ("Lainnya",))[0]
        contrib.append({"family": key, "label": label, "contribution": round(delta, 4)})
    contrib.sort(key=lambda c: -abs(c["contribution"]))
    return {
        "method": "occlusion_to_training_median_by_feature_family",
        "probability": round(base, 4),
        "families": contrib,
        "top_families": contrib[:top_k],
        "note": "Kontribusi = perubahan probabilitas bila kelompok fitur diganti nilai median data latih. "
                "Menjelaskan perilaku model, bukan penyebab klinis.",
    }
