"""Explainability model linguistik: oklusi per kata (leave-one-word-out).

Kontribusi kata w = p(teks) - p(teks tanpa semua kemunculan w). Positif = kata itu
menaikkan skor skrining menurut model. Ini menjelaskan pola yang dipelajari dari data
latih (tweet 2020), bukan makna klinis kata tersebut.
"""

from __future__ import annotations

import numpy as np


def explain_text(model, clean: str, top_k: int = 8, max_words: int = 80) -> dict:
    words = clean.split()
    base = float(model.predict_proba([clean])[0, 1])
    uniq = list(dict.fromkeys(words))[:max_words]
    variants = [" ".join(w for w in words if w != u) for u in uniq]
    probs = model.predict_proba(variants)[:, 1] if variants else np.zeros(0)
    contrib = [{"word": u, "contribution": round(base - float(p), 4)} for u, p in zip(uniq, probs)]
    contrib.sort(key=lambda c: -abs(c["contribution"]))
    return {
        "method": "leave_one_word_out_occlusion",
        "probability": round(base, 4),
        "top_increasing": [c for c in contrib if c["contribution"] > 0][:top_k],
        "top_decreasing": [c for c in contrib if c["contribution"] < 0][:top_k],
        "note": "Kontribusi kata menurut model yang dilatih pada tweet; bukan penilaian klinis atas kata tersebut.",
    }
