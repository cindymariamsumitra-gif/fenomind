"""Praproses teks bahasa Indonesia. Fungsi yang SAMA dipakai saat training dan inference."""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_NON_LETTER = re.compile(r"[^a-z\s]")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACE = re.compile(r"\s+")

# Normalisasi ragam santai yang umum. Sengaja kecil dan eksplisit; bisa diperluas tim.
SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "engga": "tidak", "enggak": "tidak",
    "tak": "tidak", "gk": "tidak", "tdk": "tidak", "g": "tidak",
    "udah": "sudah", "udh": "sudah", "dah": "sudah", "sdh": "sudah",
    "aja": "saja", "aj": "saja", "bgt": "banget", "bngt": "banget",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "gw": "gue", "gua": "gue", "aku": "aku", "sy": "saya", "lu": "kamu", "lo": "kamu", "loe": "kamu",
    "kalo": "kalau", "klo": "kalau", "tp": "tapi", "tpi": "tapi", "jd": "jadi", "jdi": "jadi",
    "bener": "benar", "emg": "memang", "emang": "memang", "blm": "belum", "lg": "lagi", "sm": "sama",
    "cape": "capek", "capee": "capek",
}


def clean_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = t.replace("-", " ")
    t = _NON_LETTER.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)
    tokens = [SLANG.get(w, w) for w in t.split()]
    return _SPACE.sub(" ", " ".join(tokens)).strip()
