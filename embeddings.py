"""Sentence embedding (encoder pralatih yang dibekukan) + cache di disk."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "processed" / "embeddings"

ENCODERS = {
    # nama pendek: (repo Hugging Face, prefiks input)
    "minilm_multi": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    "e5_small_multi": ("intfloat/multilingual-e5-small", "query: "),
}

_loaded: dict = {}


def get_encoder(name: str):
    if name not in _loaded:
        from sentence_transformers import SentenceTransformer

        _loaded[name] = SentenceTransformer(ENCODERS[name][0], device="cpu")
    return _loaded[name]


def encode(texts: list[str], name: str, batch_size: int = 64, cache: bool = True) -> np.ndarray:
    prefix = ENCODERS[name][1]
    key = hashlib.sha1(("\n".join(texts) + name).encode()).hexdigest()[:16]
    path = CACHE / f"{name}_{key}.npy"
    if cache and path.exists():
        return np.load(path)
    emb = get_encoder(name).encode([prefix + t for t in texts], batch_size=batch_size,
                                   normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    return emb


class EmbeddingFeaturizer:
    """Transformer sklearn: list teks -> embedding. Encoder pralatih tidak dilatih ulang."""

    def __init__(self, name: str = "minilm_multi"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return encode(list(X), self.name, cache=False)

    def get_params(self, deep=True):
        return {"name": self.name}

    def set_params(self, **p):
        self.name = p.get("name", self.name)
        return self
