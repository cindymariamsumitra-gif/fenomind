"""Fine-tuning IndoBERT untuk klasifikasi biner (CPU), dengan checkpoint per epoch untuk resume."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 96


def _length_bucketed_batches(lengths: np.ndarray, bs: int, rng: np.random.Generator):
    """Batch berisi teks dengan panjang mirip (padding minimal); urutan batch diacak."""
    order = np.argsort(lengths + rng.random(len(lengths)))  # sedikit acak di antara panjang sama
    batches = [order[b : b + bs] for b in range(0, len(order), bs)]
    rng.shuffle(batches)
    return batches


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def fit_predict(train_texts, y_train, test_texts, seed: int = 0, epochs: int = 3, lr: float = 2e-5,
                batch_size: int = 16, model_name: str = MODEL_NAME, return_model: bool = False, log=None,
                resume_path: Path | None = None):
    """Latih lalu prediksi. Bila ``resume_path`` diberikan, state disimpan setiap akhir epoch
    dan training dilanjutkan dari epoch terakhir yang tersimpan."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_texts = list(train_texts)
    ids = tok(train_texts, truncation=True, max_length=MAX_LEN)["input_ids"]
    lengths = np.array([len(i) for i in ids])
    y = torch.tensor(np.asarray(y_train), dtype=torch.long)
    rng = np.random.default_rng(seed)
    steps = epochs * int(np.ceil(len(y) / batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    # Bobot kelas untuk ketidakseimbangan ringan
    w = torch.tensor(len(y) / (2 * np.bincount(y.numpy(), minlength=2)), dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)

    start_epoch = 0
    if resume_path is not None and Path(resume_path).exists():
        state = torch.load(resume_path, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        rng.bit_generator.state = state["np_rng"]
        torch.set_rng_state(state["torch_rng"])
        start_epoch = state["epoch"] + 1
        if log:
            log(f"  melanjutkan dari checkpoint epoch {start_epoch}/{epochs}")

    model.train()
    for ep in range(start_epoch, epochs):
        losses = []
        for j in _length_bucketed_batches(lengths, batch_size, rng):
            batch = tok.pad({"input_ids": [ids[i] for i in j]}, return_tensors="pt")
            j = torch.from_numpy(j)
            opt.zero_grad()
            out = model(**batch)
            loss = lossf(out.logits, y[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        if resume_path is not None:
            _atomic_torch_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                                "epoch": ep}, Path(resume_path))
        if log:
            log(f"  epoch {ep + 1}/{epochs} selesai, rata-rata loss latih {np.mean(losses):.4f}")
    probs = predict_proba(model, tok, test_texts) if len(test_texts) else np.zeros(0)
    if return_model:
        return probs, model, tok
    return probs


@torch.no_grad()
def predict_proba(model, tok, texts, batch_size: int = 64) -> np.ndarray:
    model.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    for b in range(0, len(texts), batch_size):
        j = order[b : b + batch_size]
        enc = tok([texts[i] for i in j], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        out[j] = torch.softmax(model(**enc).logits, -1)[:, 1].numpy()
    return out
