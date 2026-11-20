"""Fine-tuning IndoBERT untuk klasifikasi biner (CPU)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 96


def _batches(enc, idx, bs):
    for b in range(0, len(idx), bs):
        j = idx[b : b + bs]
        yield {k: v[j] for k, v in enc.items()}, j


def fit_predict(train_texts, y_train, test_texts, seed: int = 0, epochs: int = 3, lr: float = 2e-5,
                batch_size: int = 16, model_name: str = MODEL_NAME, return_model: bool = False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    enc_tr = tok(list(train_texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    y = torch.tensor(np.asarray(y_train), dtype=torch.long)
    steps = epochs * int(np.ceil(len(y) / batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    # Bobot kelas untuk ketidakseimbangan ringan
    w = torch.tensor(len(y) / (2 * np.bincount(y.numpy(), minlength=2)), dtype=torch.float)
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(y))
        for batch, j in _batches(enc_tr, perm, batch_size):
            opt.zero_grad()
            out = model(**batch)
            loss = lossf(out.logits, y[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    probs = predict_proba(model, tok, test_texts) if len(test_texts) else np.zeros(0)
    if return_model:
        return probs, model, tok
    return probs


@torch.no_grad()
def predict_proba(model, tok, texts, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(texts), batch_size):
        enc = tok(list(texts[b : b + batch_size]), truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)
