"""Model temporal kecil untuk urutan perilaku wajah per detik (LMVD).

1D-CNN + attention pooling. Ukurannya sengaja kecil (~10 ribu parameter) karena
jumlah sampel hanya ~1.600.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

MAX_T = 300  # detik; urutan lebih panjang dipotong (mulai dari detik pertama)


def prepare(seqs: list[np.ndarray], mean: np.ndarray, std: np.ndarray, max_t: int = MAX_T):
    n, d = len(seqs), seqs[0].shape[1]
    X = np.zeros((n, max_t, d * 2), np.float32)  # nilai + mask
    lengths = np.zeros(n, np.int64)
    for i, s in enumerate(seqs):
        s = s[:max_t].astype(np.float32)
        mask = ~np.isnan(s)
        z = np.where(mask, (s - mean) / std, 0.0)
        X[i, : len(s), :d] = z
        X[i, : len(s), d:] = mask
        lengths[i] = max(len(s), 1)
    return torch.from_numpy(X), torch.from_numpy(lengths)


class TemporalCNN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, hidden, kernel_size=5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2), nn.ReLU(),
        )
        self.att = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B,T,H)
        t = torch.arange(h.shape[1], device=h.device)[None, :]
        valid = t < lengths[:, None]
        a = self.att(h).squeeze(-1).masked_fill(~valid, -1e9)
        w = torch.softmax(a, dim=1)
        pooled = (w.unsqueeze(-1) * h).sum(1)
        return self.head(pooled).squeeze(-1), w


def fit_predict(seq_train, y_train, seq_test, seed: int = 0, epochs: int = 40, lr: float = 2e-3,
                patience: int = 6) -> np.ndarray:
    """Latih dengan validasi internal 15% (early stopping), kembalikan probabilitas untuk seq_test."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train, np.float32)
    idx = rng.permutation(len(seq_train))
    n_val = max(int(0.15 * len(idx)), 20)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    stack = np.concatenate([seq_train[i][:MAX_T] for i in tr_idx]).astype(np.float32)
    mean, std = np.nanmean(stack, 0), np.nanstd(stack, 0) + 1e-6
    Xtr, Ltr = prepare([seq_train[i] for i in tr_idx], mean, std)
    Xva, Lva = prepare([seq_train[i] for i in val_idx], mean, std)
    Xte, Lte = prepare(seq_test, mean, std)
    ytr, yva = torch.from_numpy(y_train[tr_idx]), torch.from_numpy(y_train[val_idx])
    model = TemporalCNN(Xtr.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for b in range(0, len(perm), 64):
            j = perm[b : b + 64]
            opt.zero_grad()
            logit, _ = model(Xtr[j], Ltr[j])
            lossf(logit, ytr[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva, Lva)[0], yva).item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte, Lte)[0]).numpy()
