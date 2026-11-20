"""Cross-validation berbasis grup dengan tuning bersarang (nested).

Grup memastikan sampel dari partisipan/pengguna yang sama (atau proksinya) tidak
pernah berada di fold latih dan fold uji sekaligus.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold


def outer_splits(y, groups, n_splits: int, seed: int, grouped: bool = True):
    y = np.asarray(y)
    if grouped:
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def assert_no_group_leak(groups, splits) -> None:
    g = np.asarray(groups)
    for tr, te in splits:
        leak = set(g[tr]) & set(g[te])
        if leak:
            raise AssertionError(f"Kebocoran grup antara train dan test: {sorted(leak)[:5]}")


def oof_predict(estimator, param_grid: dict | None, X, y, groups, splits, inner_splits: int = 3,
                seed: int = 0, scoring: str = "roc_auc", fit_fn: Callable | None = None):
    """Prediksi probabilitas out-of-fold. Hiperparameter dipilih di inner CV (per grup)."""
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    prob = np.full(len(y), np.nan)
    chosen = []
    for tr, te in splits:
        if fit_fn is not None:
            prob[te] = fit_fn(X[tr], y[tr], groups[tr], X[te])
            continue
        if param_grid:
            inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
            gs = GridSearchCV(clone(estimator), param_grid, cv=inner, scoring=scoring, n_jobs=-1)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            model = gs.best_estimator_
            chosen.append(gs.best_params_)
        else:
            model = clone(estimator).fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]
    return prob, chosen
