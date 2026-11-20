import numpy as np
import pytest

from fenomind.evaluation.cv import assert_no_group_leak, outer_splits
from fenomind.evaluation.metrics import binary_metrics, bootstrap_ci


def test_binary_metrics_known_values():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.2, 0.3]
    m = binary_metrics(y, p)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["recall_sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_grouped_splits_never_leak():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    y = (groups % 2).astype(int)
    splits = outer_splits(y, groups, 5, seed=1, grouped=True)
    assert_no_group_leak(groups, splits)
    leaky = outer_splits(y, groups, 5, seed=1, grouped=False)
    with pytest.raises(AssertionError):
        assert_no_group_leak(groups, leaky)


def test_cluster_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, groups=np.arange(300) // 3, n_boot=200)
    est = binary_metrics(y, p)["roc_auc"]
    assert ci["roc_auc"][0] <= est <= ci["roc_auc"][1]
