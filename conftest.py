import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.visual.features import AU_C, AU_R  # noqa: E402


def make_openface(n=300, fps=30.0, success_ratio=1.0, au12=0.5, seed=0, rot_speed=0.0) -> pd.DataFrame:
    """Frame OpenFace sintetis HANYA untuk menguji kode (bukan untuk evaluasi model)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    d = {"frame": np.arange(1, n + 1), "face_id": 0, "timestamp": t,
         "confidence": np.where(np.arange(n) < n * success_ratio, 0.95, 0.1),
         "success": (np.arange(n) < n * success_ratio).astype(float)}
    for c in AU_R:
        d[c] = np.clip(rng.normal(0.3, 0.1, n), 0, 5)
    d["AU12_r"] = np.full(n, au12)
    for c in AU_C:
        d[c] = (rng.random(n) > 0.7).astype(float)
    d["AU45_c"] = (np.sin(t * 2 * np.pi * 0.3) > 0.95).astype(float)  # ~18 kedipan/menit
    d.update({"pose_Tx": rng.normal(0, 1, n), "pose_Ty": rng.normal(0, 1, n), "pose_Tz": 500 + rng.normal(0, 1, n),
              "pose_Rx": rot_speed * t, "pose_Ry": np.zeros(n), "pose_Rz": np.zeros(n),
              "gaze_angle_x": rng.normal(0, 0.1, n), "gaze_angle_y": rng.normal(0, 0.1, n)})
    base_x = np.linspace(100, 300, 68)
    base_y = np.linspace(100, 300, 68)
    for i in range(68):
        d[f"x_{i}"] = base_x[i] + rng.normal(0, 0.5, n)
        d[f"y_{i}"] = base_y[i] + rng.normal(0, 0.5, n)
    for g in ("gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z"):
        d[g] = 0.0
    return pd.DataFrame(d)


@pytest.fixture
def openface_df():
    return make_openface()
