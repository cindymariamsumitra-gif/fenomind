"""Ekstraksi fitur perilaku wajah dari keluaran OpenFace 2.0 (per frame).

Masukan: tabel per frame dengan kolom standar OpenFace FeatureExtraction
(timestamp, confidence, success, AUxx_r, AUxx_c, pose_*, gaze_angle_*, x_0..x_67, y_0..y_67).
Keluaran:
- ``session_features``: satu vektor fitur per rekaman (statistik + dinamika temporal),
  semua fitur punya nama yang bisa dijelaskan.
- ``session_sequence``: urutan rata-rata per detik (untuk model temporal).

Fungsi ini dipakai SAMA PERSIS saat training (LMVD) dan saat inference (API), sehingga
tidak ada perbedaan definisi fitur antara keduanya.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AU_R = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r",
        "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
AU_C = ["AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c", "AU09_c", "AU10_c", "AU12_c",
        "AU14_c", "AU15_c", "AU17_c", "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c"]
POSE_R = ["pose_Rx", "pose_Ry", "pose_Rz"]
POSE_T = ["pose_Tx", "pose_Ty", "pose_Tz"]
GAZE = ["gaze_angle_x", "gaze_angle_y"]
LMK_X = [f"x_{i}" for i in range(68)]
LMK_Y = [f"y_{i}" for i in range(68)]
SEQ_COLS = AU_R + POSE_R + GAZE

REQUIRED = ["timestamp", "confidence", "success"] + AU_R + AU_C + POSE_R + POSE_T + GAZE + LMK_X + LMK_Y
MIN_CONFIDENCE = 0.8  # ambang umum OpenFace untuk frame yang dapat dipercaya
MIN_VALID_FRAMES = 30

# Komposit yang bisa diurai (bukan label emosi; hanya kelompok otot)
COMPOSITES = {
    "smile_AU06_AU12": ["AU06_r", "AU12_r"],
    "brow_lower_AU04": ["AU04_r"],
    "inner_brow_AU01": ["AU01_r"],
    "lip_corner_down_AU15": ["AU15_r"],
}


def usecols(columns) -> list[str]:
    """Kolom yang perlu dibaca (menghemat memori untuk file OpenFace besar)."""
    cols = {c.strip(): c for c in columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"Kolom OpenFace tidak lengkap, hilang: {missing[:10]}")
    return [cols[c] for c in REQUIRED]


def read_openface_csv(src) -> pd.DataFrame:
    header = pd.read_csv(src, nrows=0)
    if hasattr(src, "seek"):
        src.seek(0)
    df = pd.read_csv(src, usecols=usecols(header.columns), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Sebagian file memuat sel rusak (teks); jadikan NaN lalu buang frame yang tidak lengkap.
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    bad = df[REQUIRED].isna().any(axis=1)
    if bad.any():
        df.loc[bad, "success"] = 0.0
        df.loc[bad, "confidence"] = 0.0
        df["timestamp"] = df["timestamp"].interpolate(limit_direction="both")
    return df


def valid_frames(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["success"] > 0.5) & (df["confidence"] >= MIN_CONFIDENCE)]


def _velocity(values: np.ndarray, t: np.ndarray, max_gap: float = 0.5) -> np.ndarray:
    """|perubahan| per detik antar-frame valid berurutan (abaikan celah > max_gap detik)."""
    if len(values) < 2:
        return np.zeros((0,) + values.shape[1:], dtype=np.float32)
    dt = np.diff(t)
    ok = (dt > 0) & (dt <= max_gap)
    dv = np.abs(np.diff(values, axis=0))
    return dv[ok] / dt[ok].reshape(-1, *([1] * (values.ndim - 1)))


def _onsets_per_minute(binary: np.ndarray, duration_s: float) -> float:
    if len(binary) < 2 or duration_s <= 0:
        return 0.0
    onsets = np.sum((binary[1:] > 0.5) & (binary[:-1] <= 0.5))
    return float(onsets / duration_s * 60.0)


def session_features(df: pd.DataFrame) -> dict[str, float]:
    """Satu vektor fitur per rekaman. Nilai NaN bila frame valid terlalu sedikit."""
    total = len(df)
    duration = float(df["timestamp"].max() - df["timestamp"].min()) if total else 0.0
    v = valid_frames(df)
    out: dict[str, float] = {
        "q_n_frames": float(total),
        "q_duration_s": duration,
        "q_valid_ratio": float(len(v) / total) if total else 0.0,
        "q_confidence_mean": float(df["confidence"].mean()) if total else 0.0,
    }
    if len(v) < MIN_VALID_FRAMES:
        out["q_too_few_valid_frames"] = 1.0
        return out
    out["q_too_few_valid_frames"] = 0.0
    t = v["timestamp"].to_numpy(np.float64)
    valid_span = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    au = v[AU_R].to_numpy(np.float32)
    au_vel = _velocity(au, t)
    for i, c in enumerate(AU_R):
        name = c[:-2]
        out[f"au_{name}_mean"] = float(au[:, i].mean())
        out[f"au_{name}_std"] = float(au[:, i].std())
        out[f"au_{name}_p90"] = float(np.percentile(au[:, i], 90))
        out[f"au_{name}_vel"] = float(au_vel[:, i].mean()) if len(au_vel) else 0.0
    auc = v[AU_C].to_numpy(np.float32)
    for i, c in enumerate(AU_C):
        out[f"auc_{c[:-2]}_rate"] = float(auc[:, i].mean())

    for name, cols in COMPOSITES.items():
        s = v[cols].to_numpy(np.float32).mean(axis=1)
        out[f"comp_{name}_mean"] = float(s.mean())
        out[f"comp_{name}_std"] = float(s.std())
    # Variabilitas ekspresi keseluruhan: rata-rata simpangan baku AU
    out["expr_overall_variability"] = float(au.std(axis=0).mean())
    out["expr_overall_activity"] = float(au.mean())

    pose = v[POSE_R].to_numpy(np.float32)
    pose_vel = _velocity(pose, t)
    for i, c in enumerate(POSE_R):
        out[f"head_{c}_std"] = float(pose[:, i].std())
        out[f"head_{c}_vel"] = float(pose_vel[:, i].mean()) if len(pose_vel) else 0.0
    trans = v[POSE_T].to_numpy(np.float32)
    out["head_translation_std"] = float(np.linalg.norm(trans.std(axis=0)))

    gaze = v[GAZE].to_numpy(np.float32)
    gaze_vel = _velocity(gaze, t)
    for i, c in enumerate(GAZE):
        out[f"gaze_{c[-1]}_mean"] = float(gaze[:, i].mean())
        out[f"gaze_{c[-1]}_std"] = float(gaze[:, i].std())
        out[f"gaze_{c[-1]}_vel"] = float(gaze_vel[:, i].mean()) if len(gaze_vel) else 0.0

    # Landmark 2D, dinormalisasi oleh jarak antar sudut luar mata (36-45)
    lx, ly = v[LMK_X].to_numpy(np.float32), v[LMK_Y].to_numpy(np.float32)
    iod = np.hypot(lx[:, 45] - lx[:, 36], ly[:, 45] - ly[:, 36])
    iod = np.where(iod < 1e-3, np.nan, iod)
    mouth_open = np.hypot(lx[:, 66] - lx[:, 62], ly[:, 66] - ly[:, 62]) / iod
    mouth_width = np.hypot(lx[:, 54] - lx[:, 48], ly[:, 54] - ly[:, 48]) / iod

    def ear(a, b, c, d, e, f):  # eye aspect ratio
        v1 = np.hypot(lx[:, b] - lx[:, f], ly[:, b] - ly[:, f])
        v2 = np.hypot(lx[:, c] - lx[:, e], ly[:, c] - ly[:, e])
        h = np.hypot(lx[:, a] - lx[:, d], ly[:, a] - ly[:, d])
        return (v1 + v2) / (2 * np.where(h < 1e-3, np.nan, h))

    eye = (ear(36, 37, 38, 39, 40, 41) + ear(42, 43, 44, 45, 46, 47)) / 2
    for name, arr in (("mouth_open", mouth_open), ("mouth_width", mouth_width), ("eye_aspect", eye)):
        out[f"lmk_{name}_mean"] = float(np.nanmean(arr))
        out[f"lmk_{name}_std"] = float(np.nanstd(arr))
    # Energi gerak wajah: kecepatan landmark rata-rata (dinormalisasi IOD), tanpa gerak kepala global
    cx, cy = lx.mean(axis=1, keepdims=True), ly.mean(axis=1, keepdims=True)
    rel = np.concatenate([(lx - cx), (ly - cy)], axis=1) / iod[:, None]
    lm_vel = _velocity(rel, t)
    out["lmk_motion_energy"] = float(np.nanmean(lm_vel)) if len(lm_vel) else 0.0
    out["blink_rate_per_min"] = _onsets_per_minute(v["AU45_c"].to_numpy(), valid_span)
    return out


def session_sequence(df: pd.DataFrame, max_seconds: int = 600) -> np.ndarray:
    """Rata-rata per detik dari frame valid: (T, len(SEQ_COLS)) float16. Detik tanpa frame valid = NaN."""
    v = valid_frames(df)
    if v.empty:
        return np.zeros((0, len(SEQ_COLS)), dtype=np.float16)
    sec = np.floor(v["timestamp"].to_numpy() - df["timestamp"].min()).astype(int)
    sec = np.clip(sec, 0, max_seconds - 1)
    T = int(sec.max()) + 1
    vals = v[SEQ_COLS].to_numpy(np.float32)
    sums = np.zeros((T, len(SEQ_COLS)), np.float64)
    counts = np.zeros(T)
    np.add.at(sums, sec, vals)
    np.add.at(counts, sec, 1)
    with np.errstate(invalid="ignore"):
        seq = sums / counts[:, None]
    seq[counts == 0] = np.nan
    return seq.astype(np.float16)
