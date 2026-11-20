import io

import numpy as np
import pytest

from conftest import make_openface
from fenomind.visual.features import (
    MIN_VALID_FRAMES, REQUIRED, read_openface_csv, session_features, session_sequence, valid_frames,
)


def test_feature_vector_names_and_values(openface_df):
    f = session_features(openface_df)
    assert f["q_too_few_valid_frames"] == 0.0
    assert f["q_valid_ratio"] == pytest.approx(1.0)
    assert f["au_AU12_mean"] == pytest.approx(0.5)
    assert f["au_AU12_std"] == pytest.approx(0.0, abs=1e-6)
    assert f["au_AU12_vel"] == pytest.approx(0.0, abs=1e-6)
    assert 5 < f["blink_rate_per_min"] < 40
    assert not any(np.isnan(v) for v in f.values())


def test_head_rotation_velocity():
    f = session_features(make_openface(rot_speed=0.2))  # 0,2 rad/detik
    assert f["head_pose_Rx_vel"] == pytest.approx(0.2, rel=1e-3)


def test_low_quality_returns_only_quality_fields():
    df = make_openface(n=100, success_ratio=(MIN_VALID_FRAMES - 1) / 100)
    f = session_features(df)
    assert f["q_too_few_valid_frames"] == 1.0
    assert not any(k.startswith("au_") for k in f)


def test_read_openface_csv_tolerates_corrupt_cells(openface_df):
    df = openface_df.copy()
    df.columns = [" " + c for c in df.columns]  # OpenFace menulis spasi di depan nama kolom
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    text = buf.getvalue().splitlines()
    cells = text[5].split(",")
    cells[list(df.columns).index(" AU12_r")] = "rusak"
    text[5] = ",".join(cells)
    parsed = read_openface_csv(io.StringIO("\n".join(text)))
    assert sorted(parsed.columns) == sorted(REQUIRED)
    assert len(valid_frames(parsed)) == len(df) - 1


def test_missing_columns_rejected(openface_df):
    buf = io.StringIO()
    openface_df.drop(columns=["AU12_r"]).to_csv(buf, index=False)
    with pytest.raises(ValueError, match="tidak lengkap"):
        read_openface_csv(io.StringIO(buf.getvalue()))


def test_sequence_per_second(openface_df):
    s = session_sequence(openface_df)
    assert s.shape == (10, 22)  # 300 frame / 30 fps
