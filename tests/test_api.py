"""Tes API dengan model hasil training (dilewati bila artefak belum ada)."""

import io

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, make_openface
from fenomind.api.app import app

HAS_VISUAL = (ROOT / "artifacts/visual/model.joblib").exists()
HAS_TEXT = (ROOT / "artifacts/text/model.joblib").exists()
client = TestClient(app)

FORBIDDEN = ["kamu depresi", "terdeteksi depresi", "didiagnosis", "kamu sedih"]


def _csv(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _no_diagnostic_language(obj):
    s = str(obj).lower()
    return not any(f in s for f in FORBIDDEN)


def test_health_and_models():
    assert client.get("/v1/health").json()["ok"] is True
    m = client.get("/v1/models").json()
    assert m["fusion"]["status"] == "NOT_TRAINED"
    assert client.get("/v1/health").headers["cache-control"] == "no-store"


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_text_screening():
    r = client.post("/v1/screen/text", json={"text": "akhir akhir ini aku capek banget dan merasa sendirian, "
                                                      "tidak ada yang peduli sama aku"})
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["probability"] <= 1 and b["quality"]["ok"]
    assert b["explanation"]["method"] == "leave_one_word_out_occlusion"
    assert b["model"]["validation_status"].startswith("VALIDATED_ON_TWEETS_ONLY")
    assert b["safety"]["triggered"] is False
    assert b["interpretation"].startswith("Pola bahasa ")
    assert _no_diagnostic_language(b)


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_text_too_short_has_no_score():
    b = client.post("/v1/screen/text", json={"text": "biasa aja"}).json()
    assert b["probability"] is None and b["quality"]["ok"] is False


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_text_safety_phrase_triggers_help():
    b = client.post("/v1/screen/text", json={"text": "kadang aku pengen mati aja rasanya, semua terasa berat"}).json()
    assert b["safety"]["triggered"] is True
    assert "fallback_text" in b["safety"]["help_resources"]


def test_text_validation():
    assert client.post("/v1/screen/text", json={"text": ""}).status_code == 422
    assert client.post("/v1/screen/text", json={"text": "x" * 5001}).status_code == 422


@pytest.mark.skipif(not HAS_VISUAL, reason="model visual belum dilatih")
def test_visual_screening_from_openface_csv():
    r = client.post("/v1/screen/visual", files={"openface_csv": ("of.csv", _csv(make_openface(n=600)), "text/csv")})
    assert r.status_code == 200
    b = r.json()
    assert 0 <= b["probability"] <= 1
    fams = {f["family"] for f in b["explanation"]["families"]}
    assert {"au_intensity", "head_pose", "gaze"} <= fams
    assert _no_diagnostic_language(b)


@pytest.mark.skipif(not HAS_VISUAL, reason="model visual belum dilatih")
def test_visual_low_quality_has_no_score():
    b = client.post("/v1/screen/visual",
                    files={"openface_csv": ("of.csv", _csv(make_openface(n=600, success_ratio=0.3)), "text/csv")}).json()
    assert b["probability"] is None and not b["quality"]["ok"]


def test_visual_rejects_non_openface_csv():
    r = client.post("/v1/screen/visual", files={"openface_csv": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422


def test_video_endpoint_without_openface_returns_501(monkeypatch):
    monkeypatch.delenv("FENOMIND_OPENFACE_BIN", raising=False)
    r = client.post("/v1/screen/visual/video", files={"video": ("v.mp4", b"0000", "video/mp4")})
    assert r.status_code == 501


@pytest.mark.skipif(not (HAS_TEXT and HAS_VISUAL), reason="model belum dilatih")
def test_multimodal_returns_unimodal_results_and_no_fused_score():
    r = client.post("/v1/screen/multimodal",
                    data={"text": "aku merasa lelah setiap hari dan tidak semangat kuliah lagi", "participant_id": "p01",
                          "session_id": "s1"},
                    files={"openface_csv": ("of.csv", _csv(make_openface(n=600)), "text/csv")})
    assert r.status_code == 200
    b = r.json()
    assert b["visual"]["probability"] is not None and b["linguistic"]["probability"] is not None
    assert b["fusion"]["status"] == "NOT_TRAINED" and b["fusion"]["combined_probability"] is None
    h = b["fusion"]["heuristic"]
    assert h["status"] == "HEURISTIC_NOT_VALIDATED" and h["modalities_used"] == ["linguistic", "visual"]
    assert h["probability"] == pytest.approx((b["visual"]["probability"] + b["linguistic"]["probability"]) / 2, abs=1e-3)
    assert b["missing_modalities"] == []


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_multimodal_missing_modality_and_id_validation():
    b = client.post("/v1/screen/multimodal", data={"text": "aku senang hari ini bisa main bareng teman teman"}).json()
    assert b["missing_modalities"] == ["visual"]
    assert b["fusion"]["heuristic"]["modalities_used"] == ["linguistic"]
    assert client.post("/v1/screen/multimodal", data={"text": "halo", "participant_id": "../etc"}).status_code == 422
    assert client.post("/v1/screen/multimodal", data={}).status_code == 422


@pytest.mark.skipif(not HAS_VISUAL, reason="model visual belum dilatih")
def test_demo_examples_are_valid_openface_csv():
    for f in sorted((ROOT / "examples").glob("*.csv")):
        with open(f, "rb") as fh:
            b = client.post("/v1/screen/visual", files={"openface_csv": (f.name, fh, "text/csv")}).json()
        assert b["quality"]["ok"] and 0 <= b["probability"] <= 1, f.name


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_session_text_only_like_website():
    answers = ["akhir akhir ini aku merasa capek dan tidak semangat kuliah setiap hari",
               "", "aku jarang tidur nyenyak dan susah fokus belajar", "biasanya aku cerita ke teman dekat"]
    r = client.post("/v1/screen/session", data={"answers": answers},
                    headers={"Origin": "https://resonant-crumble-c5770f.netlify.app"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "https://resonant-crumble-c5770f.netlify.app"
    b = r.json()
    assert [p["answered"] for p in b["per_prompt"]] == [True, False, True, True]
    scored = [p["probability"] for p in b["per_prompt"] if p.get("probability") is not None]
    assert b["linguistic"]["n_prompts_scored"] == len(scored)
    assert b["linguistic"]["probability"] == pytest.approx(sum(scored) / len(scored), abs=1e-3)
    assert b["visual"] is None and "Tidak ada input visual" in b["visual_note"]
    assert b["fusion"]["heuristic"]["modalities_used"] == ["linguistic"]
    assert _no_diagnostic_language(b)


@pytest.mark.skipif(not HAS_TEXT, reason="model teks belum dilatih")
def test_session_safety_in_any_answer_and_video_without_openface(monkeypatch):
    monkeypatch.delenv("FENOMIND_OPENFACE_BIN", raising=False)
    r = client.post("/v1/screen/session",
                    data={"answers": ["hari ini biasa saja seperti hari hari lain", "kadang aku pengen mati aja rasanya"]},
                    files=[("videos", ("p1.webm", b"0000", "video/webm"))])
    b = r.json()
    assert b["safety"]["triggered"] is True and "help_resources" in b["safety"]
    assert b["visual"] is None and "OpenFace" in b["visual_note"]


@pytest.mark.skipif(not (HAS_TEXT and HAS_VISUAL), reason="model belum dilatih")
def test_session_with_openface_csv_and_validation():
    with open(ROOT / "examples/lmvd_664_label0.csv", "rb") as f:
        b = client.post("/v1/screen/session", data={"answers": ["aku senang bisa jalan jalan sama teman teman ke pantai"]},
                        files={"openface_csv": ("of.csv", f, "text/csv")}).json()
    assert b["visual"]["probability"] is not None
    assert b["fusion"]["heuristic"]["modalities_used"] == ["linguistic", "visual"]
    assert client.post("/v1/screen/session", data={"answers": ["", " "]}).status_code == 422
    assert client.post("/v1/screen/session", data={"answers": ["x" * 5001]}).status_code == 422


def test_concat_sessions_offsets_timestamps():
    from fenomind.api.inference import concat_sessions
    a, b = make_openface(n=60), make_openface(n=30)
    c = concat_sessions([a, b])
    assert len(c) == 90 and c["timestamp"].is_monotonic_increasing
    assert c["timestamp"].iloc[60] == pytest.approx(a["timestamp"].max() - a["timestamp"].min() + 1.0)
    assert concat_sessions([]) is None
