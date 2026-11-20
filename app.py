"""FENOMIND inference API (FastAPI).

Jalankan:  uvicorn fenomind.api.app:app --host 127.0.0.1 --port 8000
Dokumentasi interaktif: /docs   Spesifikasi: /openapi.json

Variabel lingkungan:
- FENOMIND_ALLOWED_ORIGINS   daftar origin website (dipisah koma) untuk CORS, mis. https://fenomind.example
- FENOMIND_OPENFACE_BIN      path ke OpenFace FeatureExtraction (opsional; untuk endpoint video)
- FENOMIND_MAX_UPLOAD_MB     batas unggahan (bawaan 200)

Privasi: tidak ada yang disimpan. File unggahan diproses di memori atau folder sementara
yang dihapus di blok finally. Log tidak memuat teks atau fitur pengguna.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..visual.features import read_openface_csv
from . import inference as inf

log = logging.getLogger("fenomind.api")
MAX_MB = int(os.environ.get("FENOMIND_MAX_UPLOAD_MB", "200"))
VERSION = "0.1.0"

app = FastAPI(
    title="FENOMIND Inference API",
    version=VERSION,
    description="Skrining awal pola gejala depresi (BUKAN diagnosis). Modality visual (fitur OpenFace) dan "
                "linguistik (teks Bahasa Indonesia). Fusion berstatus NOT_TRAINED.",
)
origins = [o.strip() for o in os.environ.get("FENOMIND_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Jawaban teks atau hasil voice-to-text")


def _guard(fn, *a):
    try:
        return fn(*a)
    except inf.ModelUnavailable as e:
        raise HTTPException(503, str(e)) from None


async def _read_limited(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File melebihi {MAX_MB} MB.")
    return data


def _parse_openface(data: bytes) -> pd.DataFrame:
    try:
        return read_openface_csv(io.BytesIO(data))
    except ValueError as e:
        raise HTTPException(422, f"CSV OpenFace tidak valid: {e}") from None


@app.get("/v1/health")
def health():
    return {"ok": True, "version": VERSION}


@app.get("/v1/models")
def models():
    """Status model, metrik validasi silang, dan status fusion."""
    return inf.models_status()


@app.post("/v1/screen/text")
def screen_text(body: TextIn):
    return _guard(inf.screen_text, body.text)


@app.post("/v1/screen/visual")
async def screen_visual(openface_csv: UploadFile = File(..., description="CSV keluaran OpenFace FeatureExtraction")):
    data = await _read_limited(openface_csv)
    return _guard(inf.screen_openface, _parse_openface(data))


@app.post("/v1/screen/visual/video")
async def screen_visual_video(video: UploadFile = File(...)):
    """Menjalankan OpenFace di server. Butuh FENOMIND_OPENFACE_BIN; tanpa itu mengembalikan 501."""
    binary = os.environ.get("FENOMIND_OPENFACE_BIN")
    if not binary or not Path(binary).exists():
        raise HTTPException(501, "OpenFace belum dikonfigurasi di server (FENOMIND_OPENFACE_BIN). "
                                 "Kirim CSV OpenFace ke /v1/screen/visual.")
    data = await _read_limited(video)
    tmp = Path(tempfile.mkdtemp(prefix="fenomind_"))
    try:
        src = tmp / "input.mp4"
        src.write_bytes(data)
        del data
        subprocess.run([binary, "-f", str(src), "-out_dir", str(tmp), "-2Dfp", "-pose", "-aus", "-gaze", "-q"],
                       check=True, timeout=1800, capture_output=True)
        csvs = list(tmp.glob("*.csv"))
        if not csvs:
            raise HTTPException(422, "OpenFace tidak menghasilkan fitur untuk video ini.")
        frames = read_openface_csv(csvs[0])
        return _guard(inf.screen_openface, frames)
    except subprocess.CalledProcessError:
        raise HTTPException(422, "OpenFace gagal memproses video.") from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # video dan fitur mentah selalu dihapus


@app.post("/v1/screen/multimodal")
async def screen_multimodal(
    text: str | None = Form(None, max_length=5000),
    openface_csv: UploadFile | None = File(None),
    participant_id: str | None = Form(None),
    session_id: str | None = Form(None),
):
    """Menjalankan modality yang tersedia. Skor gabungan TIDAK dihitung (fusion NOT_TRAINED)."""
    try:
        pid, sid = inf.safe_id(participant_id), inf.safe_id(session_id)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if not text and openface_csv is None:
        raise HTTPException(422, "Kirim setidaknya satu modality: text atau openface_csv.")
    out = {"participant_id": pid, "session_id": sid, "visual": None, "linguistic": None}
    if text:
        out["linguistic"] = _guard(inf.screen_text, text)
    if openface_csv is not None:
        out["visual"] = _guard(inf.screen_openface, _parse_openface(await _read_limited(openface_csv)))
    out["missing_modalities"] = [k for k in ("visual", "linguistic") if out[k] is None]
    out["fusion"] = inf.fusion_block()
    out["safety"] = (out["linguistic"] or {}).get("safety", {"triggered": False})
    out["disclaimer"] = inf.DISCLAIMER
    return out
