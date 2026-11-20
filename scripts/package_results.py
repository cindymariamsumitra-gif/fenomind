"""Kemas hasil ke ZIP (ditulis atomik: file lama tetap utuh sampai ZIP baru selesai).

    python scripts/package_results.py

- fenomind-ai-hasil.zip               : source code, model .joblib, metrik, log, dokumen (kecil)
- fenomind-indobert-fold<k>.zip       : checkpoint IndoBERT fold k (fp16, ~250 MB; dipisah per fold agar
                                        tiap file < 500 MB, batas unggah/unduh)
Disalin ke folder induk repo dan ke /mnt/user-data/outputs bila ada.
Data mentah (data/), .git, dan file resume training tidak pernah dimasukkan.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTS = [ROOT.parent, Path("/mnt/user-data/outputs")]
CKPT_DIR = ROOT / "artifacts" / "text" / "indobert" / "checkpoints"
FINAL_DIR = ROOT / "artifacts" / "text" / "indobert" / "final"
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}  # di level mana pun
SKIP_TOP = {"data"}  # hanya folder data mentah di root (BUKAN paket kode fenomind/data/)


def _iter_files(base: Path, include_ckpt: bool, only: Path | None = None):
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS or rel.parts[0] in SKIP_TOP or p.suffix in (".tmp", ".pt"):
            continue
        in_ckpt = CKPT_DIR in p.parents or FINAL_DIR in p.parents
        if in_ckpt != include_ckpt or (only is not None and only not in p.parents):
            continue
        yield p, Path("fenomind-ai") / rel


def _write_zip(name: str, include_ckpt: bool, compress: bool, only: Path | None = None) -> Path | None:
    files = list(_iter_files(ROOT, include_ckpt, only))
    if not files:
        return None
    out = ROOT.parent / name
    tmp = out.with_suffix(".zip.tmp")
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(tmp, "w", mode) as z:
        for src, arc in files:
            z.write(src, arc)
    os.replace(tmp, out)
    for d in DESTS[1:]:
        if d.is_dir():
            shutil.copy2(out, d / (name + ".tmp"))
            os.replace(d / (name + ".tmp"), d / name)
    return out


def package() -> list[Path]:
    made = [_write_zip("fenomind-ai-hasil.zip", include_ckpt=False, compress=True)]
    if CKPT_DIR.is_dir():
        folds = sorted(p for p in CKPT_DIR.iterdir() if p.is_dir())
        for fold in folds + ([FINAL_DIR] if FINAL_DIR.is_dir() else []):
            name = f"fenomind-indobert-{fold.name.replace('_', '')}.zip"  # fold1..fold5, final
            if (ROOT.parent / name).exists() and (ROOT.parent / name).stat().st_mtime > max(
                    f.stat().st_mtime for f in fold.rglob("*")):
                made.append(ROOT.parent / name)  # sudah terbaru; jangan tulis ulang 250 MB
                continue
            made.append(_write_zip(name, include_ckpt=True, compress=False, only=fold))
    for old in (ROOT.parent / "fenomind-indobert-checkpoints.zip", *(d / "fenomind-indobert-checkpoints.zip" for d in DESTS[1:])):
        old.unlink(missing_ok=True)  # format lama (satu file > 500 MB)
    return [m for m in made if m]


if __name__ == "__main__":
    for p in package():
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
