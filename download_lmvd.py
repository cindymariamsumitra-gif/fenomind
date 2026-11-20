"""Unduh LMVD (Figshare, Apache-2.0) dan ekstrak fitur visual tanpa menyimpan data mentah.

    python scripts/download_lmvd.py [--workers 8] [--limit N]

- Membaca anggota ZIP 16 GB satu per satu lewat HTTP range request (paket ``remotezip``).
- Setiap CSV OpenFace diproses di memori -> fitur sesi + urutan per detik, lalu dibuang.
- Hasil: data/processed/lmvd/features.jsonl, data/processed/lmvd/seq/<id>.npy, labels.csv
- Bisa dilanjutkan: sampel yang sudah ada di features.jsonl dilewati.
- Integritas: zipfile memeriksa CRC-32 setiap anggota saat dibaca.

Sumber: He, Lang (2024). LMVD. figshare. https://doi.org/10.6084/m9.figshare.25698351.v1
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.visual.features import read_openface_csv, session_features, session_sequence  # noqa: E402

ZIP_URL = "https://ndownloader.figshare.com/files/45872469"
LABEL_URL = "https://raw.githubusercontent.com/helang818/LMVD/main/label/label.zip"
OUT = ROOT / "data" / "processed" / "lmvd"

_zip = None


def _remote():
    global _zip
    if _zip is None:
        from remotezip import RemoteZip

        _zip = RemoteZip(ZIP_URL)
    return _zip


def process_member(name: str) -> dict:
    sample_id = Path(name).stem
    t0 = time.time()
    for attempt in range(4):
        try:
            raw = _remote().read(name)
            break
        except Exception as e:  # noqa: BLE001 - jaringan; coba lagi dengan koneksi baru
            global _zip
            _zip = None
            if attempt == 3:
                return {"id": sample_id, "error": f"{type(e).__name__}: {e}"}
            time.sleep(5 * (attempt + 1))
    n_bytes = len(raw)
    try:
        df = read_openface_csv(io.BytesIO(raw))
        del raw
        feats = session_features(df)
        seq = session_sequence(df)
    except Exception as e:  # noqa: BLE001
        return {"id": sample_id, "error": f"{type(e).__name__}: {e}"}
    np.save(OUT / "seq" / f"{sample_id}.npy", seq)
    return {"id": sample_id, "features": feats, "bytes": n_bytes, "secs": round(time.time() - t0, 1)}


def fetch_labels() -> None:
    dst = OUT / "labels.csv"
    if dst.exists():
        return
    data = urlopen(LABEL_URL, timeout=60).read()
    rows = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in z.namelist():
            if n.endswith("_Depression.csv"):
                value = z.read(n).decode().strip().splitlines()[0].strip()
                rows.append((Path(n).name.split("_")[0], int(float(value))))
    rows.sort(key=lambda r: int(r[0]))
    dst.write_text("id,label\n" + "".join(f"{i},{v}\n" for i, v in rows))
    print(f"labels: {len(rows)} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    (OUT / "seq").mkdir(parents=True, exist_ok=True)
    fetch_labels()

    from remotezip import RemoteZip

    with RemoteZip(ZIP_URL) as z:
        members = sorted(n for n in z.namelist() if n.startswith("Video_feature/") and n.endswith(".csv"))
        sizes = {i.filename: i.compress_size for i in z.infolist()}
    done = set()
    feats_path = OUT / "features.jsonl"
    if feats_path.exists():
        for line in feats_path.read_text().splitlines():
            r = json.loads(line)
            if "features" in r:
                done.add(r["id"])
    todo = [m for m in members if Path(m).stem not in done]
    # Besar dulu supaya ekor antrean tidak didominasi file raksasa
    todo.sort(key=lambda m: -sizes[m])
    if args.limit:
        todo = todo[: args.limit]
    total_bytes = sum(sizes[m] for m in todo)
    print(f"{len(members)} anggota, {len(done)} sudah selesai, {len(todo)} diproses ({total_bytes / 1e9:.1f} GB terkompresi)")

    t0, got = time.time(), 0
    with ProcessPoolExecutor(args.workers) as ex, open(feats_path, "a") as fout:
        futs = {ex.submit(process_member, m): m for m in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            got += sizes[futs[fut]]
            if i % 20 == 0 or "error" in r:
                rate = got / 1e6 / (time.time() - t0)
                eta = (total_bytes - got) / 1e6 / max(rate, 1e-6) / 60
                print(f"[{i}/{len(todo)}] {rate:.1f} MB/s, sisa ~{eta:.0f} menit"
                      + (f"  ERROR {r['id']}: {r['error']}" if "error" in r else ""), flush=True)
    print("selesai")


if __name__ == "__main__":
    main()
