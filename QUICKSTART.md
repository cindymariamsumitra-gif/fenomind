# QUICKSTART — menjalankan FENOMIND AI untuk demo

Tidak perlu mengunduh dataset atau melatih ulang: model sudah ada di folder `artifacts/`.

**Paling mudah: Google Colab** [![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cindymariamsumitra-gif/fenomind/blob/main/notebooks/colab_demo.ipynb). Buka, lalu jalankan sel dari atas ke bawah.

## 1. Pasang (sekali)

Butuh Python 3.11 atau lebih baru.

```bash
pip install -r requirements.txt
```

**Model teks IndoBERT** (250 MB) tidak ada di repo karena melebihi batas 100 MB GitHub.
Unduh `fenomind-indobert-final.zip` dari halaman **Releases → V0.1**, ekstrak, lalu salin folder
`final` di dalamnya (`fenomind-ai/artifacts/text/indobert/final/`) ke folder proyek ini, sehingga ada:

```
artifacts/text/indobert/final/model.safetensors
```

Tanpa folder itu, API otomatis memakai model teks TF-IDF yang lebih ringan. Cek backend yang aktif di
`GET /v1/models` (kolom `text.name`: `indobert_finetune` atau `tfidf_logreg`).

## 2. Jalankan server

```bash
uvicorn fenomind.api.app:app --host 127.0.0.1 --port 8000
```

Buka **http://127.0.0.1:8000/docs**. Di sana semua endpoint bisa dicoba lewat tombol *Try it out*.

Bila website berjalan di alamat lain, izinkan alamatnya (CORS):

```bash
FENOMIND_ALLOWED_ORIGINS=http://localhost:3000 uvicorn fenomind.api.app:app --host 127.0.0.1 --port 8000
```

Di Windows (PowerShell): `$env:FENOMIND_ALLOWED_ORIGINS="http://localhost:3000"` lalu jalankan uvicorn.

## 3. Coba

Teks:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/screen/text -H "Content-Type: application/json" -d "{\"text\":\"akhir-akhir ini aku merasa capek dan tidak semangat kuliah\"}"
```

Visual (contoh CSV OpenFace di `examples/`):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/screen/visual -F openface_csv=@examples/lmvd_155_label1.csv
```

Multimodal (teks + visual sekaligus):

```bash
curl -s -X POST http://127.0.0.1:8000/v1/screen/multimodal -F text="aku sering merasa lelah dan sendirian akhir-akhir ini" -F openface_csv=@examples/lmvd_664_label0.csv
```

## 4. Yang ditampilkan di website

- Hasil per modality: `visual.probability`, `linguistic.probability`, `interpretation`, dan `explanation`.
- Gabungan: `fusion.heuristic.probability`, yaitu **rata-rata sederhana, belum divalidasi**. Selalu beri label
  "gabungan sederhana" dan tampilkan bersama hasil per modality.
- Selalu tampilkan `disclaimer`. Bila `safety.triggered` bernilai `true`, tampilkan layar bantuan lebih dulu.

Detail kontrak API ada di [`docs/INTEGRATION.md`](docs/INTEGRATION.md); metrik di [`docs/RESULTS.md`](docs/RESULTS.md).

## Status yang perlu disampaikan saat presentasi

| Komponen | Status |
|---|---|
| Model visual (OpenFace → XGBoost) | Dilatih & dievaluasi di LMVD (vlog), AUC 0,81 |
| Model teks (IndoBERT) | Dilatih & dievaluasi di tweet Indonesia, AUC 0,81 |
| Fusion terlatih | **NOT TRAINED**: belum ada data video + teks dari orang yang sama |
| Gabungan di API | Heuristik rata-rata, **belum divalidasi** |
| Validasi pada mahasiswa Indonesia | **Belum**; ini langkah berikutnya |

FENOMIND adalah alat skrining awal, **bukan alat diagnosis**.
