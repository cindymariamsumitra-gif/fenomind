# FENOMIND AI

> **Mau langsung demo?** [![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cindymariamsumitra-gif/fenomind/blob/main/notebooks/colab_demo.ipynb) (tanpa instalasi), atau lihat [`QUICKSTART.md`](QUICKSTART.md).

Pipeline AI dan backend inference untuk **FENOMIND**: skrining awal pola gejala depresi pada mahasiswa dari
**perilaku wajah** (visual) dan **teks jawaban** (linguistik). **Bukan alat diagnosis klinis.**

Repo ini hanya berisi AI dan API. Website FENOMIND yang sudah ada tetap dipakai dan cukup memanggil API
(lihat [`docs/INTEGRATION.md`](docs/INTEGRATION.md)).

## Status singkat

| Komponen | Status | Catatan |
|---|---|---|
| Audit dataset | ✅ selesai | [`DATASET_AUDIT.md`](DATASET_AUDIT.md) |
| Model visual (OpenFace → XGBoost) | ✅ dilatih & dievaluasi | hanya pada LMVD (vlog); konfon sumber data terdokumentasi |
| Model linguistik (IndoBERT fine-tune; cadangan TF-IDF + LR) | ✅ dilatih & dievaluasi | hanya pada tweet Indonesia 2020 |
| Pembanding (SVM, RF, 1D-CNN temporal, sentence embedding, IndoBERT) | ✅ dievaluasi | [`docs/RESULTS.md`](docs/RESULTS.md) |
| Multimodal fusion | ⚠️ **NOT TRAINED / NOT VALIDATED** | arsitektur + eksperimen siap; belum ada data berpasangan |
| Gabungan untuk tampilan | ⚠️ heuristik (rata-rata), belum divalidasi | `fusion.heuristic` di respons API |
| Explainability | ✅ | oklusi per kelompok fitur/kata + importance global |
| Inference API | ✅ | FastAPI, OpenAPI, CORS, tanpa penyimpanan |
| Validasi pada mahasiswa Indonesia | ❌ belum | syarat sebelum dipakai sungguhan |

Angka performa lengkap ada di [`docs/RESULTS.md`](docs/RESULTS.md), yang dihasilkan otomatis dari artefak eksperimen.

## Struktur

```
fenomind/
  data/        lmvd.py (dedup, grup, filter kualitas), indikasi_depresi.py (dedup, grup pengguna)
  visual/      features.py (OpenFace → 117 fitur sesi + urutan per detik), temporal.py (1D-CNN)
  text/        preprocess.py, embeddings.py, finetune.py (IndoBERT)
  evaluation/  metrics.py (metrik + bootstrap CI per grup), cv.py (CV bersarang berbasis grup)
  fusion/      model.py (LateFusion, FeatureLevelFusion, validate_pairs), experiment.py
  explain/     visual.py, text.py
  api/         app.py (FastAPI), inference.py
  resources/   safety_phrases.yaml, crisis_resources.yaml (TODO tim)
scripts/       download_*, audit_datasets, train_visual, train_text, confound_analysis, explain_models, make_report
artifacts/     model terlatih (.joblib), metadata, hasil CV, importance, status fusion
notebooks/     fusion_experiment.ipynb
docs/          RESULTS.md, MODEL_CARD.md, INTEGRATION.md, openapi.json
tests/         26 tes (fitur, data, evaluasi, fusion, API)
```

## Reproduksi dari nol

```bash
pip install -r requirements.txt
```

```bash
python scripts/download_lmvd.py --workers 8
```

```bash
python scripts/download_text_data.py
```

```bash
python scripts/audit_datasets.py
```

```bash
python scripts/train_visual.py
```

```bash
python scripts/train_text.py
```

```bash
python scripts/train_text.py --indobert-only
```

```bash
python scripts/confound_analysis.py
```

```bash
python scripts/explain_models.py
```

```bash
python scripts/make_report.py
```

```bash
python -m pytest
```

- `download_lmvd.py` membaca ZIP LMVD 16 GB per anggota lewat HTTP range, mengekstrak fitur di memori, dan
  hanya menyimpan fitur (≈ 35 MB). Data mentah tidak pernah ditulis ke disk.
- Seed tetap (42). `train_text.py --indobert-only` lama di CPU (± 11 menit per fold). Progres disimpan
  bertahap di `artifacts/text/indobert/`: state resume tiap epoch, lalu setelah tiap fold `fold_k.json`
  (prediksi + metrik), checkpoint fp16 `checkpoints/fold_k/`, `train_log.txt`, dan metrik gabungan sementara
  di `artifacts/text/results.json`. Bila terhenti, jalankan perintah yang sama untuk melanjutkan.
- `python scripts/package_results.py` membuat `fenomind-ai-hasil.zip` (kode, model, metrik) dan
  `fenomind-indobert-fold1.zip` … `fold5.zip` (checkpoint IndoBERT, ±250 MB per fold) secara atomik; skrip IndoBERT memanggilnya
  otomatis setelah setiap fold.
- Data mentah dan turunannya ada di `data/` (diabaikan git). Dataset teks tidak berlisensi eksplisit,
  sehingga tidak didistribusikan ulang.

## Menjalankan API

```bash
FENOMIND_ALLOWED_ORIGINS=http://localhost:3000 uvicorn fenomind.api.app:app --host 127.0.0.1 --port 8000
```

Buka `http://127.0.0.1:8000/docs`. Contoh:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/screen/text -H 'Content-Type: application/json' -d '{"text":"akhir-akhir ini aku merasa capek dan tidak semangat kuliah"}'
```

## Prinsip yang dijaga

1. **Tidak ada angka performa yang tidak berasal dari eksperimen.** Laporan dibuat dari `artifacts/*/results.json`.
2. **Tidak ada kebocoran yang disengaja.** Dedup sebelum split, CV berbasis grup, tuning bersarang.
3. **Tidak ada pasangan multimodal artifisial.** Fusion menolak data tanpa `participant_id` + `session_id` yang cocok.
4. **Non-diagnostik.** Keluaran API memakai bahasa skrining, menyertakan disclaimer dan status validasi,
   dan tidak memberi skor untuk input berkualitas rendah.
5. **Keselamatan.** Frasa terkait menyakiti diri memicu `safety.triggered`. Daftar layanan krisis **wajib diisi tim**.
6. **Privasi.** API tidak menyimpan apa pun; respons `no-store`.

## Sitasi dataset

- He, L. (2024). *LMVD*. figshare. https://doi.org/10.6084/m9.figshare.25698351.v1 (Apache-2.0)
- Budiman, A. (2021). *DatasetIndikasiDepresi*. https://github.com/andrebudiman/DatasetIndikasiDepresi
