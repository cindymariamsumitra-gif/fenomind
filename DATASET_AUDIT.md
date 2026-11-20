# DATASET_AUDIT — FENOMIND

Semua angka di dokumen ini dihasilkan oleh `python scripts/audit_datasets.py` (tersimpan di
`artifacts/audit.json`) dan `python scripts/confound_analysis.py`. Tanggal audit: 19 November 2026.

## Ringkasan

| | Visual | Linguistik | Berpasangan (fusion) |
|---|---|---|---|
| Dataset | LMVD | DatasetIndikasiDepresi | **tidak ada** |
| Bahasa / populasi | vlog Weibo, Bilibili, TikTok, YouTube (bukan mahasiswa Indonesia) | tweet Indonesia 2020 (bukan jawaban mahasiswa) | — |
| Sampel dipakai | 1.600 rekaman (setelah dedup + filter kualitas) | 3.853 teks unik | 0 |
| Label | depresi / normal (dari penyusun dataset) | terindikasi / tidak terindikasi | — |
| Status | dipakai untuk melatih model visual | dipakai untuk melatih model linguistik | fusion **NOT TRAINED** |

Tidak ada partisipan atau sesi yang sama di kedua dataset. Keduanya **tidak boleh** dipasangkan
untuk melatih fusion; pasangan buatan seperti itu tidak bermakna dan akan menghasilkan angka palsu.

## Dataset yang diperiksa

| Dataset | Akses | Keputusan |
|---|---|---|
| **LMVD** (He, 2024) — figshare [10.6084/m9.figshare.25698351.v1](https://doi.org/10.6084/m9.figshare.25698351.v1), label di [github.com/helang818/LMVD](https://github.com/helang818/LMVD) | Publik, **Apache-2.0**, 16,3 GB | **Dipakai** (visual) |
| **DatasetIndikasiDepresi** (Budiman, 2021) — [github.com/andrebudiman/DatasetIndikasiDepresi](https://github.com/andrebudiman/DatasetIndikasiDepresi), commit `c3cbd20` | Publik, **tanpa lisensi eksplisit** | **Dipakai** (linguistik), tidak didistribusikan ulang |
| FVPD (Glasgow, 103 mahasiswa, AU + PHQ-9) | File data dibatasi; harus mengajukan permohonan akses | Tidak dipakai (tidak bisa diunduh) |
| DAIC-WOZ / E-DAIC | Butuh EULA | Tidak dipakai |
| D-Vlog, CMDC | Butuh permohonan/EULA | Tidak dipakai |

---

## 1. LMVD (visual)

### Struktur
- Satu ZIP: `Audio_feature/*.npy` (VGGish, 0,5 GB) dan `Video_feature/*.csv` (61,5 GB terekstrak).
- Tiap CSV video = keluaran **OpenFace 2.0** per frame, 470 kolom: `timestamp, confidence, success`,
  17 AU intensitas (`AUxx_r`), 18 AU kehadiran (`AUxx_c`), gaze (6 vektor + 2 sudut), 56 landmark mata (2D+3D),
  68 landmark wajah 2D (`x_i, y_i`), pose kepala (`pose_T*`, `pose_R*`).
- Video mentah **tidak** tersedia; hanya fitur.
- Label: `label.zip` di GitHub, satu file per sampel (`<id>_Depression.csv`), isinya 0 atau 1.
- Fitur audio tidak dipakai, karena FENOMIND hanya memakai modality visual dan teks.

### Jumlah dan distribusi

| Tahap | Sampel | Depresi | Normal |
|---|---|---|---|
| Mentah | 1.823 | 908 | 915 |
| Setelah deduplikasi | 1.805 | 899 | 906 |
| Setelah filter kualitas (≥ 50% frame valid, ≥ 30 frame) | **1.600** | 878 (54,9%) | 722 |

- Penulis melaporkan **1.475 partisipan** untuk 1.823 sampel, tetapi **tidak menyediakan ID partisipan**.
- Nilai hilang pada 117 fitur setelah filter: 0 sel.

### Temuan kualitas data

1. **Duplikat berlabel bertentangan.** Ada 10 kelompok rekaman dengan vektor fitur dan durasi identik.
   Tujuh di antaranya punya label berbeda (satu "depresi", satu "normal"). Ke-14 rekaman bertentangan
   dibuang, dan 4 salinan berlabel sama juga dibuang. Ini menunjukkan ada noise label di dataset asli.
2. **Urutan ID mengikuti label.** Proporsi label positif per 100 ID:
   `1,0 ×6 → 0,0 ×5 → 0,84 → 1,0 → 1,0 → 0,23 → 0,0 ×4`. Kode resmi LMVD memakai `StratifiedKFold` acak.
3. **Konfon kualitas rekaman.** Median rasio frame wajah valid: depresi 0,993, normal 0,872.
   Dari 102 rekaman dengan rasio frame valid < 10%, hanya 5,9% yang berlabel depresi.
   Model kontrol yang **hanya** memakai fitur kualitas (jumlah frame, durasi, rasio valid, confidence)
   mencapai **ROC-AUC 0,733** di semua sampel dan **0,712** setelah filter kualitas.
   Dugaan penyebab: video depresi dan normal dikumpulkan dari kanal/kata kunci berbeda.
4. **Kualitas rendah.** 31 rekaman memiliki < 30 frame valid; 205 rekaman < 50% frame valid.

### Penanganan
- Deduplikasi seperti di atas (`fenomind/data/lmvd.py::deduplicate`).
- Fitur kualitas (`q_*`) **dikeluarkan** dari input model.
- Filter kualitas: ≥ 50% frame valid dan ≥ 30 frame valid (frame valid: `success=1` dan `confidence ≥ 0,8`).
- Pembagian data: `StratifiedGroupKFold` dengan grup = blok 10 ID berurutan, sebagai **proksi**
  partisipan. Hasil CV acak (seperti protokol asli LMVD) dilaporkan terpisah sebagai pembanding kebocoran.
- Analisis konfon: AUC model dihitung di dalam strata kualitas dan frame rate (`artifacts/visual/confound_analysis.json`).

### Limitasi
- Tanpa ID partisipan, kebocoran antar-fold tidak bisa dihilangkan sepenuhnya.
- Populasi (pembuat vlog, mayoritas Tiongkok) dan konteks (vlog bebas) berbeda dari mahasiswa Indonesia
  yang menjawab pertanyaan di website FENOMIND.
- Definisi label "depresi" berasal dari penyusun dataset (vlog yang menyatakan diagnosis/gejala), bukan
  wawancara klinis terstandar.
- Fitur visual bergantung pada OpenFace 2.0; ekstraktor lain (MediaPipe, py-feat) **tidak setara**.

---

## 2. DatasetIndikasiDepresi (linguistik)

### Struktur
Kolom: `tweet_id, length of word, location, username, tweet, deEmoji, cleanTweet, sentiment, date`.
Kolom `sentiment` berisi label (`Terindikasi Depresi` / `Tidak Terindikasi`). Menurut README, anotasi
melibatkan psikolog; prosedur dan kesepakatan antar-anotator tidak dijelaskan.

### Jumlah dan distribusi

| Tahap | Baris | Terindikasi | Tidak |
|---|---|---|---|
| Mentah (README menyebut 10.801) | 10.082 | 4.465 | 5.617 |
| Setelah dedup + buang konflik label | **3.853** | 1.720 (44,6%) | 2.133 |

- Pengguna unik: 3.571 (mentah); 2.398 pengguna punya > 1 baris. Setelah pembersihan: 3.461 grup pengguna.
- Nilai hilang: 0 di semua kolom.
- Panjang teks setelah praproses: median 28 kata (min 4, maks 56).
- Rentang tanggal: 9 Nov – 7 Des 2020 (README menyebut mulai 18 Nov). Lokasi: Jawa Barat, Jawa Timur,
  Jawa Tengah, Sumatera Utara, Sumatera Selatan, Bali.

### Temuan kualitas data
1. **Duplikasi masif.** 6.194 dari 10.082 baris adalah salinan teks persis. Pembagian acak akan
   menempatkan teks yang sama di train dan test, sehingga performa akan terlihat jauh lebih baik.
2. **12 teks berlabel bertentangan** antar-salinan → dibuang.
3. **`tweet_id` rusak** (presisi hilang, contoh `1.33E+18`) → tidak bisa dipakai sebagai kunci.
4. **Bias kata kunci pengumpulan.** Kata "pernah" muncul di 99,9% teks; data dikumpulkan dengan kata
   kunci bernada negatif. Model bisa mempelajari gaya tweet hasil crawling, bukan gejala.
5. **Domain berbeda.** Tweet publik, bukan jawaban mahasiswa atas pertanyaan reflektif.
6. **Data pribadi.** Ada `username` dan `location`. FENOMIND hanya memakai `username` untuk
   pengelompokan, tidak menyimpannya di artefak model, dan tidak mendistribusikan ulang dataset.

### Penanganan
- Deduplikasi berdasarkan `cleanTweet`, buang konflik, grup = komponen terhubung pengguna yang berbagi teks.
- Input model = kolom `deEmoji` diproses ulang dengan `fenomind.text.preprocess.clean_text`, fungsi yang
  sama dengan saat inference (normalisasi `cleanTweet` manual milik penyusun tidak bisa direplikasi saat inference).
- Pembagian: `StratifiedGroupKFold` per grup pengguna.

### Limitasi
- Tanpa lisensi eksplisit: boleh dipakai untuk riset dengan sitasi, tetapi jangan didistribusikan ulang.
- Label "terindikasi" bukan diagnosis dan bukan skor PHQ-9.

---

## 3. Kelayakan multimodal fusion

| Syarat | LMVD × IndikasiDepresi |
|---|---|
| Partisipan sama | ✗ |
| Sesi sama | ✗ |
| Label diturunkan dengan cara yang sama | ✗ |
| Bahasa/budaya sama | ✗ |

**Kesimpulan:** tidak ada data berpasangan yang valid. Fusion diimplementasikan (`fenomind/fusion/`) dan
diuji hanya untuk kebenaran kode; status resmi **NOT TRAINED / NOT VALIDATED**
(`artifacts/fusion/status.json`).

### Dataset yang dibutuhkan untuk melatih fusion
- Mahasiswa Indonesia, idealnya ≥ 200 partisipan (≥ 60 dengan PHQ-9 ≥ 10) agar CI cukup sempit.
- Per sesi: video wajah (atau CSV OpenFace 2.0) **dan** teks jawaban dari sesi yang sama, plus PHQ-9.
- `participant_id` dan `session_id` konsisten di kedua modality; format manifest di
  `notebooks/fusion_experiment.ipynb`.
- Izin etik dan persetujuan eksplisit (data kesehatan & biometrik = data pribadi spesifik, UU 27/2022 Pasal 4).
