# Model Card — FENOMIND v0.1

## Tujuan dan batas pemakaian

- **Tujuan:** skrining awal pola yang mirip data berlabel depresi, dari perilaku wajah (visual) dan teks
  (linguistik), sebagai bahan refleksi dan dorongan mencari bantuan.
- **Bukan:** alat diagnosis, penentu kelayakan layanan, atau dasar keputusan akademik/administratif.
- **Pengguna yang dimaksud:** mahasiswa, untuk dirinya sendiri, melalui website FENOMIND.
- **Tidak boleh:** dipakai untuk memantau orang lain tanpa persetujuan, atau hasilnya dikirim ke kampus/dosen.

## Model

| | Visual | Linguistik | Fusion |
|---|---|---|---|
| Arsitektur | XGBoost (max_depth 2, min_child_weight 5) atas 117 fitur sesi | IndoBERT-base-p1 fine-tune (3 epoch, lr 2e-5, max 96 token), dilatih ulang pada seluruh data; cadangan: TF-IDF + regresi logistik | Late fusion & feature-level fusion (belum dilatih); API: heuristik rata-rata |
| Masukan | CSV per frame OpenFace 2.0 (AU, gaze, landmark 2D, pose) | Teks Bahasa Indonesia ≥ 5 kata | Keluaran dua model di atas |
| Data latih | LMVD (vlog; 1.600 rekaman) | DatasetIndikasiDepresi (tweet 2020; 3.853 teks unik) | — |
| Status | `VALIDATED_ON_LMVD_ONLY` | `VALIDATED_ON_TWEETS_ONLY` | **`NOT_TRAINED`** |
| Explainability | Oklusi per kelompok fitur (per prediksi); permutation importance (global) | Oklusi per kata (per prediksi); koefisien kata model TF-IDF (global, pembanding) | — |

Metrik lengkap, CI 95%, dan model pembanding ada di [`RESULTS.md`](RESULTS.md).

### Alasan pemilihan model
- **Visual:** XGBoost punya AUC CV berbasis grup tertinggi di antara model tabular. Model temporal 1D-CNN
  lebih buruk, kemungkinan karena data terlalu sedikit untuk belajar dari urutan.
- **Linguistik:** IndoBERT fine-tune adalah model teks terbaik di CV berbasis grup, dan CI 95% AUC-nya tidak
  tumpang tindih dengan TF-IDF + LR (lihat `RESULTS.md`). API memakainya secara bawaan
  (`FENOMIND_TEXT_MODEL=auto`). Model final dilatih ulang pada seluruh 3.853 teks dengan konfigurasi yang sama
  dengan CV. Metrik yang dilaporkan adalah hasil CV, bukan dari data latih. TF-IDF + LR tetap tersedia
  (`FENOMIND_TEXT_MODEL=tfidf`) bila server tidak sanggup memuat model 500 MB.
- **Kalibrasi:** probabilitas IndoBERT cenderung ekstrem (mendekati 0 atau 1) dan **belum dikalibrasi**.
  Perlakukan sebagai skor peringkat, bukan peluang yang sebenarnya.

## Faktor risiko dan bias yang diketahui

1. **Konfon sumber data (visual).** Fitur kualitas rekaman saja sudah memprediksi label LMVD (AUC ≈ 0,71).
   Model wajah tetap berkinerja di dalam strata kualitas/frame rate, tetapi bisa saja masih menangkap gaya
   vlog/platform, bukan gejala.
2. **Artefak pengumpulan (teks).** Model mempelajari bigram "pernah …" (kata kunci crawling) dan
   cenderung menilai kata religius ("allah", "alhamdulillah") ke arah "tidak terindikasi". Ini bias
   dataset, bukan pengetahuan klinis.
3. **Pergeseran domain.** Tidak ada data mahasiswa Indonesia di kedua set latih. Teks latih berupa tweet,
   bukan jawaban reflektif. Video latih berupa vlog, bukan rekaman menjawab pertanyaan.
4. **Label.** Label LMVD dan IndikasiDepresi bukan hasil wawancara klinis terstandar. LMVD bahkan memuat
   rekaman identik berlabel bertentangan (sudah dibuang).
5. **Kebocoran partisipan (visual).** LMVD tidak menyediakan ID partisipan; grup berbasis blok ID hanya proksi.
6. **Ekstraktor fitur.** Model visual hanya valid untuk keluaran OpenFace 2.0.

## Ambang dan keluaran
- Ambang 0,5 atas probabilitas. Ambang belum dikalibrasi untuk populasi target, dan `screening_flag`
  bukan "positif depresi".
- Input berkualitas rendah (teks < 5 kata, < 50% frame wajah valid) **tidak** diberi skor.
- Teks yang memuat frasa terkait menyakiti diri memicu `safety.triggered`, apa pun skornya.

## Syarat sebelum dipakai mahasiswa sungguhan
- [ ] Validasi eksternal pada mahasiswa Indonesia (izin etik, PHQ-9 sebagai acuan, ID partisipan & sesi).
- [ ] Kalibrasi ambang pada data tersebut; laporkan sensitivitas/spesifisitas dengan CI.
- [ ] Daftar layanan krisis terverifikasi di `fenomind/resources/crisis_resources.yaml`.
- [ ] Tinjauan psikolog atas teks interpretasi dan frasa keselamatan.
- [ ] Melatih dan memvalidasi fusion pada data berpasangan, atau tetap menampilkan hasil per modality saja.
