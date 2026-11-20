# Hasil Eksperimen FENOMIND

> Dibuat otomatis oleh `scripts/make_report.py` dari `artifacts/*/results.json`. Semua metrik dihitung
> dari prediksi **out-of-fold** (sampel tidak pernah diprediksi oleh model yang melihatnya saat training).
> CI 95% = bootstrap per grup (1.000 resampling). Ambang keputusan 0,5.

**Ini bukan validasi klinis.** Angka di bawah hanya menunjukkan seberapa baik model membedakan label
di dataset publik masing-masing, dengan semua keterbatasan di `DATASET_AUDIT.md`.

## 1. Model visual (LMVD, fitur OpenFace)

- Data: 1600 rekaman setelah deduplikasi dan filter kualitas (positif 54.9%); 117 fitur perilaku wajah; 183 grup (proksi: blok 10 ID berurutan (LMVD tidak menyediakan ID partisipan)).
- Deduplikasi: {'conflicting_label_duplicates_removed': 14, 'same_label_duplicates_removed': 4}.
- Tuning hiperparameter bersarang (inner CV 3-fold berbasis grup); outer CV 5-fold.

### 1a. CV berbasis grup (protokol utama)

| Model | ROC-AUC [CI 95%] | Balanced acc. | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Baseline (tebak prior kelas) | 0.496 [0.410–0.574] | 0.500 | 0.549 | 0.549 | 1.000 | 0.000 | 0.709 |
| Regresi logistik | 0.788 [0.758–0.815] | 0.706 | 0.709 | 0.731 | 0.744 | 0.668 | 0.737 |
| SVM (RBF) | 0.799 [0.768–0.824] | 0.713 | 0.719 | 0.729 | 0.777 | 0.650 | 0.752 |
| Random Forest | 0.802 [0.771–0.828] | 0.719 | 0.727 | 0.729 | 0.798 | 0.640 | 0.762 |
| XGBoost **(dipakai API)** | 0.810 [0.784–0.834] | 0.731 | 0.738 | 0.741 | 0.802 | 0.659 | 0.770 |
| 1D-CNN temporal (urutan per detik) | 0.717 [0.672–0.756] | 0.652 | 0.664 | 0.667 | 0.774 | 0.529 | 0.717 |

Confusion matrix model terpilih (XGBoost):

| | Prediksi normal | Prediksi depresi/terindikasi |
|---|---|---|
| **Label normal** | 476 (TN) | 246 (FP) |
| **Label depresi/terindikasi** | 174 (FN) | 704 (TP) |

### 1b. Pembanding: CV acak (protokol asli LMVD, berpotensi bocor)

| Model | ROC-AUC [CI 95%] | Balanced acc. | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Baseline (tebak prior kelas) | 0.498 [0.473–0.524] | 0.500 | 0.549 | 0.549 | 1.000 | 0.000 | 0.709 |
| Regresi logistik | 0.795 [0.766–0.821] | 0.713 | 0.716 | 0.740 | 0.743 | 0.683 | 0.741 |
| SVM (RBF) | 0.808 [0.779–0.832] | 0.732 | 0.737 | 0.751 | 0.778 | 0.687 | 0.764 |
| Random Forest | 0.806 [0.778–0.831] | 0.726 | 0.734 | 0.732 | 0.814 | 0.637 | 0.771 |
| XGBoost | 0.818 [0.793–0.841] | 0.730 | 0.736 | 0.745 | 0.788 | 0.672 | 0.766 |
| 1D-CNN temporal (urutan per detik) | 0.734 [0.702–0.764] | 0.651 | 0.666 | 0.660 | 0.804 | 0.497 | 0.725 |

Selisih AUC XGBoost: acak 0.818 vs grup 0.810. Selisih kecil ini **tidak** membuktikan bahwa tidak ada kebocoran, karena grup hanya proksi (blok ID), bukan ID partisipan sebenarnya.

### 1c. Kontrol konfon kualitas rekaman

| Model kontrol (hanya fitur kualitas rekaman) | ROC-AUC [CI 95%] |
|---|---|
| all_samples | 0.733 [0.689–0.778] |
| after_quality_filter | 0.712 [0.673–0.749] |

Kualitas rekaman saja sudah memprediksi label jauh di atas acak. Ini menandakan perbedaan sumber atau
pengumpulan video antar kelas. AUC model wajah di dalam strata yang sama:

| Strata | n | Proporsi positif | ROC-AUC |
|---|---|---|---|
| rasio frame valid 0.5–0.9 | 419 | 0.27 | 0.786 |
| rasio frame valid 0.9–0.98 | 365 | 0.53 | 0.749 |
| rasio frame valid 0.98–1.0 | 816 | 0.70 | 0.766 |
| frame rate 24 fps | 67 | 0.45 | 0.688 |
| frame rate 25 fps | 224 | 0.66 | 0.818 |
| frame rate 30 fps | 1133 | 0.55 | 0.828 |

AUC tetap jauh di atas 0,5 di dalam strata, jadi daya pembeda tidak semata-mata berasal dari kualitas
atau frame rate. Namun konfon sumber data lain (platform, gaya vlog, kata kunci pencarian) tetap mungkin.

### 1d. Kelompok fitur terpenting (permutation importance pada fold uji)

| Kelompok fitur | Penurunan AUC bila diacak |
|---|---|
| Kecepatan perubahan action unit | +0.0412 |
| Gerak dan posisi kepala | +0.0351 |
| Frekuensi kemunculan action unit | +0.0315 |
| Bukaan mulut/mata, gerak landmark, kedipan | +0.0234 |
| Variabilitas action unit | +0.0148 |
| Intensitas action unit (rata-rata, p90) | +0.0147 |
| Arah dan gerak pandangan | +0.0077 |
| Komposit otot (senyum AU6+12, alis AU4/AU1, sudut bibir AU15) | +0.0027 |
| Laju kedipan | +0.0007 |
| Ekspresivitas keseluruhan | +0.0002 |

Fitur individual teratas: `head_translation_std` (+0.0105), `au_AU26_vel` (+0.0100), `gaze_y_mean` (+0.0089), `lmk_mouth_open_mean` (+0.0067), `lmk_eye_aspect_mean` (+0.0057), `auc_AU04_rate` (+0.0053), `auc_AU20_rate` (+0.0049), `auc_AU26_rate` (+0.0043), `au_AU12_std` (+0.0041), `lmk_eye_aspect_std` (+0.0040)

## 2. Model linguistik (DatasetIndikasiDepresi)

- Data: 3853 teks unik (dari 10082 baris), 3461 grup pengguna, positif 44.6%.
- Outer CV 5-fold berbasis grup pengguna; tuning bersarang (inner 3-fold).

| Model | ROC-AUC [CI 95%] | Balanced acc. | Accuracy | Precision | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|
| Baseline (tebak prior kelas) | 0.500 [0.484–0.515] | 0.500 | 0.554 | 0.000 | 0.000 | 1.000 | 0.000 |
| TF-IDF + regresi logistik **(dipakai API)** | 0.761 [0.745–0.775] | 0.694 | 0.696 | 0.656 | 0.672 | 0.715 | 0.664 |
| TF-IDF + Linear SVM (terkalibrasi) | 0.758 [0.742–0.773] | 0.688 | 0.697 | 0.685 | 0.597 | 0.779 | 0.638 |
| TF-IDF + Complement NB | 0.745 [0.728–0.761] | 0.672 | 0.684 | 0.674 | 0.565 | 0.780 | 0.614 |
| Sentence embedding MiniLM-L12 multibahasa + LR | 0.776 [0.760–0.791] | 0.707 | 0.707 | 0.659 | 0.712 | 0.703 | 0.684 |
| Sentence embedding multilingual-e5-small + LR | 0.750 [0.735–0.765] | 0.684 | 0.684 | 0.635 | 0.686 | 0.683 | 0.660 |

Confusion matrix model yang dipakai API (TF-IDF + regresi logistik):

| | Prediksi normal | Prediksi depresi/terindikasi |
|---|---|---|
| **Label normal** | 1526 (TN) | 607 (FP) |
| **Label depresi/terindikasi** | 564 (FN) | 1156 (TP) |

### 2a. Kata dengan bobot terbesar (model TF-IDF + LR final)

- Ke arah *terindikasi*: `pernah ada`, `pernah bisa`, `sedih`, `aku yang`, `apa aku`, `aku pernah`, `namun`, `diri`, `sedangkan`, `pernah tidak`, `pernah mau`, `aku`, `takut`, `rindu`, `tapi kenapa`, `apa`, `pernah tau`, `tidak bisa`, `rasanya`, `sakit hati`
- Ke arah *tidak terindikasi*: `terimakasih`, `pernah menyesal`, `dan`, `allah`, `dan tidak`, `beli`, `jangan`, `kasih`, `perlu`, `tapi aku`, `itu`, `dari`, `bangga`, `pernah menyesali`, `alhamdulillah`, `ayah`, `ikhlas`, `org yang`, `yakin`, `pernah bilang`

Bobot ini mencerminkan pola di tweet 2020 hasil crawling kata kunci, bukan penanda klinis.

## 3. Multimodal fusion

**Status: NOT_TRAINED.** Belum ada dataset berpasangan (rekaman wajah + teks dari partisipan dan sesi yang sama) berlabel depresi. LMVD tidak memuat teks; DatasetIndikasiDepresi tidak memuat video.

Tidak ada angka performa fusion yang dilaporkan, karena tidak ada eksperimen fusion yang valid.

## 4. Overfitting dan generalisasi

- Semua angka berasal dari data yang tidak dipakai melatih model tersebut (out-of-fold); hiperparameter
  dipilih di inner CV, jadi tidak ada seleksi model pada data uji.
- Model kompleks tidak selalu lebih baik: lihat perbandingan model temporal dan IndoBERT di atas terhadap
  model tabular/TF-IDF yang sederhana.
- Generalisasi ke mahasiswa Indonesia **belum diketahui**: populasi, bahasa, konteks perekaman, dan definisi
  label berbeda dari kedua dataset latih. Validasi eksternal wajib sebelum dipakai untuk keputusan apa pun.
