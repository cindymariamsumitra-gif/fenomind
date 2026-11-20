# Contoh input untuk demo

| File | Sumber | Label LMVD | Durasi |
|---|---|---|---|
| `lmvd_155_label1.csv` | LMVD sampel 155 | 1 (depresi) | 102,4 detik |
| `lmvd_664_label0.csv` | LMVD sampel 664 | 0 (normal) | 67,3 detik |

- Kolom = keluaran OpenFace 2.0 yang dipakai FENOMIND (kolom lain dibuang).
- Dipilih **acak** (seed 7) dari rekaman berkualitas baik (> 95% frame valid, 60–120 detik), tanpa melihat skor model.
- **Kedua sampel termasuk data latih model visual**, jadi skornya hanya untuk mendemokan alur API,
  bukan bukti performa. Performa yang sah ada di `docs/RESULTS.md` (out-of-fold).
- Sumber & lisensi: He, L. (2024). *LMVD*. figshare. https://doi.org/10.6084/m9.figshare.25698351.v1 — Apache-2.0.

Contoh panggilan:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/screen/visual -F openface_csv=@examples/lmvd_155_label1.csv
```
