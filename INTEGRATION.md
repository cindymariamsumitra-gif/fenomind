# Integrasi dengan website FENOMIND

Backend ini **tidak** mengubah frontend. Website yang sudah ada cukup memanggil endpoint HTTP di bawah.
Spesifikasi lengkap (OpenAPI 3) tersedia saat server berjalan di `/openapi.json`, dan dokumentasi
interaktif di `/docs`. Salinan statisnya ada di `docs/openapi.json`.

## Menjalankan server

```bash
pip install -r requirements.txt
```

```bash
FENOMIND_ALLOWED_ORIGINS=https://alamat-website-fenomind.example uvicorn fenomind.api.app:app --host 127.0.0.1 --port 8000
```

| Variabel | Fungsi | Bawaan |
|---|---|---|
| `FENOMIND_ALLOWED_ORIGINS` | Origin website yang boleh memanggil API (CORS), dipisah koma | `http://localhost:3000` |
| `FENOMIND_OPENFACE_BIN` | Path ke `FeatureExtraction` OpenFace 2.0 untuk endpoint video | tidak diset → endpoint video 501 |
| `FENOMIND_MAX_UPLOAD_MB` | Batas ukuran unggahan | 200 |
| `HF_HUB_OFFLINE` | Set `1` setelah model diunduh agar tidak ada akses jaringan | — |

Gunakan HTTPS (reverse proxy) bila API diakses lewat internet; data kesehatan dan biometrik termasuk
data pribadi spesifik (UU 27/2022 Pasal 4).

## Endpoint

| Metode | Path | Masukan | Keluaran |
|---|---|---|---|
| GET | `/v1/health` | — | `{ok, version}` |
| GET | `/v1/models` | — | status model, metrik CV, status validasi, status fusion |
| POST | `/v1/screen/text` | JSON `{text}` (≤ 5.000 karakter) | hasil linguistik |
| POST | `/v1/screen/visual` | multipart `openface_csv` (CSV OpenFace 2.0) | hasil visual |
| POST | `/v1/screen/visual/video` | multipart `video` | hasil visual (butuh OpenFace di server) |
| POST | `/v1/screen/multimodal` | multipart: `text?`, `openface_csv?`, `participant_id?`, `session_id?` | hasil per modality + blok fusion |

### Bentuk hasil per modality

```json
{
  "modality": "linguistic",
  "probability": 0.62,
  "screening_flag": true,
  "threshold": 0.5,
  "interpretation": "Pola bahasa pada input ini lebih mirip ... Ini bukan diagnosis; ...",
  "quality": {"ok": true, "issues": [], "n_words": 24},
  "explanation": {
    "method": "leave_one_word_out_occlusion",
    "top_increasing": [{"word": "...", "contribution": 0.05}],
    "top_decreasing": [{"word": "...", "contribution": -0.03}]
  },
  "safety": {"triggered": false},
  "model": {"name": "tfidf_logreg", "validation_status": "VALIDATED_ON_TWEETS_ONLY — ...", "cv_metrics": {...}},
  "disclaimer": "FENOMIND adalah alat skrining awal, bukan alat diagnosis. ..."
}
```

Nilai contoh di atas hanya ilustrasi bentuk, bukan hasil model.

Aturan yang wajib diikuti frontend:

1. **`probability: null`** artinya kualitas input tidak memenuhi syarat. Tampilkan `quality.issues`, jangan tampilkan skor.
2. **`safety.triggered: true`** artinya teks memuat frasa terkait menyakiti diri atau kematian. Tampilkan layar
   bantuan (`safety.help_resources`) **sebelum** hasil lain. Daftar layanan ada di
   `fenomind/resources/crisis_resources.yaml` dan **wajib diisi serta diverifikasi tim**.
3. Selalu tampilkan `disclaimer` dan `model.validation_status`.
4. Jangan menulis "kamu depresi" atau label emosi dari wajah. Pakai teks `interpretation` yang disediakan.
5. Untuk multimodal, `fusion.combined_probability` selalu `null` selama `fusion.status` = `NOT_TRAINED`.
   Tampilkan hasil per modality secara terpisah, jangan menggabungkannya sendiri di frontend.

## Contoh pemanggilan dari website (JavaScript)

```js
const API = "https://api-fenomind.example"; // alamat backend

export async function screenText(text) {
  const r = await fetch(`${API}/v1/screen/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.status);
  return r.json();
}

export async function screenMultimodal({ text, openfaceCsvFile, participantId, sessionId }) {
  const fd = new FormData();
  if (text) fd.append("text", text);
  if (openfaceCsvFile) fd.append("openface_csv", openfaceCsvFile);
  if (participantId) fd.append("participant_id", participantId);
  if (sessionId) fd.append("session_id", sessionId);
  const r = await fetch(`${API}/v1/screen/multimodal`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail ?? r.status);
  return r.json();
}
```

Teks dari fitur voice-to-text di browser bisa langsung dikirim ke `/v1/screen/text`.

## Modality visual: dari video ke fitur

Model visual dilatih pada keluaran **OpenFace 2.0** (`FeatureExtraction`). Ekstraktor lain (MediaPipe,
py-feat) menghasilkan nilai dengan skala berbeda dan **tidak boleh** dipakai tanpa melatih ulang model.

Pilihan alur:
- **A. OpenFace di server.** Set `FENOMIND_OPENFACE_BIN`, lalu website mengunggah video ke
  `/v1/screen/visual/video`. Video dihapus segera setelah fitur diekstrak.
- **B. OpenFace di perangkat/peneliti.** Jalankan
  `FeatureExtraction -f video.mp4 -out_dir out -2Dfp -pose -aus -gaze`, lalu unggah CSV-nya ke `/v1/screen/visual`.

Syarat kualitas: ≥ 30 frame dengan wajah terdeteksi yakin, dan ≥ 50% frame valid. Durasi yang
disarankan ≥ 60 detik, karena rekaman latih LMVD rata-rata beberapa menit.

## Privasi
- API tidak menyimpan teks, video, CSV, atau hasil. Respons memakai `Cache-Control: no-store`.
- Log tidak memuat isi teks atau fitur.
- Endpoint video menulis file ke folder sementara dan menghapusnya di blok `finally`.
