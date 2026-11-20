# Website FENOMIND (tersambung ke AI)

- `index.html`: versi website yang memanggil backend FENOMIND AI. Tampilan dan alur sama dengan
  website asli; halaman **Session Overview** kini menampilkan hasil analisis.
- `index.original.html`: salinan website asli (sebelum disambungkan), untuk perbandingan.

## Memperbarui website di Netlify

1. Buka https://app.netlify.com, lalu pilih situs `resonant-crumble-c5770f`.
2. Masuk ke tab **Deploys**, lalu seret **folder `website`** (yang berisi `index.html`) ke kotak
   *"Drag and drop your site output folder here"*.
3. Tunggu status **Published**.

Kalau situsnya terhubung ke repo GitHub, cukup ganti `index.html` di repo itu dengan file ini.

## Menyambungkan ke backend

Backend harus bisa diakses lewat **HTTPS**, karena website Netlify memakai HTTPS. Pilihannya:

- **Google Colab**: jalankan sel terakhir di `notebooks/colab_demo.ipynb`. Sel itu mencetak tautan siap pakai.
- **Laptop**: jalankan backend (`uvicorn fenomind.api.app:app --port 8000`), lalu di terminal lain:
  `cloudflared tunnel --url http://localhost:8000` (pasang dulu: `winget install --id Cloudflare.cloudflared`).

Lalu buka website dengan alamat backend di parameter `api`:

```
https://resonant-crumble-c5770f.netlify.app/?api=https://xxxx.trycloudflare.com
```

Alamat ini disimpan di browser, jadi kunjungan berikutnya tidak perlu `?api=` lagi. Untuk alamat permanen,
isi `DEFAULT_API` di bagian skrip `index.html`. Tanpa alamat backend, website berjalan dalam **mode demo**
(tidak mengirim data).

Backend sudah mengizinkan domain Netlify ini (CORS). Kalau domain website berubah, set
`FENOMIND_ALLOWED_ORIGINS=https://domain-baru` saat menjalankan backend.

## Yang dianalisis

| Input di website | Dianalisis? |
|---|---|
| Jawaban teks / voice-to-text (4 prompt) | Ya, IndoBERT per jawaban, lalu dirata-rata |
| Video yang direkam di website | Hanya bila **OpenFace** terpasang di server (`FENOMIND_OPENFACE_BIN`); bila tidak, ditampilkan keterangan |
| CSV OpenFace (unggahan opsional di halaman hasil) | Ya, model visual XGBoost. Contoh di `examples/` |
| Gabungan | Rata-rata sederhana, **heuristik, belum divalidasi** |
