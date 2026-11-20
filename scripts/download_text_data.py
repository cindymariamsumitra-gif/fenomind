"""Unduh DatasetIndikasiDepresi (Budiman, 2021) dari GitHub ke data/raw/.

    python scripts/download_text_data.py

Repositori sumber tidak mencantumkan lisensi; data TIDAK disertakan di repo FENOMIND.
Sitasi: Budiman, A. (2021). DatasetIndikasiDepresi. https://github.com/andrebudiman/DatasetIndikasiDepresi
"""

from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/andrebudiman/DatasetIndikasiDepresi/c3cbd20c8dc12cd21c7df515b1bc2651cd8ecdd8/Data10k_Csv.csv"
DST = Path(__file__).resolve().parents[1] / "data" / "raw" / "indikasi_depresi" / "Data10k_Csv.csv"

if __name__ == "__main__":
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_bytes(urlopen(URL, timeout=120).read())
    print("tersimpan:", DST, DST.stat().st_size, "byte")
