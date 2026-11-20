import pandas as pd

from fenomind.data.indikasi_depresi import load_indikasi_depresi
from fenomind.text.preprocess import clean_text


def test_clean_text():
    assert clean_text("Aku GAK bisa tidurrrr!!! @teman https://x.co #capek") == "aku tidak bisa tidurr capek"
    assert clean_text("Hari-hari  berat 😢") == "hari hari berat"


def test_loader_dedup_conflict_and_groups(tmp_path):
    rows = [
        ("u1", "aku capek banget", "Terindikasi Depresi"),
        ("u2", "aku capek banget", "Terindikasi Depresi"),   # salinan dari pengguna lain -> u1,u2 satu grup
        ("u3", "hari ini cerah", "Tidak Terindikasi"),
        ("u3", "makan bakso enak", "Tidak Terindikasi"),
        ("u4", "label bertentangan", "Terindikasi Depresi"),
        ("u5", "label bertentangan", "Tidak Terindikasi"),
    ]
    df = pd.DataFrame(rows, columns=["username", "cleanTweet", "sentiment"])
    df["deEmoji"] = df["cleanTweet"]
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out, stats = load_indikasi_depresi(p)
    assert stats["n_conflicting_label_texts"] == 1
    assert len(out) == 3
    g = dict(zip(out["text"], out["group"]))
    assert g["hari ini cerah"] == g["makan bakso enak"]  # pengguna sama
    assert out["text"].is_unique
