"""Analisis konfon model visual: apakah daya pembeda hanya berasal dari kualitas/sumber video?

    python scripts/confound_analysis.py  ->  artifacts/visual/confound_analysis.json

AUC prediksi out-of-fold (CV berbasis grup) dihitung DI DALAM strata kualitas rekaman
dan frame rate. Bila AUC tetap tinggi di dalam strata, sinyal tidak semata dari konfon itu.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fenomind.data.lmvd import load_lmvd, quality_filter  # noqa: E402


def main() -> None:
    results = json.loads((ROOT / "artifacts/visual/results.json").read_text())
    model = results["selected_model"]
    df = quality_filter(load_lmvd())
    df["fps"] = (df["q_n_frames"] / df["q_duration_s"].clip(lower=1)).round()
    oof = pd.read_csv(ROOT / "artifacts/visual/oof_predictions.csv").set_index("id")
    p = oof.loc[df.index, f"grouped_cv__{model}"]
    out = {"model": model, "overall_auc": round(float(roc_auc_score(df["label"], p)), 3),
           "quality_only_control_auc": round(results["quality_only_control"]["after_quality_filter"]["roc_auc"], 3),
           "by_valid_ratio": [], "by_fps": []}
    for lo, hi in [(0.5, 0.9), (0.9, 0.98), (0.98, 1.01)]:
        m = (df["q_valid_ratio"] >= lo) & (df["q_valid_ratio"] < hi)
        out["by_valid_ratio"].append({"range": [lo, min(hi, 1.0)], "n": int(m.sum()),
                                      "positive_rate": round(float(df["label"][m].mean()), 3),
                                      "auc": round(float(roc_auc_score(df["label"][m], p[m])), 3)})
    for f, m in df.groupby("fps").groups.items():
        sub = df.loc[m]
        if len(sub) >= 50 and sub["label"].nunique() == 2:
            out["by_fps"].append({"fps": float(f), "n": int(len(sub)), "positive_rate": round(float(sub["label"].mean()), 3),
                                  "auc": round(float(roc_auc_score(sub["label"], p.loc[m])), 3)})
    (ROOT / "artifacts/visual/confound_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
