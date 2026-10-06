"""Check reproducibility artifacts and numeric consistency after final scoring."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval as eval_lib


def main():
    missing = []
    for exp in ("B01", "B02", "B03", "B04", "B05", "T01", "T02", "T03", "T04"):
        for name in ("summary.json", "history.csv", "curves.png", "best.pt", "val_logits.npz"):
            path = SUB / "outputs" / "runs" / exp / "seed0" / name
            if not path.exists():
                missing.append(str(path.relative_to(SUB)))
    for exp in ("T00", "F01"):
        for seed in (0, 1, 2):
            path = SUB / "predictions" / f"{exp}_seed{seed}_test.csv"
            if not path.exists():
                missing.append(str(path.relative_to(SUB)))
    workbook = SUB / "results.xlsx"
    expected_sheets = {"Backbones", "Training", "Inference", "Final", "PerClass", "Latency", "Summary"}
    if workbook.exists():
        actual_sheets = set(pd.ExcelFile(workbook).sheet_names)
        missing.extend(f"sheet:{name}" for name in sorted(expected_sheets - actual_sheets))
    else:
        missing.append("results.xlsx")
    for name in ("README.md", "report.md", "curves/B03_inference_tradeoff.png",
                 "curves/F01_confusion.png", "curves/F01_error_examples.png"):
        if not (SUB / name).exists():
            missing.append(name)
    if missing:
        print("Missing artifacts:\n" + "\n".join(missing))
        raise SystemExit(1)

    final = pd.read_excel(workbook, sheet_name="Final")
    for exp in ("T00", "F01"):
        scores = []
        for seed in (0, 1, 2):
            frame = pd.read_csv(SUB / "predictions" / f"{exp}_seed{seed}_test.csv")
            y = frame.y_true.to_numpy(dtype=int)
            probs = frame[[f"p{i}" for i in range(9)]].to_numpy(dtype=float)
            metrics = eval_lib.compute_metrics(y, probs.argmax(1), probs)
            scores.append(metrics["macro_f1"])
            recorded = final.loc[(final.exp_id == exp) & (final.seed == seed), "test_macro_f1"]
            if len(recorded) != 1 or not np.isclose(recorded.iloc[0], metrics["macro_f1"], atol=1e-10):
                raise AssertionError(f"Workbook and predictions disagree: {exp} seed {seed}")
        summary = final.loc[(final.exp_id == exp) & (final.seed == "mean")]
        if len(summary) != 1 or not np.isclose(summary.test_macro_f1_mean.iloc[0], np.mean(scores), atol=1e-10):
            raise AssertionError(f"Workbook mean disagrees with predictions: {exp}")
    print("Artifact inventory and workbook prediction consistency: OK")


if __name__ == "__main__":
    main()
