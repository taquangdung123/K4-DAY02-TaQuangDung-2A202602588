"""Run controlled fold-0 ablations; test is never loaded at this stage."""
from __future__ import annotations

from pathlib import Path
import json
import shutil

import train

ROOT = Path(__file__).resolve().parents[3]
SUBMISSION = Path(__file__).resolve().parents[1]
RUNS = SUBMISSION / "outputs" / "runs"
PRED = SUBMISSION / "outputs" / "predictions"
CURVES = SUBMISSION / "curves"

# B03 is the existing common-formula baseline (T00). Change one variable per row.
EXPERIMENTS = [
    ("T01", "frozen", {"init": "frozen"}),
    ("T02", "color", {"aug": "color"}),
    ("T03", "smoothing", {"loss": "ls", "label_smoothing": 0.1}),
    ("T04", "color_smoothing", {"aug": "color", "loss": "ls", "label_smoothing": 0.1}),
]


def main():
    CURVES.mkdir(exist_ok=True)
    for exp_id, description, changes in EXPERIMENTS:
        output = RUNS / exp_id / "seed0" / "summary.json"
        if output.exists():
            print(f"Skipping completed {exp_id}", flush=True)
            continue
        settings = dict(
            exp_id=exp_id, seed=0, backbone="convnext_tiny", init="finetune",
            img_size=224, aug="basic", loss="ce", epochs=12, batch_size=32,
            lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05,
            warmup_epochs=1.0, amp=True, num_workers=2,
            images_dir=str(ROOT / "data" / "images"),
            labels_dir=str(ROOT / "data" / "labels"),
            out_dir=str(RUNS), pred_dir=str(PRED), save_test_predictions=False,
        )
        cfg = train.Config(**(settings | changes))
        print(f"Running {exp_id}: {description}", flush=True)
        result = train.run(cfg)
        shutil.copy2(result["curves"], CURVES / f"{exp_id}_{description}.png")
        print(json.dumps({"exp_id": exp_id, "val_macro_f1": result["val_macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
