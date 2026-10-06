"""Final stage only: train 3 seeds, then evaluate the untouched test split once per seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval as eval_lib
import dataset
import inference
import model as model_lib
import train

RUNS = SUB / "outputs" / "runs"
PRED = SUB / "predictions"
CURVES = SUB / "curves"


def config_from(exp_id, source, seed):
    raw = json.loads((RUNS / source / "seed0" / "config.json").read_text(encoding="utf-8"))
    raw.pop("torch", None)
    raw.update(exp_id=exp_id, seed=seed, images_dir=str(ROOT / "data" / "images"),
               labels_dir=str(ROOT / "data" / "labels"), out_dir=str(RUNS),
               pred_dir=str(PRED), save_test_predictions=False)
    return train.Config(**raw)


def evaluate_existing(checkpoint_id, config_id, output_id, seed):
    """Evaluate an already trained checkpoint with a single test image pass."""
    cfg = config_from(output_id, config_id, seed)
    model = model_lib.build_model(cfg.backbone, pretrained=False, num_classes=9,
                                  drop_rate=cfg.drop_rate, init="finetune")
    source = RUNS / checkpoint_id / f"seed{seed}"
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    test_df = pd.read_csv(ROOT / "data" / "labels" / "test_subset0.csv")
    loader = dataset.make_loader(test_df, ROOT / "data" / "images",
                                 dataset.build_transforms(False, cfg.img_size),
                                 cfg.batch_size, False, num_workers=cfg.num_workers)
    names, labels, logits = inference.predict_logits(model, loader, device)
    assert names == test_df.Filename.tolist()
    output = RUNS / output_id / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "test_logits.npz", filenames=np.asarray(names),
                        y_true=labels, logits=logits)
    if source != output:
        shutil.copy2(source / "val_logits.npz", output / "val_logits.npz")
        shutil.copy2(source / "history.csv", output / "history.csv")
    shutil.copy2(source / "curves.png", CURVES / f"{output_id}_seed{seed}.png")
    eval_lib.save_predictions(PRED / f"{output_id}_seed{seed}_test.csv", names, labels,
                              inference.apply_temperature(logits, 1))
    return cfg


def save_calibrated(exp_id, seed):
    run = RUNS / exp_id / f"seed{seed}"
    with np.load(run / "val_logits.npz") as val, np.load(run / "test_logits.npz") as test:
        t = inference.fit_temperature(val["logits"], val["y_true"])
        val_names, val_y = val["filenames"].tolist(), val["y_true"]
        test_names, test_y = test["filenames"].tolist(), test["y_true"]
        val_probs = inference.apply_temperature(val["logits"], t)
        test_probs = inference.apply_temperature(test["logits"], t)
        eval_lib.save_predictions(PRED / f"{exp_id}_seed{seed}_val.csv", val_names, val_y, val_probs)
        eval_lib.save_predictions(PRED / f"{exp_id}uncal_seed{seed}_test.csv", test_names, test_y,
                                  inference.apply_temperature(test["logits"], 1))
        eval_lib.save_predictions(PRED / f"{exp_id}_seed{seed}_test.csv", test_names, test_y, test_probs)
    (run / "temperature.json").write_text(json.dumps({"T_from_val": t}, indent=2), encoding="utf-8")
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True, choices=["B03", "T01", "T02", "T03", "T04"])
    parser.add_argument("--validation-decision", required=True,
                        help="Path to a frozen note explaining selection from val, written before test")
    args = parser.parse_args()
    decision = Path(args.validation_decision)
    if not decision.is_file():
        raise FileNotFoundError("Write and review a validation-only decision note before running test")
    match = re.search(r"Selected: \*\*(\w+)\*\*", decision.read_text(encoding="utf-8"))
    if match is None or match.group(1) != args.selected:
        raise ValueError("--selected must match the frozen validation decision")
    PRED.mkdir(exist_ok=True)
    CURVES.mkdir(exist_ok=True)
    if any(PRED.glob("F01_seed*_test.csv")) or any(PRED.glob("T00_seed*_test.csv")):
        raise RuntimeError("Final test predictions already exist; refusing a repeat test run")
    # Finish every training run before opening test images.
    for exp_id, source in (("T00", "B03"), ("F01", args.selected)):
        if exp_id == "F01" and args.selected == "B03":
            continue  # Same weights/configuration; reuse T00 logits after one test pass.
        for seed in (1, 2):
            cfg = config_from(exp_id, source, seed)
            result = train.run(cfg)
            shutil.copy2(result["curves"], CURVES / f"{exp_id}_seed{seed}.png")
    # Test is opened only from this point onward, once per distinct trained checkpoint.
    for seed in (0, 1, 2):
        evaluate_existing("B03" if seed == 0 else "T00", "B03", "T00", seed)
        if args.selected != "B03":
            evaluate_existing(args.selected if seed == 0 else "F01", args.selected, "F01", seed)
        else:
            origin = RUNS / "T00" / f"seed{seed}"
            target = RUNS / "F01" / f"seed{seed}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("val_logits.npz", "test_logits.npz", "history.csv"):
                shutil.copy2(origin / name, target / name)
            shutil.copy2(CURVES / f"T00_seed{seed}.png", CURVES / f"F01_seed{seed}.png")
    for seed in (0, 1, 2):
        print(f"F01 seed {seed}: T={save_calibrated('F01', seed):.4f}", flush=True)
    (SUB / "outputs" / "final_selection.json").write_text(json.dumps({
        "selected_training_run": args.selected, "final_inference": "temperature_scaled_one_view",
        "baseline": "B03 configuration, one view", "seeds": [0, 1, 2],
        "validation_decision_note": str(decision),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
