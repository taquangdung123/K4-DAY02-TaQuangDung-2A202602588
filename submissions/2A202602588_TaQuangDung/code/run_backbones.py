"""Run the five-backbone Part 1 comparison with a common T00 configuration."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import dataset
import model as model_lib
import train


OUTPUT_DIR = SUBMISSION_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
CURVES_DIR = OUTPUT_DIR / "curves"
RESULTS_PATH = SUBMISSION_DIR / "results.xlsx"

BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "efficientnet_b0"),
]


def _single_image_latency_ms(backbone: str, checkpoint_path: str, image: torch.Tensor) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model_lib.build_model(backbone, pretrained=False, num_classes=dataset.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model"])
    network = network.to(device).eval()
    image = image.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(3):
            network(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        network(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def _save_workbook(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(RESULTS_PATH, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Backbones", index=False)
        sheet = writer.sheets["Backbones"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            width = min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 48)
            sheet.column_dimensions[column_cells[0].column_letter].width = width


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    labels_dir = REPO_DIR / "data" / "labels"
    images_dir = REPO_DIR / "data" / "images"
    val_df = pd.read_csv(labels_dir / "val_subset0.csv")
    val_dataset = dataset.DeepWeedsDataset(
        val_df.head(1), images_dir, dataset.build_transforms(False, img_size=224)
    )
    val_image, _, _ = val_dataset[0]
    rows = []

    for exp_id, backbone in BACKBONES:
        print(f"\n=== {exp_id}: {backbone} ===", flush=True)
        cfg = train.Config(
            exp_id=exp_id,
            seed=0,
            fold=0,
            backbone=backbone,
            init="finetune",
            img_size=224,
            aug="basic",
            loss="ce",
            epochs=12,
            batch_size=32,
            lr_backbone=1e-4,
            lr_head=1e-3,
            weight_decay=0.05,
            warmup_epochs=1.0,
            amp=True,
            grad_accum_steps=1,
            num_workers=2,
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            out_dir=str(RUNS_DIR),
            pred_dir=str(PREDICTIONS_DIR),
            save_test_predictions=False,
        )
        result = train.run(cfg)
        curve_path = CURVES_DIR / f"{exp_id}_{backbone}.png"
        shutil.copy2(result["curves"], curve_path)
        latency = _single_image_latency_ms(backbone, result["checkpoint"], val_image)
        row = {
            "exp_id": exp_id,
            "backbone": backbone,
            "pretrained_tag": result["pretrained_tag"],
            "init": cfg.init,
            "seed": cfg.seed,
            "fold": cfg.fold,
            "epochs": cfg.epochs,
            "img_size": cfg.img_size,
            "batch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "amp": cfg.amp,
            "optimizer": "AdamW",
            "lr_backbone": cfg.lr_backbone,
            "lr_head": cfg.lr_head,
            "weight_decay": cfg.weight_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "loss": cfg.loss,
            "best_epoch": result["best_epoch"],
            "val_macro_f1": result["val_macro_f1"],
            "val_top1": result["val_top1"],
            "params_m": result["params_m"],
            "gmacs": result["gmacs_estimate"],
            "epoch_seconds_mean": result["epoch_seconds_mean"],
            "batch1_latency_ms_preliminary": latency,
            "curve": str(curve_path.relative_to(SUBMISSION_DIR)),
            "checkpoint": result["checkpoint"],
            "val_predictions": result["val_predictions"],
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }
        rows.append(row)
        _save_workbook(rows)
        print(json.dumps(row, indent=2, default=str), flush=True)

    print(f"\nSaved comparison workbook: {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
