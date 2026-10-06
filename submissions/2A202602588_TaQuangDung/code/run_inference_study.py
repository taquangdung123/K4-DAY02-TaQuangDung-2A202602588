"""Compare inference methods on validation only; never accesses test labels/images."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval as eval_lib
import benchmark
import dataset
import inference
import model as model_lib


def load_model(exp_id, backbone, device):
    path = SUB / "outputs" / "runs" / exp_id / "seed0" / "best.pt"
    net = model_lib.build_model(backbone, pretrained=False, num_classes=9)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return net.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", default="B03")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = pd.read_csv(ROOT / "data" / "labels" / "val_subset0.csv")
    loader = dataset.make_loader(val, ROOT / "data" / "images",
                                 dataset.build_transforms(False), 32, False, num_workers=2)
    loader_192 = dataset.make_loader(val, ROOT / "data" / "images",
                                     dataset.build_transforms(False, 192), 32, False, num_workers=2)
    loader_256 = dataset.make_loader(val, ROOT / "data" / "images",
                                     dataset.build_transforms(False, 256), 32, False, num_workers=2)
    model = load_model(args.exp_id, "convnext_tiny", device)
    names, y, raw = inference.predict_logits(model, loader, device)
    assert names == val.Filename.tolist()
    _, _, flipped = inference.predict_logits(model, loader, device, inference.view_hflip)
    large_names, large_y, larger = inference.predict_logits(model, loader_256, device)
    small_names, small_y, smaller = inference.predict_logits(model, loader_192, device)
    assert large_names == small_names == names
    assert np.array_equal(large_y, y) and np.array_equal(small_y, y)
    temperature = inference.fit_temperature(raw, y)
    secondary = load_model("B01", "resnet50", device)
    names_2, y_2, other = inference.predict_logits(secondary, loader, device)
    assert names_2 == names and np.array_equal(y_2, y)
    candidates = [
        ("I00", "one_view", inference.apply_temperature(raw, 1), 1),
        ("I01", "horizontal_flip_prob", inference.aggregate_views([raw, flipped]), 2),
        ("I02", "three_scale_prob", inference.aggregate_views([smaller, raw, larger]), 3),
        ("I03", "horizontal_flip_logit", inference.aggregate_views([raw, flipped], "logit"), 2),
        ("I04", "resolution_256", inference.apply_temperature(larger, 1), 1),
        ("I05", "ensemble_candidate_B01", inference.ensemble_probs([
            inference.apply_temperature(raw, 1), inference.apply_temperature(other, 1)]), 2),
        ("I07", "temperature_scaling", inference.apply_temperature(raw, temperature), 1),
    ]
    out = SUB / "outputs" / "inference" / args.exp_id
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp_id, method, probs, views in candidates:
        metrics = eval_lib.compute_metrics(y, probs.argmax(1), probs)
        rows.append({"exp_id": exp_id, "method": method, "model": args.exp_id if exp_id != "I05" else args.exp_id + "+B01",
                     "k_views": views, "val_macro_f1": metrics["macro_f1"],
                     "val_top1": metrics["top1"], "val_ece": metrics["ece"]})
    pd.DataFrame(rows).to_csv(out / "val_methods.csv", index=False)
    np.savez_compressed(out / "val_views.npz", filenames=np.asarray(names), y_true=y,
                        raw=raw, flipped=flipped, smaller=smaller, larger=larger, other=other)
    (out / "temperature.json").write_text(json.dumps({"T": temperature}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"Validation temperature: {temperature:.4f}", flush=True)


if __name__ == "__main__":
    main()
