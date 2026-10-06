"""Measure forward-only latency with 10 warmups and 50 synchronized trials."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch

import benchmark
import inference
import model as model_lib

SUB = Path(__file__).resolve().parents[1]


def load(exp_id, backbone):
    model = model_lib.build_model(backbone, pretrained=False, num_classes=9)
    path = SUB / "outputs" / "runs" / exp_id / "seed0" / "best.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.eval()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for submitted latency results")
    b03 = load("B03", "convnext_tiny")
    b01 = load("B01", "resnet50")
    fused = inference.fuse_conv_bn(b01)
    with torch.inference_mode():
        images = torch.randn(2, 3, 224, 224)
        error = (b01(images) - fused(images)).abs().max().item()
    if error > 1e-4:
        raise AssertionError(f"Conv-BN fusion changed output by {error}")
    rows = []
    configs = [
        ("B03_fp32_b1", b03, 1, 224, "fp32", 1, False),
        ("B03_amp_b1", b03, 1, 224, "amp", 1, False),
        ("B03_fp32_b32", b03, 32, 224, "fp32", 1, False),
        ("B03_flip_b1", b03, 1, 224, "fp32", 2, False),
        ("B03_256_b1", b03, 1, 256, "fp32", 1, False),
        ("B01_fp32_b1", b01, 1, 224, "fp32", 1, False),
        ("B01_fused_b1", fused, 1, 224, "fp32", 1, True),
    ]
    for name, model, batch, size, dtype, k, is_fused in configs:
        measured = benchmark.tta_latency(model, k, batch_size=batch, img_size=size,
                                         dtype=dtype, iters=50) if k > 1 else benchmark.latency_report(
                                             model, batch, size, dtype=dtype, iters=50)
        row = {"configuration": name, "fused_bn": is_fused, **measured}
        rows.append(row)
        print(row, flush=True)
    b03 = b03.cuda().eval()
    b01 = b01.cuda().eval()
    multiscale_inputs = [torch.randn(1, 3, size, size, device="cuda") for size in (192, 224, 256)]
    ensemble_input = multiscale_inputs[1]

    def measure_special(name, forward, k_views):
        def timed():
            with torch.inference_mode():
                forward()
        measured = benchmark.bench(timed, warmup=10, iters=50, sync=torch.cuda.synchronize)
        row = {"configuration": name, "fused_bn": False, "gpu": torch.cuda.get_device_name(0),
               "dtype": "fp32", "batch": 1, "img_size": 224, "k_views": k_views,
               **measured, "images_per_s": 1000 / measured["p50"],
               "torch": torch.__version__, "preprocessing_included": False}
        rows.append(row)
        print(row, flush=True)

    measure_special("B03_multiscale_b1", lambda: [b03(x) for x in multiscale_inputs], 3)
    measure_special("B03_B01_ensemble_b1", lambda: (b03(ensemble_input), b01(ensemble_input)), 2)
    out = SUB / "outputs" / "inference"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "latency.csv", index=False)
    print(f"Maximum Conv-BN fusion error: {error:.2e}", flush=True)


if __name__ == "__main__":
    main()
