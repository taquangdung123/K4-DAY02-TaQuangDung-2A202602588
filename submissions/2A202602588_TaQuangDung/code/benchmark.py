"""Synchronized inference latency measurement; excludes preprocessing."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if warmup < 10 or iters < 50:
        raise ValueError("Need at least 10 warmup and 50 measured iterations")
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(iters):
        if sync:
            sync()
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(np.mean(samples)), "n": iters}


def _report(model, batch_size, img_size, dtype, device, warmup, iters, views):
    target = torch.device(device)
    if dtype not in {"fp32", "amp", "fp16"} or (dtype == "fp16" and target.type != "cuda"):
        raise ValueError("Invalid dtype/device")
    network = copy.deepcopy(model).to(target).eval()
    if dtype == "fp16":
        network.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=target,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    inputs = [x if i % 2 == 0 else torch.flip(x, (-1,)) for i in range(views)]

    def forward():
        with torch.inference_mode(), torch.autocast(target.type, enabled=dtype == "amp"):
            for image in inputs:
                network(image)

    result = bench(forward, warmup, iters, torch.cuda.synchronize if target.type == "cuda" else None)
    return {"gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "k_views": views,
            **result, "images_per_s": batch_size * 1000 / result["p50"],
            "torch": torch.__version__, "preprocessing_included": False}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    return _report(model, batch_size, img_size, dtype, device, warmup, iters, 1)


def tta_latency(model, k_views: int, **kw) -> dict:
    if k_views < 1:
        raise ValueError("k_views must be positive")
    return _report(model, kw.pop("batch_size", 1), kw.pop("img_size", 224),
                   kw.pop("dtype", "fp32"), kw.pop("device", "cuda"),
                   kw.pop("warmup", 10), kw.pop("iters", 100), k_views)
