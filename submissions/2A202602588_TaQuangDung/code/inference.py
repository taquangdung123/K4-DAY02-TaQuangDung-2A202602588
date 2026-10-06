"""Inference helpers for the DeepWeeds experiments."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    model.eval()
    names, labels, outputs = [], [], []
    with torch.inference_mode():
        for images, targets, filenames in loader:
            images = images.to(device)
            if view is not None:
                images = view(images)
            outputs.append(model(images).float().cpu().numpy())
            names.extend(filenames)
            labels.extend(targets.tolist())
    return names, np.asarray(labels), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    h, w = x.shape[-2:]
    if not 0 < crop <= min(h, w):
        raise ValueError("crop must fit inside image")
    cy, cx = (h - crop) // 2, (w - crop) // 2
    return [x[..., y:y + crop, z:z + crop] for y, z in
            ((0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (cy, cx))]


def views_multiscale(x, sizes):
    return [F.interpolate(x, size=(int(s), int(s)), mode="bilinear", align_corners=False)
            for s in sizes]


def _arrays(values):
    arrays = [np.asarray(v, dtype=np.float64) for v in values]
    if not arrays or arrays[0].ndim != 2 or any(a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected finite arrays of identical (N, classes) shape")
    return arrays


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    arrays = _arrays(logits_per_view)
    if space == "prob":
        return np.mean([_softmax(a) for a in arrays], axis=0)
    if space == "logit":
        return _softmax(np.mean(arrays, axis=0))
    raise ValueError("space must be prob or logit")


def ensemble_probs(list_of_probs):
    arrays = _arrays(list_of_probs)
    if any((a < 0).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-3) for a in arrays):
        raise ValueError("Expected normalized probabilities")
    return np.mean(arrays, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    logits = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    labels = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    if logits.ndim != 2 or labels.shape != (len(logits),) or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Invalid validation logits or labels")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_t.detach().exp().clamp(0.01, 100))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Temperature must be positive and finite")
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model):
    fused = copy.deepcopy(model).eval()
    count = 0
    for parent in fused.modules():
        children = list(parent.named_children())
        for (conv_name, conv), (bn_name, bn) in zip(children, children[1:]):
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and bn.num_features == conv.out_channels:
                setattr(parent, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, nn.Identity())
                count += 1
    fused.fused_conv_bn_count = count
    return fused
