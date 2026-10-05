"""Classification losses and batch-level Mixup/CutMix."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight"))
    raise ValueError(f"Unsupported loss: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - log_pt.exp()).clamp_min(0) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or torch.any(counts <= 0):
        raise ValueError("counts must be a 1D tensor of positive train class counts")
    if beta == 0:
        weights = counts.reciprocal()
    elif 0 < beta < 1:
        weights = (1 - beta) / (1 - torch.pow(torch.tensor(beta, dtype=counts.dtype), counts))
    else:
        raise ValueError("beta must be 0 or strictly between 0 and 1")
    weights = weights / weights.mean()
    return weights.float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0 or mode not in {"mixup", "cutmix"}:
        raise ValueError("alpha must be positive and mode must be mixup or cutmix")
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        return lam * x + (1 - lam) * x[perm], (y_a, y_b, lam)

    _, _, height, width = x.shape
    cut_ratio = (1 - lam) ** 0.5
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
