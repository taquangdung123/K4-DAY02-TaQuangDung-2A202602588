"""Reusable DeepWeeds training loop; checkpoint selection uses validation macro-F1 only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import sys
import time
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch

import dataset
import losses
import model as model_lib


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if not groups:
        raise ValueError("No trainable parameters are available")
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    warmup_steps = min(max(int(cfg.warmup_epochs * steps_per_epoch), 0), total_steps)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model) -> None:
        for name, value in model.state_dict().items():
            current = value.detach()
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                shadow.copy_(current)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.shadow)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    if getattr(model, "_backbone_frozen", False):
        for module in model.modules():
            if id(module) in model._frozen_backbone_module_ids:
                module.eval()
    total_loss, total_items = 0.0, 0
    optimizer_steps, skipped_updates = 0, 0
    last_lr = optimizer.param_groups[0]["lr"]
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    accumulation_steps = cfg.grad_accum_steps
    if accumulation_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    loader_length = len(loader)
    sample_count = len(loader.dataset) if hasattr(loader, "dataset") else sum(targets.size(0) for _, targets, _ in loader)
    if getattr(loader, "drop_last", False) and sample_count % cfg.batch_size:
        sample_count -= sample_count % cfg.batch_size
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mixed_targets = None
        if cfg.mix:
            images, mixed_targets = losses.mix_batch(images, targets, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, mixed_targets) if mixed_targets else criterion(logits, targets)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_end = min(group_start + accumulation_steps, loader_length)
        sample_start = group_start * cfg.batch_size
        sample_end = min(group_end * cfg.batch_size, sample_count)
        loss_scale = targets.size(0) / (sample_end - sample_start)
        scaler.scale(loss * loss_scale).backward()
        update_now = batch_index + 1 == group_end
        if update_now:
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
                optimizer_steps += 1
                if ema is not None:
                    ema.update(model)
            else:
                skipped_updates += 1
        batch_size = targets.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        last_lr = max(group["lr"] for group in optimizer.param_groups)
    if total_items == 0:
        raise ValueError("Training loader produced no batches")
    return {
        "train_loss": total_loss / total_items,
        "lr": last_lr,
        "optimizer_steps": optimizer_steps,
        "amp_skipped_updates": skipped_updates,
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    filenames, targets_all, logits_all = [], [], []
    total_loss, total_items = 0.0, 0
    for images, targets, batch_names in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        count = targets.size(0)
        filenames.extend(str(name) for name in batch_names)
        targets_all.append(targets.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
        total_loss += float(loss) * count
        total_items += count
    if not total_items:
        raise ValueError("Evaluation loader produced no batches")
    return filenames, np.concatenate(targets_all), np.concatenate(logits_all), total_loss / total_items


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Cannot plot an empty training history")
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_macro_f1"] for row in history], label="val macro-F1")
    axes[1].set(title="Validation metric", xlabel="Epoch", ylabel="Macro-F1")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _resolve_eval():
    try:
        import eval as eval_module
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        import eval as eval_module
    return eval_module


def run(cfg: Config) -> dict:
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.grad_accum_steps < 1:
        raise ValueError("epochs, batch_size and grad_accum_steps must be positive")
    set_seed(cfg.seed)
    output_dir = run_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "torch": torch.__version__}, indent=2), encoding="utf-8"
    )

    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers,
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
        cfg.batch_size, False, num_workers=cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, dataset.build_transforms(False, cfg.img_size),
            cfg.batch_size, False, num_workers=cfg.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model_lib.build_model(cfg.backbone, pretrained=True, num_classes=dataset.NUM_CLASSES,
                                   drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    train_counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0)
    weights = None
    if cfg.loss == "ce_weighted":
        beta = 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta
        weights = losses.class_weights(train_counts.to_numpy(), beta).to(device)
    criterion = losses.build_criterion(
        cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weights,
    ).to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(
        optimizer, cfg, math.ceil(len(train_loader) / cfg.grad_accum_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None
    params_m = model_lib.count_params(network)
    gmacs = model_lib.count_gmacs(network, cfg.img_size)
    network.train()

    eval_module = _resolve_eval()
    history = []
    best_f1 = -float("inf")
    best_epoch = 0
    epoch_seconds = []
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(network, train_loader, criterion, optimizer,
                                      scheduler, scaler, cfg, device, ema)
        eval_model = network
        if ema is not None:
            import copy
            eval_model = copy.deepcopy(network)
            ema.copy_to(eval_model)
        _, y_val, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        metrics = eval_module.compute_metrics(y_val, val_probs.argmax(1), val_probs)
        elapsed = time.perf_counter() - start
        epoch_seconds.append(elapsed)
        row = {
            "epoch": epoch, **train_stats, "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
            "val_balanced_acc": metrics["balanced_acc"], "epoch_seconds": elapsed,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"epoch {epoch}/{cfg.epochs}: train_loss={row['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
              f"time={elapsed:.1f}s")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            state = eval_model.state_dict()
            torch.save({"epoch": epoch, "val_macro_f1": best_f1,
                        "model": {key: value.detach().cpu() for key, value in state.items()}},
                       checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network.load_state_dict(checkpoint["model"])
    val_names, y_val, val_logits, _ = evaluate(network, val_loader, criterion, device)
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    val_metrics = eval_module.compute_metrics(y_val, val_probs.argmax(1), val_probs)
    np.savez_compressed(output_dir / "val_logits.npz", filenames=np.asarray(val_names),
                        y_true=y_val, logits=val_logits)
    eval_module.save_predictions(pred_path(cfg, "val"), val_names, y_val, val_probs)
    test_metrics = None
    if test_loader is not None:
        test_names, y_test, test_logits, _ = evaluate(network, test_loader, criterion, device)
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        np.savez_compressed(output_dir / "test_logits.npz", filenames=np.asarray(test_names),
                            y_true=y_test, logits=test_logits)
        eval_module.save_predictions(pred_path(cfg, "test"), test_names, y_test, test_probs)
        test_metrics = eval_module.compute_metrics(y_test, test_probs.argmax(1), test_probs)

    curves_path = output_dir / "curves.png"
    plot_curves(history, curves_path, f"{cfg.exp_id} / {cfg.backbone} / seed {cfg.seed}")
    result = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"], "val_top1": val_metrics["top1"],
        "epoch_seconds_mean": float(np.mean(epoch_seconds)),
        "params_m": params_m, "gmacs_estimate": gmacs,
        "pretrained_tag": getattr(network, "pretrained_tag", None),
        "split": split_report, "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path), "history": str(output_dir / "history.csv"),
        "curves": str(curves_path), "val_predictions": str(pred_path(cfg, "val")),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def parse_overrides(pairs: list[str]) -> dict:
    hints = get_type_hints(Config)
    valid_fields = {item.name for item in fields(Config)}
    parsed = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        if key not in valid_fields:
            raise ValueError(f"Unknown Config field {key!r}; valid fields: {sorted(valid_fields)}")
        annotation = hints[key]
        union_args = get_args(annotation) if get_origin(annotation) is UnionType else ()
        allows_none = type(None) in union_args
        if raw.lower() in {"none", "null"}:
            if not allows_none:
                raise ValueError(f"{key} is not optional and cannot be set to None")
            parsed[key] = None
            continue
        value_type = next((arg for arg in union_args if arg is not type(None)), annotation)
        if value_type is bool:
            if raw.lower() not in {"true", "false"}:
                raise ValueError(f"{key} must be true or false, got {raw!r}")
            parsed[key] = raw.lower() == "true"
        elif value_type is int:
            parsed[key] = int(raw)
        elif value_type is float:
            parsed[key] = float(raw)
        else:
            parsed[key] = raw
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepWeeds experiment")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2, default=str))


if __name__ == "__main__":
    main()
