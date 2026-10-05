"""Model construction, freezing, optimizer groups and lightweight MAC estimates."""
from __future__ import annotations

import torch
from torch import nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    import timm

    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"init must be scratch, frozen, or finetune; got {init!r}")
    use_pretrained = init != "scratch" and pretrained
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    model.pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag")
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("model.get_classifier() must return a torch module")
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    if not head_ids:
        raise ValueError("Model classifier has no parameters; cannot freeze its backbone")
    head_module_ids = {id(module) for module in classifier.modules()}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._backbone_frozen = True
    model._frozen_backbone_module_ids = {
        id(module) for module in model.modules()
        if module is not model and id(module) not in head_module_ids
    }
    for module in model.modules():
        if id(module) in model._frozen_backbone_module_ids:
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    groups = [[], [], []]
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups[2].append(parameter)
        elif parameter.ndim <= 1:
            groups[1].append(parameter)
        else:
            groups[0].append(parameter)
    settings = [
        (lr_backbone, weight_decay),
        (lr_backbone, 0.0),
        (lr_head, weight_decay),
    ]
    return [
        {"params": parameters, "lr": lr, "weight_decay": decay}
        for parameters, (lr, decay) in zip(groups, settings) if parameters
    ]


def count_params(model) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """Estimate MACs from Conv2d, Linear and timm-style attention modules."""
    device = next(model.parameters()).device
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    macs = 0
    handles = []

    def count_conv(module, inputs, output):
        nonlocal macs
        out = output
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs += out.numel() * (module.in_channels // module.groups) * kernel

    def count_linear(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    def count_attention(module, inputs, output):
        nonlocal macs
        tokens, channels = output.shape[-2:]
        batch_windows = output.numel() // (tokens * channels)
        head_dim = channels // module.num_heads
        macs += 2 * batch_windows * module.num_heads * tokens * tokens * head_dim

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
        elif all(hasattr(module, attribute) for attribute in ("num_heads", "qkv", "proj")):
            handles.append(module.register_forward_hook(count_attention))
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states.items():
            module.training = was_training
    return macs / 1e9
