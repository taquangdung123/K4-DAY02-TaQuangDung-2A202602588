"""DeepWeeds split validation, transforms and data loading."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negative",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_IMAGES = 17_509


def load_split(labels_dir: str | Path, fold: int = 0):
    """Load the authors' untouched train/val/test CSVs for a fold."""
    labels_dir = Path(labels_dir)
    paths = [labels_dir / f"{split}_subset{fold}.csv" for split in ("train", "val", "test")]
    frames = [pd.read_csv(path) for path in paths]
    required = {"Filename", "Label"}
    for path, frame in zip(paths, frames):
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{path} contains missing filenames or labels")
        frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype("int64")
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Validate fold integrity, exact total coverage and image availability."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {}
    per_class = {}
    for split, frame in frames.items():
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"{split} split requires Filename and Label columns")
        if frame["Filename"].duplicated().any():
            raise ValueError(f"{split} contains duplicate filenames")
        labels = frame["Label"].astype(int)
        invalid = sorted(set(labels) - set(range(NUM_CLASSES)))
        if invalid:
            raise ValueError(f"{split} contains labels outside 0..{NUM_CLASSES - 1}: {invalid}")
        names[split] = set(frame["Filename"].astype(str))
        counts = labels.value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[split] = {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)}

    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    nonempty = {key: len(value) for key, value in overlap.items() if value}
    if nonempty:
        raise ValueError(f"Fold has filename overlap: {nonempty}")
    union = names["train"] | names["val"] | names["test"]
    if len(union) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} unique images, found {len(union)}")

    images_dir = Path(images_dir)
    missing = sorted(name for name in union if not (images_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images are missing under {images_dir}: {missing[:10]}")
    counts = {split: len(frame) for split, frame in frames.items()}
    result = {
        "n": counts,
        "per_class": per_class,
        "overlap": {key: 0 for key in overlap},
        "union": len(union),
        "missing": 0,
    }
    print(f"Split sizes: {counts}; union={len(union)}; overlap={result['overlap']}; missing=0")
    for split, split_counts in per_class.items():
        print(f"{split} per class: {split_counts}")
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Use ImageNet normalization; validation resizes to 256 then center-crops."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if aug not in {"basic", "color", "trivial", "randaug"}:
        raise ValueError(f"Unsupported augmentation: {aug}")
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    ops = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    elif aug == "trivial":
        ops.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(transforms.RandAugment())
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class DeepWeedsDataset(Dataset):
    """Return normalized image tensor, integer label and original filename."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        return tensor, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Build a deterministic-order evaluation loader or shuffled training loader."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if sampler not in (None, "balanced"):
        raise ValueError(f"Unsupported sampler: {sampler}")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    weighted_sampler = None
    shuffle = bool(train and sampler is None)
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = df["Label"].map(lambda label: 1.0 / counts[label]).to_numpy(dtype=np.float64)
        weighted_sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, sampler=weighted_sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=bool(train and len(dataset) >= batch_size and len(dataset) % batch_size == 1),
        worker_init_fn=_seed_worker, generator=generator, persistent_workers=num_workers > 0,
    )
