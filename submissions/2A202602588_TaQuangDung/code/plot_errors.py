"""Render confusion matrix and a reproducible sample of test errors after final evaluation."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval as eval_lib

NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
         "Rubber Vine", "Siam Weed", "Snake Weed", "Negative"]


def main():
    prediction = SUB / "predictions" / "F01_seed0_test.csv"
    if not prediction.exists():
        raise FileNotFoundError("Final seed-0 test prediction is required")
    frame = pd.read_csv(prediction)
    cm = sum(
        (eval_lib.confusion_matrix(part.y_true.to_numpy(), part.y_pred.to_numpy(), 9)
         for part in (pd.read_csv(SUB / "predictions" / f"F01_seed{seed}_test.csv") for seed in (0, 1, 2))),
        np.zeros((9, 9), dtype=int),
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), NAMES)
    ax.set(xlabel="Predicted", ylabel="True", title="F01, three seeds summed — DeepWeeds fold-0 test")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    out = SUB / "curves"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "F01_confusion.png", dpi=160)
    plt.close(fig)

    wrong = frame[frame.y_true != frame.y_pred].copy()
    wrong["confidence"] = wrong[[f"p{i}" for i in range(9)]].max(axis=1)
    # Fixed selection rule: prioritize Chinee Apple/Snake Weed, then highest-confidence errors.
    wrong["priority"] = wrong.y_true.isin([0, 7]).astype(int)
    selected = wrong.sort_values(["priority", "confidence"], ascending=False).head(12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        with Image.open(ROOT / "data" / "images" / row.Filename) as source:
            ax.imshow(source.convert("RGB"))
        ax.set_title(f"{NAMES[int(row.y_true)]} → {NAMES[int(row.y_pred)]}\np={row.confidence:.2f}", fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "F01_error_examples.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(cm, index=NAMES, columns=NAMES).to_csv(out / "F01_confusion.csv")


if __name__ == "__main__":
    main()
