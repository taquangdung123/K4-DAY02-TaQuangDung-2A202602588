"""Save the fold-0 class distribution chart used in the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
names = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
         "Rubber Vine", "Siam Weed", "Snake Weed", "Negative"]
frame = pd.DataFrame({split: pd.read_csv(LABELS / f"{split}_subset0.csv").Label.value_counts().reindex(range(9), fill_value=0)
                      for split in ("train", "val", "test")}, index=range(9))
frame.index = names
ax = frame.plot.bar(figsize=(12, 5), color=["#0f766e", "#f59e0b", "#475569"])
ax.set(title="DeepWeeds fold 0: class distribution", ylabel="Images", xlabel="Class")
ax.tick_params(axis="x", rotation=35)
ax.figure.tight_layout()
out = SUB / "curves" / "eda_class_distribution.png"
out.parent.mkdir(exist_ok=True)
ax.figure.savefig(out, dpi=160)
plt.close(ax.figure)
