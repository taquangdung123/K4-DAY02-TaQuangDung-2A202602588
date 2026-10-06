"""Plot measured validation macro-F1 against batch-1 p95 latency."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SUB = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--exp-id", required=True)
args = parser.parse_args()
methods = pd.read_csv(SUB / "outputs" / "inference" / args.exp_id / "val_methods.csv")
latency = pd.read_csv(SUB / "outputs" / "inference" / "latency.csv").set_index("configuration")
mapping = {"I00": "B03_fp32_b1", "I01": "B03_flip_b1", "I02": "B03_multiscale_b1",
           "I03": "B03_flip_b1", "I04": "B03_256_b1", "I05": "B03_B01_ensemble_b1",
           "I07": "B03_fp32_b1"}
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in methods.iterrows():
    p95 = float(latency.loc[mapping[row.exp_id], "p95"])
    ax.scatter(p95, row.val_macro_f1, s=70)
    ax.annotate(row.exp_id, (p95, row.val_macro_f1), xytext=(5, 5), textcoords="offset points")
ax.axvline(100, color="#c2410c", linestyle="--", label="100 ms budget")
ax.set(xlabel="Batch-1 p95 latency (ms), forward only", ylabel="Validation macro-F1",
       title=f"Accuracy vs latency — {args.exp_id}")
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
out = SUB / "curves" / f"{args.exp_id}_inference_tradeoff.png"
fig.savefig(out, dpi=160)
plt.close(fig)
