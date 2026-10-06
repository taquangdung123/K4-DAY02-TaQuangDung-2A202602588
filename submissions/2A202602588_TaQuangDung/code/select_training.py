"""Freeze a training choice using validation metrics only, before test access."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

SUB = Path(__file__).resolve().parents[1]
RUNS = SUB / "outputs" / "runs"


def main():
    rows = []
    for exp_id in ("B03", "T01", "T02", "T03", "T04"):
        path = RUNS / exp_id / "seed0" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Finish validation run {exp_id} before selecting")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("test_metrics") is not None:
            raise ValueError(f"Test accessed during validation selection: {exp_id}")
        rows.append({"exp_id": exp_id, "val_macro_f1": result["val_macro_f1"],
                     "val_top1": result["val_top1"], "best_epoch": result["best_epoch"]})
    frame = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    selected = str(frame.iloc[0].exp_id)
    base_f1 = float(frame.loc[frame.exp_id == "B03", "val_macro_f1"].iloc[0])
    if float(frame.iloc[0].val_macro_f1) - base_f1 < 0.003:
        selected = "B03"  # A sub-0.3-point change with one seed is too small to prefer extra complexity.
    destination = SUB / "outputs" / "validation_decision.md"
    if destination.exists():
        raise FileExistsError("Validation choice is already frozen; inspect it before proceeding")
    destination.write_text(
        "# Validation-only training decision\n\n"
        "All runs use fold 0, seed 0, 12 epochs, and the same ConvNeXt-Tiny backbone. "
        "No test predictions were read. A gain below 0.003 over B03 is treated as inconclusive "
        "at one seed; in that case the baseline is preferred.\n\n"
        + "```text\n" + frame.to_string(index=False) + "\n```"
        + f"\n\nSelected: **{selected}**. "
        "Inference method will be selected from validation separately; final evaluation uses "
        "one-view temperature scaling fitted on each seed's validation logits.\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"Selected {selected}; wrote {destination}")


if __name__ == "__main__":
    main()
