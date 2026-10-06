"""Run the original evaluator on all final and baseline prediction files."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
PRED = SUB / "predictions"
LABELS = ROOT / "data" / "labels"
OUT = SUB / "outputs" / "eval"


def main():
    for exp in ("F01", "T00"):
        paths = sorted(PRED.glob(f"{exp}_seed*_test.csv"))
        if len(paths) != 3:
            raise FileNotFoundError(f"Expected three test predictions for {exp}; found {len(paths)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    for exp in ("F01", "T00"):
        subprocess.run([sys.executable, str(ROOT / "eval.py"), "score", "--pred",
                        str(PRED / f"{exp}_seed*_test.csv"), "--test-csv",
                        str(LABELS / "test_subset0.csv"), "--labels", str(LABELS / "labels.csv"),
                        "--tag", exp, "--out", str(OUT)], check=True, env=env)
    import pandas as pd
    latency = pd.read_csv(SUB / "outputs" / "inference" / "latency.csv")
    p95 = float(latency.loc[latency.configuration == "B03_fp32_b1", "p95"].iloc[0])
    subprocess.run([sys.executable, str(ROOT / "eval.py"), "grade",
                    "--final", str(PRED / "F01_seed*_test.csv"),
                    "--baseline", str(PRED / "T00_seed*_test.csv"),
                    "--uncal", str(PRED / "F01uncal_seed*_test.csv"),
                    "--final-val", str(PRED / "F01_seed*_val.csv"),
                    "--latency-p95-ms", str(p95), "--latency-method", "proper",
                    "--test-csv", str(LABELS / "test_subset0.csv"),
                    "--labels", str(LABELS / "labels.csv"), "--out", str(OUT)], check=True, env=env)


if __name__ == "__main__":
    main()
