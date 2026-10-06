"""Keep notebook orchestration consistent with executable scripts (does not run experiments)."""
from __future__ import annotations

import json
from pathlib import Path
import re

path = Path(__file__).with_name("lab_day2.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
setup = "".join(notebook["cells"][2]["source"])
setup = re.sub(r'REPO_DIR = Path\(r"[^"]+"\)',
               'REPO_DIR = next((p for p in (Path.cwd(), *Path.cwd().parents) if (p / "GUIDE.md").is_file()), None)',
               setup)
notebook["cells"][2]["source"] = setup.splitlines(keepends=True)

sources = {
    12: '''# Bước 2: so sánh ba trục có kiểm soát trên ConvNeXt-Tiny, chỉ train/val.
import subprocess, sys
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "run_training_ablation.py")], check=True)
''',
    14: '''# Bước 3: chốt training trên val, rồi đo các phương pháp suy luận trên val.
import re, subprocess, sys
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "select_training.py")], check=True)
decision_note = SUBMISSION_DIR / "outputs" / "validation_decision.md"
selected_exp = re.search(r"Selected: \\*\\*(\\w+)\\*\\*", decision_note.read_text(encoding="utf-8")).group(1)
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "run_inference_study.py"),
                "--exp-id", selected_exp], check=True)
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "run_latency_study.py")], check=True)
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "plot_tradeoff.py"),
                "--exp-id", selected_exp], check=True)
''',
    16: '''# Bước 4: chỉ chạy sau khi validation_decision.md đã được ghi, không đổi lựa chọn sau test.
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "run_final.py"),
                "--selected", selected_exp, "--validation-decision", str(decision_note)], check=True)
''',
    17: '''# Chấm hai nhóm dự đoán test bằng eval.py gốc.
for tag in ("F01", "T00"):
    subprocess.run([sys.executable, str(REPO_DIR / "eval.py"), "score",
                    "--pred", str(SUBMISSION_DIR / "predictions" / f"{tag}_seed*_test.csv"),
                    "--test-csv", str(REPO_DIR / "data" / "labels" / "test_subset0.csv"),
                    "--labels", str(REPO_DIR / "data" / "labels" / "labels.csv"),
                    "--tag", tag, "--out", str(SUBMISSION_DIR / "outputs" / "eval")], check=True)
''',
    18: '''# Tự chấm phần I: dùng bản chưa hiệu chuẩn và val để chấm I4.
import pandas as pd
latency = pd.read_csv(SUBMISSION_DIR / "outputs" / "inference" / "latency.csv")
p95 = latency.loc[latency.configuration == "B03_fp32_b1", "p95"].iloc[0]
subprocess.run([sys.executable, str(REPO_DIR / "eval.py"), "grade",
                "--final", str(SUBMISSION_DIR / "predictions" / "F01_seed*_test.csv"),
                "--baseline", str(SUBMISSION_DIR / "predictions" / "T00_seed*_test.csv"),
                "--uncal", str(SUBMISSION_DIR / "predictions" / "F01uncal_seed*_test.csv"),
                "--final-val", str(SUBMISSION_DIR / "predictions" / "F01_seed*_val.csv"),
                "--latency-p95-ms", str(p95), "--latency-method", "proper",
                "--test-csv", str(REPO_DIR / "data" / "labels" / "test_subset0.csv"),
                "--labels", str(REPO_DIR / "data" / "labels" / "labels.csv"),
                "--out", str(SUBMISSION_DIR / "outputs" / "eval")], check=True)
''',
    20: '''# Bước 5: tạo workbook từ log và dự đoán thật.
subprocess.run([sys.executable, str(SUBMISSION_DIR / "code" / "build_results.py")], check=True)
''',
}

for index, source in sources.items():
    cell = notebook["cells"][index]
    if cell["cell_type"] != "code":
        raise ValueError(f"Expected code cell at index {index}")
    cell["source"] = source.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None

path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
