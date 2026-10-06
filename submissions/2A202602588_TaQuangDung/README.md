# DeepWeeds Lab Day 2 — 2A202602588 Ta Quang Dung

## Môi trường và dữ liệu

- Windows 11, NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB), PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.30, NumPy 2.5.2, pandas 3.0.6, scikit-learn 1.9.1, Pillow 12.3.0, openpyxl 3.1.5, matplotlib 3.11.2.
- Dùng 17.509 ảnh DeepWeeds tại `../../data/images`, CSV fold 0 nguyên bản tại `../../data/labels`. Không đưa ảnh và checkpoint vào Git.
- Trên Windows, đặt `PYTHONUTF8=1` trước khi chạy `eval.py`, để ghi JSON tiếng Việt bằng UTF-8.

## Thứ tự chạy lại

Từ thư mục gốc repository:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m unittest discover -s submissions/2A202602588_TaQuangDung/code -p test_inference.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/run_backbones.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/run_training_ablation.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/select_training.py
$selectedExp = [regex]::Match((Get-Content submissions/2A202602588_TaQuangDung/outputs/validation_decision.md -Raw), 'Selected: \*\*(\w+)\*\*').Groups[1].Value
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/run_inference_study.py --exp-id $selectedExp
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/run_latency_study.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/plot_tradeoff.py --exp-id $selectedExp
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/run_final.py --selected $selectedExp --validation-decision submissions/2A202602588_TaQuangDung/outputs/validation_decision.md
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/score_final.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/build_results.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/plot_eda.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/plot_errors.py
.\.venv\Scripts\python.exe submissions/2A202602588_TaQuangDung/code/audit_submission.py
```

`code/train.py --set KEY=VALUE ...` chạy một cấu hình huấn luyện bất kỳ. `run_backbones.py` ghi đè các experiment B01–B05 hiện có, vì vậy hãy sao lưu `outputs/` nếu cần giữ log cũ. `run_training_ablation.py` tự bỏ qua experiment có `summary.json` hoàn chỉnh.

`run_final.py` từ chối chạy lại nếu đã có file dự đoán test. Lựa chọn cấu hình được ghi vào `outputs/validation_decision.md` trước khi mở test. Các seed chung kết là 0, 1 và 2; temperature scaling được khớp riêng trên val của từng seed.

Bản sao quyết định val để lưu cùng bài nộp: [selection.md](selection.md).

Notebook tại [code/lab_day2.ipynb](code/lab_day2.ipynb) chứa EDA và các bước kiểm tra pipeline. Các kết quả đo được nằm trong `outputs/`; bảng so sánh và phân tích ở [results.xlsx](results.xlsx) và [report.md](report.md).

[Mở notebook từ nhánh `main` bằng Colab](https://colab.research.google.com/github/taquangdung123/K4-DAY02-TaQuangDung-2A202602588/blob/main/submissions/2A202602588_TaQuangDung/code/lab_day2.ipynb). Link này hiển thị phiên bản trên GitHub; các thay đổi local trong bài làm chỉ xuất hiện ở đó sau khi được commit và push. Trong Colab, cần clone repository rồi đặt working directory tại gốc repo trước khi chạy notebook.
