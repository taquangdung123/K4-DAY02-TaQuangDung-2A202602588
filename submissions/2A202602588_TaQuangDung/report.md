# Part 0 — EDA và kiểm tra pipeline

## Dữ liệu và split

Đã dùng nguyên bản ba CSV fold 0 của tác giả, không sửa hay chia lại. `images.zip` có MD5 đúng `b7b30f96d466fba86016aa5a26606e0f`.

| Split | Số ảnh | Chinee Apple | Lantana | Parkinsonia | Parthenium | Prickly Acacia | Rubber Vine | Siam Weed | Snake Weed | Negative |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 10,501 | 675 | 637 | 618 | 613 | 637 | 605 | 644 | 609 | 5,463 |
| Val | 3,501 | 225 | 213 | 206 | 204 | 212 | 202 | 215 | 203 | 1,821 |
| Test | 3,507 | 226 | 213 | 207 | 205 | 213 | 202 | 215 | 204 | 1,822 |

Ba giao theo `Filename` đều rỗng; hợp có đủ 17,509 ảnh và không thiếu ảnh nào. Phân bố toàn bộ `labels.csv` có lớp lớn nhất/lớp nhỏ nhất là 9.02 (9,106/1,009), phù hợp với Table 1.

## Ảnh và EDA

Đã xem ba ảnh train mỗi lớp (27 ảnh; xem [eda_samples.png](./outputs/eda_samples.png)). Ảnh đều 256×256 RGB; `DeepWeedsDataset` chuyển sang RGB và dùng ImageNet mean/std. Trong các mẫu, nền thực địa rối và điều kiện sáng khác nhau làm việc nhận dạng theo lá/cành khó hơn; Chinee Apple và Prickly Acacia là cặp nên kiểm tra kỹ do cùng dạng bụi/cành gai. Ảnh Negative thường có đất, lá khô hoặc thảm cỏ và đôi khi có thực vật không phải mục tiêu.

## Kiểm tra pipeline

- Seed được cố định cho `random`, NumPy, PyTorch CPU/CUDA và worker DataLoader.
- CE ban đầu trên 9 lớp: 2.1921; mốc đồng đều `ln(9)`: 2.1972.
- Một batch nhỏ (8 ảnh) overfit tới loss 0.000010 sau 30 bước.
- Kiểm tra focal loss với `gamma=0` so với CE sai khác dưới `1e-6`; CutMix trả về `lambda` khớp chính xác tỉ lệ pixel thực còn nguyên.
- Kiểm tra `train()`/`eval()` đạt; BatchNorm trong backbone đóng băng vẫn ở eval trong epoch huấn luyện.

## Smoke run

Đã chạy `train.run(Config(...))` hết một epoch trên toàn bộ train fold và đánh giá val, với `resnet18` khởi tạo scratch, ảnh 96×96, batch 32, AMP. Epoch mất 57.8 giây; macro-F1 val 0.1879 (chỉ để kiểm tra pipeline, không phải kết quả baseline). Đã tạo `history.csv`, checkpoint, curve và dự đoán 3,501 ảnh val dưới `outputs/runs/P0_SMOKE/seed0/` và `outputs/predictions/`. Test loader không được tạo và test không được đánh giá.

Đây là smoke run, không thay thế thí nghiệm baseline B01 10–15 epoch với cấu hình đã chốt.

## Bước 1 — So sánh backbone

Cả năm lần chạy dùng seed 0, fold 0, pretrained ImageNet, ảnh 224×224, RandomResizedCrop + horizontal flip, ImageNet normalization, AdamW, LR backbone/head `1e-4`/`1e-3`, weight decay `0.05` (không decay norm/bias), warmup một epoch rồi cosine, CE, AMP và 12 epoch. Checkpoint được chọn theo macro-F1 val; không tạo test loader. Batch vật lý là 32 cho tất cả các backbone.

| Exp | Backbone | Tag pretrained | Params (M) | GMAC | Macro-F1 val | Top-1 val | Epoch (s) | Latency batch 1 (ms) | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | `a1_in1k` | 23.53 | 4.087 | 0.8194 | 0.8672 | 68.11 | 15.98 | 11 |
| B02 | ResNeXt-50 32×4d | `a1h_in1k` | 23.00 | 4.228 | 0.7762 | 0.8255 | 82.08 | 30.32 | 9 |
| B03 | ConvNeXt-Tiny | `in12k_ft_in1k` | 27.83 | 4.455 | **0.9677** | **0.9754** | 81.64 | 27.59 | 12 |
| B04 | DeiT-Small | `fb_in1k` | 21.67 | 4.599 | 0.9475 | 0.9640 | 62.64 | 27.93 | 12 |
| B05 | EfficientNet-B0 | `ra_in1k` | 4.02 | 0.385 | 0.8099 | 0.8569 | 45.42 | 13.17 | 9 |

`results.xlsx`, sheet `Backbones`, chứa đủ cấu hình, số liệu và đường dẫn output; mỗi run có curve trong `outputs/curves/`. Epoch time là thời gian train cộng validation. Latency chỉ là phép đo sơ bộ: batch 1 ở FP32, 3 lượt warmup, một lượt được tính giờ với đồng bộ CUDA. GMAC là ước lượng Conv/Linear/attention theo hook; không thay cho benchmark thực.

**Chọn B03 và B04 cho bước tiếp:** B03 có macro-F1 quan sát cao nhất (0.9677); B04 đứng thứ hai (0.9475), có ít tham số hơn B03 (21.67M so với 27.83M) và thời gian epoch thấp hơn (62.64s so với 81.64s), với latency gần như bằng nhau. Chênh lệch macro-F1 là 0.0202 nhưng mỗi cấu hình mới chỉ có một seed, vì vậy đây là lựa chọn sàng lọc theo val, chưa phải bằng chứng kết luận thống kê.

Diễn biến loss cho thấy B02 có dấu hiệu overfit rõ hơn: train loss cuối 0.2097 trong khi val loss đạt cực tiểu ở epoch 9 (0.4779) rồi tăng lên 0.5403; macro-F1 val cũng đạt đỉnh ở epoch 9. B01 có giảm macro-F1 nhẹ sau epoch 11; B03/B04 tiếp tục đạt macro-F1 tốt nhất tại epoch 12. B05 đạt đỉnh epoch 9 rồi gần như đi ngang. B03 có macro-F1 val epoch đầu cao nhất (0.8556), gợi ý hội tụ nhanh với tag pretrained đang dùng. GMAC không dự đoán hoàn hảo tốc độ: DeiT có GMAC cao hơn ConvNeXt nhưng epoch ngắn hơn; do khác kiến trúc, kernel và overhead, dùng thời gian/latency đo thực tế để so sánh.

## Trạng thái kiểm tra cuối cùng

Tôi đã chạy kiểm tra thực tế trên repo với lệnh:

```bash
python -m unittest discover -s tests
```

Kết quả hiện tại: 38 bài test chạy, 36 passed, 2 failed. Hai failure nằm ở `tests/test_eval.py` liên quan đến CLI `eval.py grade`/`I4a` và xuất hiện khi chấm phần tự động theo RUBRIC. Điều này cho thấy các phần Bước 3–4 chưa được chốt hoàn toàn theo tiêu chuẩn của `eval.py`, dù các bước EDA và so sánh backbone đã có dữ liệu thật và kết quả đo được từ `results.xlsx`.

Do đó, báo cáo này phản ánh những gì đã chạy và đo được thực tế trong repo hiện tại; các số liệu Part 1 và EDA là đáng tin cậy, còn phần final grading / benchmark theo `eval.py` cần tiếp tục sửa cho đến khi 2 test cuối cùng pass.
