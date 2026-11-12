# Train LoRA từ base trên Colab

1. Nén checkout **có các file mới này** thành `repo.zip`, bỏ `.venv*`, `models/`, `runs/` và dữ liệu public/private. Đặt `repo.zip` và `train.zip` của ban tổ chức trong **MyDrive/datamate/**. Zip train có thể chứa `train/`, `data/train/` hoặc trực tiếp `questions.jsonl`, `answers.jsonl`, `rulebook.md`, `calendar.csv`, `data/`.
2. Mở `notebooks/colab_train_from_base.ipynb` bằng Colab. Chọn GPU **A100 / H100 / G4**, RAM cao, Python >=3.12. Không dùng T4/P100. Drive cần chỗ cho base (~20 GB), cache package và checkpoint; kiểm tra dung lượng trống trước.
3. Chạy ô mount Drive, sửa **một ô tham số**, rồi chạy ô cuối. `OUTPUT` mặc định là `MyDrive/datamate/from_base`. Notebook stream log vào `OUTPUT/colab.log` và **trả runtime khi thành công hoặc lỗi**. Lần đầu có thể đặt `STAGES="setup,preflight"` để đo tốc độ, rồi chạy lại với `"all"`.

Không tải LFS, không dùng `adapters/sql_rft_v3`, không có `--lora`, `--lora-thinking` hoặc `--init-adapter`. Model là **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` đọc từ `docker/model_manifest.json`. Train từ đầu với LR **1e-4**, r16/alpha32, 4-bit base, loss trên reply. Ngân hàng few-shot sẵn có chỉ được dùng làm ví dụ; không đưa teacher run vào pool.

| Stage | Sản phẩm dưới OUTPUT | GPU |
|---|---|---|
| `setup` | `models/Qwen3.5-9B/`, cache tải/cài; hai venv tạm nằm trên `/content` | Không |
| `preflight` | `preflight/`, `state/preflight.json`: 10 câu train, base không LoRA, phải có SQL thực thi thành công | Có |
| `generate` | `pool/attempt-*/` gồm trace/prompt/log; `pool/trace.jsonl` hợp nhất đủ 2368 câu của 24 shop | Có |
| `sft` | `sft/sql_rft.jsonl`: giữ mẫu đúng bằng nhãn train; tối đa 4000 dòng, tối đa 800 dòng dễ | Không |
| `notes` | `sft/sql_rft_notes.jsonl`: thêm ghi chú bằng base | Có |
| `train` | `lora/checkpoint-*` (10 bước/lần, giữ 2), `lora/adapter/`, log kiểm tra fla và train | Có |

Có thể chạy CLI bằng Python thông thường; `--help` và `--dry-run` không import torch/vLLM:

```bash
python scripts/colab_train.py --dry-run --gpu a100
python scripts/colab_train.py --output /content/drive/MyDrive/datamate/from_base --data /path/holding/train --stage all
# Hoặc chọn stage bằng DATAMATE_STAGE; đường dẫn bằng DATAMATE_OUTPUT / DATAMATE_DATA.
```

**Resume:** chạy lại notebook với cùng OUTPUT và tham số. Stage đã có sản phẩm hoàn chỉnh sẽ bỏ qua; venv trên đĩa tạm được cài lại nếu mất phiên. Bản thân `datamate.predict` **không resume**: nó reset submission, mở trace ở chế độ ghi mới. Wrapper tạo attempt mới, chỉ bỏ qua shop có đủ trace/prompt và log của mọi câu, chạy lại shop dang dở rồi hợp nhất không trùng ID. Không sửa/xóa attempt cũ; có thể mất công soạn view/nợ và sinh mẫu của shop chưa hoàn tất. SFT/notes được ghi file tạm rồi rename. Train dùng `--resume` lấy checkpoint đầy đủ gần nhất, có cả optimizer/scheduler; checkpoint thiếu file do ngắt khi ghi được giữ riêng trong `lora/incomplete/`; chưa tới checkpoint đầu thì phải train lại từ bước 0. Giữ nguyên tham số (script kiểm tra `flow.json`); thí nghiệm `MAX_STEPS=3` dùng OUTPUT riêng với lượt train đầy đủ.

Inference venv cài `.[llm]` (vLLM 0.32.0); training venv cài `training/requirements.txt` (Unsloth, torch 2.14.1, transformers 5.17.0, PEFT, TRL, bitsandbytes...) và `fla-core==0.6.0 --no-deps`. Hai stack không cài vào kernel Colab hoặc vào nhau; chạy subprocess tuần tự để giải phóng GPU giữa stage. Kiểm tra fla phải PASS trước khi train; nếu FAIL thì dừng, tránh tiêu ngân sách vào kernel tham chiếu chậm. A100 bỏ fp8, dùng context 32768 và KV auto; H100/G4 có compute capability >=8.9 giữ fp8 và context 49152. `MAX_LEN=24576` theo mô tả notebook cũ giữ đủ các dòng 9k–19k token; dữ liệu mới có thể dài hơn, xem số dòng bị loại trong log.

**Ngân sách khoảng 100 CU:** chưa đo GPU Colab nên không cam kết đủ. Theo 2368 câu × 8 mẫu × prompt khoảng 9k–19k token, phần input chính khoảng **170,5–359,9 triệu token** (trung vị 12,8k: **242,5 triệu**). Tối đa 4 mẫu thinking thêm **85,2–180,0 triệu**, chưa kể output, view/nợ, sửa SQL và notes; prefix cache có thể giảm compute. Đây là ngoại suy từ độ dài SFT notebook cũ, chưa phải token đo của lượt mới. Script in questions/min ngay sau preflight, cập nhật sau mỗi shop; train in steps/min từ những bước đầu. Ước tính giờ còn lại = câu còn lại / questions/min / 60 hoặc bước còn lại / steps/min / 60, nhân **CU/giờ đang hiển thị trong Colab** và chừa ngân sách cho setup/notes/checkpoint. Preflight có startup và chưa đại diện mọi shop; không ngoại suy chính xác từ nó. Nếu dự toán vượt số CU còn lại, ngắt phiên và giữ OUTPUT để tiếp tục sau; không âm thầm giảm K hoặc train thiếu dữ liệu.

**Gửi lại:** `colab.log`, `flow.json`, thư mục `state/`, `logs/` (đặc biệt preflight, generate, sft, check-fla, train), số CU đã dùng, tên GPU, tốc độ đo được, và `lora/adapter/` nếu hoàn tất. Nếu bị ngắt giữa train, giữ nguyên `lora/checkpoint-*` trên Drive để resume; không chỉ giữ mỗi adapter. Log/trace/SFT chứa dữ liệu thi: gửi riêng, không đưa vào repo public. Chưa kiểm chứng GPU, CUDA hoặc khả năng cài các version package trên Colab thực tế; preflight và check-fla là hai cổng dừng trước các lượt tốn tiền.
