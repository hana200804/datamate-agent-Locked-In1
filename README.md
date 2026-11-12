# DataMate Agent — Locked-In1

Dự án cá nhân của **Nguyen Han ([@hana200804](https://github.com/hana200804))**: trả lời câu hỏi tiếng Việt trên dữ liệu kinh doanh CSV, Excel và SQLite bằng Qwen3.5-9B và DuckDB.

Model viết nhiều câu SQL, DuckDB thực thi trên dữ liệu đã chuẩn hóa, rồi hệ thống bỏ phiếu theo kết quả. Câu thiếu đồng thuận được bỏ qua. Dự án gồm suy luận, chấm điểm, phân tích lỗi, notebook Colab và quy trình tự sinh dữ liệu để train LoRA từ model gốc.

## Luồng xử lý

```text
CSV / XLSX / SQLite → nạp và chuẩn hóa → DuckDB
                                            ↓
Câu hỏi + rulebook → view và lớp công nợ → prompt + ví dụ train
                                            ↓
                             Qwen3.5-9B sinh nhiều mẫu SQL
                                            ↓
                             thực thi → sửa lỗi → bỏ phiếu
                                            ↓
                           đáp án có kiểu / bỏ qua → JSONL
```

[Sơ đồ chi tiết](docs/datamate_flow.excalidraw) · [Trạng thái kết quả](docs/results.md) · [Train trên Colab](docs/colab-training.md)

## Cài đặt

Python >= 3.12. Các bước nạp dữ liệu, chấm điểm và kiểm thử chạy trên CPU. Suy luận và train cần GPU phù hợp với bf16/fp8; đo tốc độ bằng preflight trước khi chạy toàn bộ.

```bash
git clone https://github.com/hana200804/datamate-agent-Locked-In1.git
cd datamate-agent-Locked-In1
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # CPU
# .venv/bin/pip install -e '.[llm,dev]'    # thêm vLLM để suy luận
.venv/bin/python -m pytest -q
```

Tải model gốc theo revision trong `docker/model_manifest.json`:

```bash
.venv/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir models/Qwen3.5-9B
```

Đặt dữ liệu được cấp vào thư mục riêng, không commit:

```text
data/
├── train/
│   ├── questions.jsonl
│   ├── answers.jsonl
│   ├── rulebook.md
│   ├── calendar.csv
│   └── data/<shop>/
└── public_test/           # cùng cấu trúc, không có answers.jsonl
```

## Chạy suy luận và đánh giá

Mặc định chạy model gốc, tự dựng view cho mỗi shop. Script dùng fp8 cho GPU hỗ trợ; với A100 nên dùng wrapper Colab bên dưới để tự chọn bf16 và context.

```bash
bash scripts/run_public.sh runs/public_base
bash scripts/run_holdout.sh runs/holdout_base

# Chọn adapter của lượt train riêng khi đã có:
LORA_DIR=runs/from_base/lora/adapter bash scripts/run_public.sh runs/public_lora

.venv/bin/python -m datamate.submission check runs/public_base/submission.jsonl \
  --questions data/public_test/questions.jsonl
.venv/bin/python -m datamate.predict --help
```

`NUM_GPUS`, `GPU_MEM`, `MODEL_DIR`, `LORA_DIR`, `PYTHON` và `ENGINE` cho phép điều chỉnh môi trường. `THINK_K` mặc định là `0`; đặt `THINK_K=4` để thử thêm mẫu thinking cho câu khó. Chỉ tái sử dụng view của lượt chạy riêng khi đặt `VIEWS_FROM=<thư mục lượt chạy>`.

Điểm PA = (đúng − sai) / tổng số câu; câu bỏ qua nhận 0. Chấm trên train có nhãn để phân tích lỗi; kết quả public phải lấy từ portal, không suy ra độ đúng chỉ từ việc SQL chạy thành công.

## Colab và train LoRA

- [Chạy public bằng model gốc](notebooks/colab_public_a100.ipynb): upload `repo.zip`, `train.zip`, `public_test.zip`; chọn GPU rồi chạy tất cả. Notebook trả runtime sau khi hoàn tất; lưu file nộp từ output trước khi mất ổ đĩa phiên.
- [Train từ model gốc](notebooks/colab_train_from_base.ipynb): giữ dữ liệu, log và checkpoint trên Drive; có preflight và resume theo stage/shop.
- [Hướng dẫn train và resume](docs/colab-training.md).

Tạo bản mã nguồn để upload Colab, không kèm dữ liệu, model hoặc kết quả chạy:

```bash
.venv/bin/python scripts/package_repo.py
```

CLI có thể xem trước mà không tải model hoặc cài torch:

```bash
.venv/bin/python scripts/colab_train.py --dry-run --gpu a100 --stage setup,public --public-think-k 0
.venv/bin/python scripts/colab_train.py --output runs/from_base --data data --stage all
```

Notebook `01_eda` đến `05_train_lora` chia quy trình thành các bước; `00_full_pipeline` ghép các bước đó. Notebook suy luận mặc định không LoRA, notebook train khởi tạo adapter mới. Mọi notebook được lưu không kèm output.

## Docker

[Hướng dẫn Docker](docker/README.md). Image mặc định dùng model gốc, tự dựng view và chạy offline. LoRA tự train có thể mount khi chạy. Chưa xác nhận lại GPU và thời gian của image sau lần dọn này.

## Cấu trúc

| Thư mục | Nội dung |
| --- | --- |
| `datamate/` | Nạp dữ liệu, view, công nợ, prompt, thực thi SQL, bỏ phiếu, chấm điểm |
| `training/` | Sinh SFT, few-shot, QLoRA, kiểm tra kernel, merge adapter |
| `scripts/` | Chạy public/holdout, Colab, đóng gói, vẽ sơ đồ |
| `notebooks/` | EDA, suy luận, đánh giá, train và Colab |
| `analysis/` | Phân tích lỗi và replay lượt chạy |
| `assets/` | Ngân hàng ví dụ train |
| `splits/` | Danh sách ID cho train/holdout/smoke test |
| `docker/` | Image và entry point suy luận offline |
| `tests/` | Kiểm thử CPU |
| `docs/` | Hướng dẫn, kết quả, sơ đồ và thông tin nguồn gốc |

`data/`, `models/`, `runs/`, adapter, bản nén trong `dist/` và cấu hình cá nhân được bỏ qua bởi Git.

## Giấy phép và nguồn gốc

Mã nguồn dùng [MIT](LICENSE). Dự án được phát triển và duy trì tại repo này; phần mã kế thừa và các tài nguyên bên ngoài được ghi tại [NOTICE.md](NOTICE.md). Model gốc và dữ liệu được cấp có giấy phép riêng.
