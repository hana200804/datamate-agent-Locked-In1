# DataMate Agent — Docker offline

Chủ dự án: Nguyen Han (@hana200804). Mã và nguồn gốc: `/code/LICENSE`, `/code/NOTICE.md`.

Cấu hình nộp: **Qwen3.5-9B gốc**, revision trong `model_manifest.json`, **không LoRA**, `--k 8 --think-k 4`, fp8, few-shot `assets/fewshot_bank.jsonl`; các cờ còn lại ở `scripts/flags.sh`. Không copy `adapters/` hoặc view public lưu sẵn. `LORA_DIR` và cờ LoRA bị entrypoint từ chối. Chưa build/chạy GPU được trên máy chuẩn bị hiện tại.

Lệnh tải model, pack, build, smoke 20 câu, full public có đo thời gian và export: xem [runbook tiếng Việt trong repo](../docs/docker-runbook.md), cũng được pack vào `/code/training_code/docker-runbook.md` trong image. Không commit/push dữ liệu, weights hoặc artifacts.

```bash
bash docker/build.sh pack
bash docker/build.sh build
NUM_GPUS=1 RESULT="$PWD/result_smoke" bash docker/build.sh test 20
NUM_GPUS=2 RESULT="$PWD/result_public" bash docker/build.sh test
bash docker/build.sh export
```

`TEAM` mặc định `locked-in1`; image `zaic2026_datamate_locked-in1:final`. `test` tự chọn tối đa 2 GPU nếu không đặt `NUM_GPUS`; mỗi `RESULT` phải mới/rỗng. Không cần `.venv` trên host để check kết quả: helper chạy checker CPU trong image. `test 20` chỉ suy luận 20 câu đầu, vẫn ghi đủ ID của toàn tập với `abstain` cho phần còn lại.

`pack` dùng Python stdlib, không GPU/mạng; tạo `build/training_code/` và manifest SHA-256. Nếu thư mục đã có, chuyển gói cũ sang tên khác trước khi pack lại. Gói mặc định chỉ chứa mã train và few-shot, không adapter/dữ liệu train. Build preflight kiểm tra mọi COPY input, revision/checksum weights qua metadata của `hf download`, checksum gói, cờ/few-shot mới nhất và từ chối adapter hoặc gói `--run`/`--train-data` trong cấu hình nộp. Giữ `.cache/huggingface/download/*.metadata` cạnh model đến khi preflight xong.

```bash
mkdir -p result_manual
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
  -v /path/to/public_test:/data:ro -v "$PWD/result_manual:/result" \
  zaic2026_datamate_locked-in1:final
```

`/data` chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<shop>/`. Output: `/result/submission.jsonl`, `/result/logs/<id>.json`, `/result/work/run.*/shard*/` và `shard*.log`. Mỗi lần chạy có shard root mới; log cũ chuyển vào `work/run.*/previous-logs/` để tránh lẫn kết quả. Pipeline lưu và merge sau mỗi shop.

Entrypoint tạo submission đủ ID, mặc định `abstain`, **trước** khi tải model; khi worker lỗi, timeout hoặc nhận TERM/INT thì dừng worker, merge và bổ sung log. Lỗi worker/checker trả mã khác 0; timeout thường là 124, TERM là 143. SIGKILL/mất điện không chạy được trap nhưng submission đã ghi nguyên tử trước đó vẫn còn. Điều này cần input hợp lệ và `/result` ghi được; `format OK` chứng minh định dạng, chưa chứng minh đã giải hết hoặc đạt PA.

`TIMEOUT_SECONDS` mặc định và tối đa **21000** (5 giờ 50 phút, tính cả model load), thêm tối đa 15 giây để kill worker; thời gian khởi động/finalize dùng phần dự phòng trước mốc 6 giờ. `NUM_GPUS` là 1 hoặc 2; mỗi GPU tải một bản model, chia shop theo data parallel, không cộng VRAM. Mặc định `QUANT=fp8`, `THINK_K=4`, `GPU_MEM=0.92`, `ENGINE` như `flags.sh`; giữ các giá trị này khi tái lập cấu hình đã chọn. GPU 24 GB vẫn phải kiểm chứng với context 49152 và fp8; không suy ra khả năng chạy từ dung lượng VRAM.

## Mã train

`/code/training_code/` luôn có source và manifest. Bản nộp base-only bỏ cài venv train để tránh nhân đôi torch/CUDA và tăng dung lượng; suy luận vẫn đầy đủ. Nếu cần một image **nghiên cứu train riêng**, build bằng `docker build --build-arg INSTALL_TRAINING_ENV=1 -f docker/Dockerfile -t datamate-train .` để có `/opt/venv-train`. SFT của lượt train riêng phải được cung cấp riêng; cấu hình nộp mặc định không kèm SFT/adapter. `docker/train.sh` chỉ dùng được khi môi trường train và SFT đã có.
