# DataMate Agent — Docker offline

Chủ dự án: Nguyen Han (@hana200804). Mã và nguồn gốc: `/code/LICENSE`, `/code/NOTICE.md`.

Image mặc định chạy Qwen3.5-9B gốc, không LoRA, không view public lưu sẵn. Model được ghim revision trong `model_manifest.json`; DuckDB chạy SQL do model sinh và đầu ra được bỏ phiếu. Chưa kiểm chứng lại image trên GPU sau lần dọn repo này.

## Chuẩn bị và build

Tải model vào `models/Qwen3.5-9B` theo README gốc của repo, rồi:

```bash
bash docker/build.sh pack
bash docker/build.sh build
bash docker/build.sh test 10
bash docker/build.sh export
```

`TEAM` mặc định `locked-in1`; image là `zaic2026_datamate_locked-in1:final`. `pack` tạo bản mã train và manifest SHA-256; không giả định có adapter hay dữ liệu train v1/v2/v3. Nếu có lượt train riêng, thêm bằng `python scripts/pack_training_code.py --run runs/from_base --train-data data/train` trước khi build. Gói có dữ liệu chỉ phục vụ build nội bộ, không commit hoặc upload vào repo public.

```bash
docker run --rm --gpus '"device=0,1"' --network none --shm-size=16g \
  -v /path/to/public_test:/data:ro -v "$PWD/result_docker:/result" \
  zaic2026_datamate_locked-in1:final
```

Dữ liệu đầu vào `/data` có `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<shop>/`. Đầu ra: `/result/submission.jsonl`, `/result/logs/<id>.json` và `/result/work/`.

`NUM_GPUS`, `TIMEOUT_SECONDS` (mặc định 21000), `GPU_MEM`, `QUANT`, `ENGINE`, `THINK_K` điều chỉnh cách chạy. Timeout là giới hạn tiến trình, chưa phải bằng chứng hoàn tất toàn bộ tập trong thời hạn thi.

Dùng adapter tự train: mount thư mục adapter vào `/code/models/own_lora:ro` và thêm `-e LORA_DIR=/code/models/own_lora`. Mặc định fp8 cần GPU tương thích; điều chỉnh quant/context/engine cho phần cứng thực tế.

## Train lại

Image có môi trường train riêng ở `/opt/venv-train`. Nếu đã pack lượt train có `sft/sql_rft_notes.jsonl`, chạy `bash /code/training_code/train.sh`. Script khởi tạo adapter mới từ model gốc; dùng tham số đúng với lượt train đã ghi trong `evidence/flow.json` khi cần tái lập cấu hình riêng.
