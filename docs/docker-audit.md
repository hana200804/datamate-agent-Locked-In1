# Docker audit — 2026-11-12

Kết quả: đã chuẩn bị source/config, pack và kiểm tra CPU trong phạm vi máy hiện tại; **chưa có image build hay archive**, chưa có bằng chứng GPU/PA/thời gian. Không commit hoặc push. Hướng dẫn máy đích: [docker-runbook.md](docker-runbook.md). Diff đầy đủ của các thay đổi thực thi và runbook: [docker-audit.patch](docker-audit.patch).

Các file được sửa/tạo chỉ ở `docker/`, `scripts/flags.sh`, `scripts/pack_training_code.py`, `docs/`; không sửa pipeline `datamate/`. Những thay đổi có sẵn ở `scripts/colab_train.py`, notebook và `datamate/holdout_report.py` được giữ nguyên.

## Vấn đề và cách xử lý

Vị trí bảng này dùng số dòng của bản **trước sửa**; diff kèm theo có cả ngữ cảnh cũ/mới.

| Mức | File:dòng | Phát hiện và xử lý |
|---|---|---|
| High | `scripts/flags.sh:5` | Mặc định `think-k=0`, sai cấu hình đã chọn. Sửa thành 4, giữ toàn bộ giá trị cờ suy luận còn lại. |
| High | `docker/predict.sh:26`, `:29` | `wait` không đối số che mã lỗi worker; checker bị `|| true`. Nay chờ/ghi mã từng shard, trả lỗi worker/checker. |
| High | `docker/predict.sh:19`, `:27` | Submission dự phòng phụ thuộc import/startup của worker, không có trap TERM/INT để dừng worker và finalize. Nay khởi tạo đủ ID trước GPU/model, trap dừng/chờ worker, merge, bổ sung logs và check. |
| Medium | `docker/predict.sh:13`, `:20` | Không kiểm tra số GPU, không tôn trọng CUDA device mask; lấy chỉ số GPU trực tiếp. Nay xác thực 1/2 GPU, chọn GPU UUID hoặc device mask có sẵn; lỗi startup vẫn để submission hợp lệ. |
| Medium | `docker/build.sh:30`, `:32` | Test bắt buộc GPU 0,1 và phụ thuộc host `.venv`. Nay hỗ trợ 1/2 GPU, checker chạy CPU trong image, giữ mã lỗi inference. |
| Medium | `docker/build.sh:29`, `docker/predict.sh:21` | Tái dùng cùng result/shard có thể merge đáp án của lượt cũ, nhất là đổi từ 2 GPU sang 1 GPU. Helper yêu cầu RESULT mới/rỗng; entrypoint dùng shard root mới và lưu riêng log cũ. |
| Medium | `docker/build.sh:20` | Chỉ kiểm tra config và manifest tồn tại, không bắt weights thiếu/sai revision/truncated, gói cũ hoặc adapter lẫn vào package. Thêm `docker/verify_context.py`: đủ COPY input, metadata revision, SHA-256 weights/package, so flags/bank hiện tại, chặn adapter và gói evidence/train-data trong cấu hình nộp. |
| Medium | `docker/Dockerfile:12` | Luôn cài thêm torch/CUDA/train stack dù bản nộp chỉ chạy base inference, tăng đáng kể dung lượng khi đĩa khoảng 80 GB. Source train vẫn luôn kèm; môi trường train chuyển sang tùy chọn `INSTALL_TRAINING_ENV=1`, mặc định 0. |
| Medium | `docker/predict.sh:15`, `docker/README.md:30` | LoRA có thể được bật từ env/mount, lệch cấu hình nộp base-only. Nay submission entrypoint từ chối LORA_DIR/cờ LoRA; package/model cũng bị preflight kiểm tra adapter artifacts. Mặc định ban đầu đã không COPY thư mục adapters. |
| Low | `docker/model_manifest.json:17` | Mô tả THINK_K mặc định 0; cập nhật thành 4 và ghi rõ base/no LoRA/few-shot. `gpu_validated` vẫn false. |

`docker/requirements.txt` không đổi: cả 8 dependency đều ghim phiên bản, vLLM 0.32.0 khớp base/manifest; chưa kiểm tra cài đặt/resolution trong base image. Dockerfile ghim base digest, reset server ENTRYPOINT và dùng `/code/predict.sh`; môi trường offline đã đặt. `.dockerignore` dùng allowlist, không cho `adapters/` vào context. Kiểm tra source pack thực tế: không có adapter config/weights hoặc sql_rft_v3; model chưa tải nên chưa thể kiểm tra nội dung weights thật.

## Bằng chứng kiểm tra

```text
$ bash -n docker/predict.sh docker/build.sh docker/train.sh scripts/flags.sh scripts/run_public.sh scripts/run_holdout.sh
exit 0, no output

$ .venv/bin/python -m pytest -q
65 passed, 4 warnings in 0.72s

$ .venv/bin/python -m pytest -q docker/test_entrypoint.py
18 passed in 6.49s

$ python3 scripts/pack_training_code.py
/home/hau/orca/datamate-agent-Locked-In1/build/training_code: 40 files; own run evidence: False
exit 0

$ <stdlib recheck of every packed manifest path, byte count and SHA-256>
package manifest/checksums OK: 40 files; 750511 bytes; no adapters

$ python3 docker/verify_context.py
context ERROR: missing /home/hau/orca/datamate-agent-Locked-In1/models/Qwen3.5-9B/model.safetensors.index.json; download the pinned model first
exit 1 (expected, model download not done)
```

Gói cuối có 40 file nguồn/tài liệu, tổng 750511 byte (không tính chính data_manifest); thêm runbook vào gói để có hướng dẫn bên trong image. Pack chỉ dùng stdlib/file I/O, không import GPU libraries hoặc gọi mạng. Gói 39 file của lần kiểm tra đầu được chuyển sang thư mục tạm trước khi pack lại; không xóa đè output bất kỳ.

18 kiểm tra bổ sung dùng **mô phỏng model và GPU discovery**, chạy nguyên shell entrypoint trừ việc đổi `/code` sang temp dir. Chúng kiểm tra: chia shard/device 1/2 GPU; CUDA mask; chạy lại không lẫn kết quả; worker exit 7; GPU 0/3/invalid/không có GPU; timeout vượt giới hạn; LORA_DIR bị chặn; GNU timeout exit 124 giữ 2 đáp án đã ghi và 2 abstain; TERM exit 143 giữ 1 đáp án và 3 abstain. Mỗi trường hợp kiểm tra đủ 4 ID và đủ logs. Fixture model nhỏ kiểm tra preflight pass và chặn sai revision/thiếu hoặc truncated weights/adapter/stale flags; mock Docker kiểm tra chọn GPU và truyền exit lỗi cho helper. **Đây không phải GPU test.**

Kiểm tra `--help` đã chạy với toàn bộ `DATAMATE_FLAGS` và JSON engine, exit 0. Vì argparse `--help` có thể thoát trước khi kiểm tra hết giá trị, còn gọi trực tiếp `datamate.predict.parse_args` với data/output/model giả, không chạy inference; các assert đạt:

```text
k=8 think_k=4 quant=fp8 lora=None lora_thinking=False
context=49152 think_budget=16384 max_tokens=2048 gpu_mem=0.92
notes=True debt_views=True debt_route=True canon_names=True answer_unanswerable=True
engine={kv_cache_dtype: fp8, max_num_seqs: 128, max_num_batched_tokens: 8192, attention_backend: TRITON_ATTN}
```

Cả 12 COPY source đã đối chiếu với filesystem và allowlist; 11 tồn tại sau pack, chỉ `models/Qwen3.5-9B` chưa có, được tạo bởi bước `hf download --revision ... --local-dir ...` trong runbook. `build/training_code/` được tạo bởi pack; các source còn lại có sẵn. Quy ước metadata revision/ETag của preflight đã đối chiếu [mã chính thức Hugging Face](https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/_local_folder.py); safetensors kiểm tra SHA-256 theo ETag LFS.

## Giới hạn và việc còn lại

- `command -v docker` không có output: Docker chưa cài. `nvidia-smi` xác nhận RTX 3050 4096 MiB, driver 580.178.04/CUDA 13.0. `df -h .` còn khoảng 54 GiB. Không tải model, cài Docker, build, chạy model hoặc export trên máy này.
- Cần máy Docker + NVIDIA Container Toolkit + GPU hỗ trợ fp8 ≥24 GB để tải pinned model, preflight thật, build Linux amd64, `pip check`, smoke 20, full public 1000 có timing, export và verify MD5 theo runbook. Sau đó nên kiểm tra trên đúng 2×RTX 5090 32 GB để chứng minh thời gian ≤6h và độ tái lập PA; kết quả CPU không chứng minh các điều đó.
- Kích thước image/archive và mốc 15–60+ phút build/5–30+ phút export chỉ là ước lượng; khoảng 80 GB có thể thiếu khi giữ đồng thời model, layers/cache và archive. Runbook chỉ cách đo dung lượng thực tế, không tự xóa cache/image của người khác.
- Host CPU tests dùng pandas 3.1.1 và python-calamine 0.9.1; image ghim 3.0.3 và 0.9.0. Vì chưa build, chưa xác nhận dependency resolution, ABI/kernel, fp8/context memory hoặc tương thích image trên RTX 5090.
- Phát hiện ngoài phạm vi, chưa sửa: `README.md:73` vẫn mô tả THINK_K mặc định 0, trong khi script nay là 4; coordinator có thể đồng bộ README. Bốn warning có sẵn tại `datamate/ingest.py:75` và `:101` báo hành vi fillna khác dtype sẽ lỗi trong pandas tương lai; test hiện vẫn đạt, pipeline không bị sửa.

Submission fallback cần input JSONL hợp lệ và `/result` ghi được. SIGKILL/mất điện không chạy trap; bản submission đã ghi nguyên tử trước đó vẫn giữ được. Việc luôn có file đúng định dạng không đồng nghĩa giải đủ câu, hoàn tất trong hạn hoặc đạt PA ~0.70.
