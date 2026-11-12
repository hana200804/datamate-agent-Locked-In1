# DataMate Agent — mã train và bằng chứng

Gói được tạo bằng `scripts/pack_training_code.py`. `src/` chứa mã train và kiểm tra kernel; `data_generation/` chứa pipeline và bộ dựng SFT; `configs/` chứa cờ suy luận, split ID và model manifest.

`--run` thêm `flow.json`, `state/`, `logs/`, `sft/` và adapter của lượt train riêng vào `evidence/`. `--train-data` thêm split train được cấp vào `data/competition_train`. Không đóng gói public/private, pool trace hoặc model gốc vào đây. SHA-256 từng file nằm trong `data_manifest.json`.

Không có lượt train thì gói chỉ chứa mã nguồn; manifest ghi rõ không có bằng chứng train. Không được dùng nó để tuyên bố đã train hay đạt điểm.

Để train mới từ SFT của lượt riêng: `bash train.sh` trong image. Tùy chỉnh tham số theo `evidence/flow.json` nếu cần khớp lượt đã ghi; script mặc định max-len 24576, LR 1e-4, rank 16, alpha 32, một epoch. Model base có sẵn tại `/code/models/Qwen3.5-9B`.
