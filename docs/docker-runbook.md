# Chạy và xuất Docker DataMate

Máy Linux x86-64 có Docker, NVIDIA Container Toolkit, Python 3 (venv), GNU time, GPU hỗ trợ fp8 **≥24 GB mỗi GPU** và khoảng **80 GB đĩa trống**. Có mạng khi tải/build; suy luận tắt mạng. Máy thi là **2×RTX 5090 32 GB**, tối đa **6 giờ gồm tải model**. GPU 24 GB cần smoke test thực tế; hai GPU chạy hai bản model độc lập. Máy chuẩn bị hiện tại không có Docker, chỉ RTX 3050 4 GB nên chưa xác nhận image/GPU.

Chạy từ thư mục repo, theo thứ tự; thay đường dẫn repo và public test cho máy đích:

```bash
cd /path/to/datamate-agent-Locked-In1
set -euo pipefail
export TEAM=locked-in1
export DATA=/path/to/public_test
export NUM_GPUS=1                 # đặt 2 khi máy có đủ hai GPU
export TIMEOUT_SECONDS=21000
IMAGE="zaic2026_datamate_${TEAM}:final"
docker version
docker info
nvidia-smi
df -h .
docker system df
ls "$DATA/questions.jsonl" "$DATA/rulebook.md" "$DATA/calendar.csv" "$DATA/data"
```

Cần thấy Docker server hoạt động, GPU đúng VRAM và đủ dữ liệu. Nếu Docker dùng phân vùng khác, kiểm tra dung lượng phân vùng đó bằng `df -h "$(docker info --format '{{.DockerRootDir}}')"`.

**1. Tải đúng revision** (không tải adapter):

```bash
python3 -m venv .venv-download
.venv-download/bin/python -m pip install huggingface_hub
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 .venv-download/bin/hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir models/Qwen3.5-9B
du -sh models/Qwen3.5-9B
```

Phải tải xong tất cả safetensors, index, tokenizer/config. Giữ metadata `.cache/huggingface/download/` để preflight kiểm tra revision và SHA-256 weights. Lệnh theo [tài liệu CLI Hugging Face](https://huggingface.co/docs/huggingface_hub/guides/cli).

**2. Pack và build:**

```bash
# Nếu pack cũ tồn tại: mv build/training_code "build/training_code.prev.$(date +%s)"
bash docker/build.sh pack
python3 docker/verify_context.py
bash docker/build.sh build
docker run --rm --gpus all --network none --entrypoint nvidia-smi "$IMAGE"
docker run --rm --network none --entrypoint python3 "$IMAGE" -m pip check
```

Pack in số file và `own run evidence: False`; preflight phải in `context OK ... no LoRA artifacts`. Build thành công in `linux/amd64 <bytes>` và cây `/code`, có model/source/few-shot/training code. Lệnh `nvidia-smi` trong image phải thấy GPU; `pip check` phải không báo xung đột. Venv train không cài trong image nộp base-only để tiết kiệm đĩa; source train luôn có.

**3. Smoke 20 câu, rồi full public có đo wall-clock:**

```bash
export RESULT="$PWD/result_smoke_$(date +%Y%m%d_%H%M%S)"
bash docker/build.sh test 20 2>&1 | tee smoke-docker.log
export RESULT="$PWD/result_public_$(date +%Y%m%d_%H%M%S)"
/usr/bin/time -f 'elapsed_seconds=%e exit=%x' -o public-docker.time \
  bash docker/build.sh test 2>&1 | tee public-docker.log
cat public-docker.time
find "$RESULT/work" -name run.json -exec cat {} +
```

Mỗi `RESULT` phải mới/rỗng. Cần thấy mỗi shard `model ready`, `loaded ... shops`, `shard... exited 0`, cuối cùng `format OK` và `exit=0`. Smoke chỉ xử lý 20 câu nhưng submission vẫn đủ ID; full public cần đủ 1000 câu trong tổng `questions` của các `run.json`, tổng thời gian **<21600 giây**, không traceback/OOM/timeout. `TIMEOUT_SECONDS=21000` dành khoảng 10 phút cho finalize trước hạn; tiến trình trả 124 hoặc 143 phải điều tra log dù JSON còn hợp lệ. Kiểm tra số ID, số câu có đáp án và log chưa xử lý:

```bash
python3 - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
out = Path(os.environ['RESULT'])
qs = [json.loads(s)['id'] for s in (Path(os.environ['DATA'])/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
rows = [json.loads(s) for s in (out/'submission.jsonl').read_text().splitlines()]
assert Counter(r['id'] for r in rows) == Counter(qs)
runs = [json.loads(p.read_text()) for p in (out/'work').rglob('run.json')]
assert sum(r['questions'] for r in runs) == len(qs), 'Thiếu câu/shard hoàn tất'
unprocessed = sum('not processed:' in p.read_text() for p in (out/'logs').glob('q_*.json'))
print('IDs:', len(rows), '| types:', Counter(r['type'] for r in rows), '| not processed:', unprocessed)
PY
```

`not processed: 0` là dấu hiệu không còn log dự phòng; `abstain` vẫn có thể là quyết định của pipeline. PA chỉ lấy từ chấm có nhãn/portal, không suy ra từ `format OK`. Cấu hình đã chọn là base Qwen3.5-9B, **không LoRA, k=8, think-k=4, fp8**, bank `assets/fewshot_bank.jsonl`; không đổi cờ để đo lại cấu hình này.

**4. Xuất sau khi full test đạt:**

```bash
bash docker/build.sh export
md5sum -c "zaic2026_datamate_${TEAM}.tar.gz.md5"
ls -lh "zaic2026_datamate_${TEAM}.tar.gz"*
```

Phải có `.tar.gz`, `.tar.gz.md5`, `uncompressed .tar bytes: ...` và checksum `OK`. Export nén streaming, không giữ thêm bản `.tar` thô. Để chứng minh archive dùng offline, có thể chuyển archive sang máy Docker+GPU khác, chạy `docker load -i ...tar.gz` rồi lặp smoke với `--network none` qua helper.

Dung lượng/thời gian là **ước lượng chưa đo image**: BF16 weights khoảng **19–20 GB** (9.65B tham số ×2 byte); source pack <2 MB; image có thể khoảng **35–50+ GB**, archive khoảng **20–35+ GB**, phụ thuộc base/libraries. **80 GB khá sát** vì đồng thời giữ model, image, build cache và archive; kiểm tra `df`/`docker system df` sau build và trước export, dùng thêm ổ đĩa nếu thiếu, không tự xóa image/cache khác. Tải 20 GB ở 100 Mbps tối thiểu khoảng 27 phút; build khoảng 15–60+ phút tùy mạng/CPU/cache; pack vài giây; export khoảng 5–30+ phút tùy đĩa/CPU. Mốc public do người dùng cung cấp: **1000 câu ≈100 phút trên một RTX PRO 6000 96 GB** với cấu hình này; README tham chiếu ghi **≈3.5 giờ trên một GPU 24 GB**. Smoke 20 câu có thể 5–20+ phút vì vẫn tải model/dựng lớp shop. Đây không phải số đo RTX 5090; full test thực tế mới chứng minh hạn 6 giờ.
