#!/usr/bin/env bash
# Train a fresh adapter from the SFT of your own packaged run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/opt/venv-train/bin/python}
BASE=${BASE_MODEL:-/code/models/Qwen3.5-9B}
OUT=${OUT_DIR:-/tmp/lora}
SFT=${SFT_FILE:-evidence/sft/sql_rft_notes.jsonl}
[ -s "$SFT" ] || { echo "Missing own SFT: package with --run or set SFT_FILE" >&2; exit 1; }
"$PY" src/check_fla.py --T 4096
"$PY" src/train_lora.py "$SFT" --out "$OUT" --model "$BASE" --r 16 --alpha 32 \
  --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking --completion-logits \
  --max-len "${MAX_LEN:-24576}" --lr "${LR:-1e-4}" "$@"
