#!/usr/bin/env bash
# DataMate Agent: base model by default; LORA_DIR selects your own adapter.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/flags.sh
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
OUT=${1:-runs/public}
NG=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
LAYERS=(); [ -z "${VIEWS_FROM:-}" ] || LAYERS=(--views-from "$VIEWS_FROM")
if ! [[ "$NG" =~ ^[1-9][0-9]*$ ]]; then echo "NUM_GPUS must be a positive integer" >&2; exit 1; fi
LORA=(); [ -z "${LORA_DIR:-}" ] || LORA=(--lora "$LORA_DIR" --lora-thinking)
mkdir -p "$OUT"
PIDS=()
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" -m datamate.predict --data data/public_test --output "$OUT/shard$i" --shard "$i/$NG" \
    --model "${MODEL_DIR:-models/Qwen3.5-9B}" "${LORA[@]}" --fewshot assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" --save-prompts "${LAYERS[@]}" > "$OUT/shard$i.log" 2>&1 &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" = 0 ] || { echo "Inference failed; inspect $OUT/shard*.log" >&2; exit 1; }
"$PY" -m datamate.submission merge "$OUT" --questions data/public_test/questions.jsonl
"$PY" -m datamate.submission check "$OUT/submission.jsonl" --questions data/public_test/questions.jsonl
