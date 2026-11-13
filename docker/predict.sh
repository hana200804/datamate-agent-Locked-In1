#!/usr/bin/env bash
# Offline entrypoint: one process per GPU, shops split over 1 or 2 GPUs.
# Each process saves/merges after every shop. Initialize all-abstain output
# before GPU/model startup, and finalize it on timeout, worker failure or TERM.
set -euo pipefail
cd /code
source ./flags.sh
DATA="${DATA_DIR:-/data}" OUT="${RESULT_DIR:-/result}"
T="${TIMEOUT_SECONDS:-21000}"
start=$(date +%s)
mkdir -p "$OUT/work"
# A new shard root prevents answers from an earlier run entering this submission.
WORK=$(mktemp -d "$OUT/work/run.XXXXXXXX")
if [ -d "$OUT/logs" ]; then mv "$OUT/logs" "$WORK/previous-logs"; fi
mkdir -p "$OUT/logs"
pids=()
finalize() {
  local status=$? pid
  trap - EXIT TERM INT
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  python3 -m datamate.submission merge "$OUT" --shards "$WORK" --no-copy-logs --questions "$DATA/questions.jsonl" || status=1
  python3 -m datamate.submission fill-logs "$OUT/logs" --questions "$DATA/questions.jsonl" || status=1
  python3 -m datamate.submission check "$OUT/submission.jsonl" --questions "$DATA/questions.jsonl" || status=1
  grep -h -E 'model ready|loaded [0-9]+ shops|view drafts|debt drafts' "$WORK"/shard*.log || true
  echo "datamate: done in $(( $(date +%s) - start ))s, status $status; work $WORK"
  exit "$status"
}
# This also checks that the input question IDs are readable before any GPU work.
python3 -m datamate.submission merge "$OUT" --shards "$WORK" --no-copy-logs --questions "$DATA/questions.jsonl"
trap finalize EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
[[ "$T" =~ ^[1-9][0-9]*$ ]] && (( T <= 21000 )) || { echo 'TIMEOUT_SECONDS must be 1..21000' >&2; exit 2; }
# The submission runs only the base model, even when the host has a LoRA env var.
[ -z "${LORA_DIR:-}" ] || { echo 'Submission image does not accept LORA_DIR' >&2; exit 2; }
for arg in "$@"; do
  case "$arg" in --lora|--lora=*|--lora-thinking) echo 'Submission image does not accept LoRA flags' >&2; exit 2;; esac
done
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
else
  gpu_list=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || { echo 'Cannot enumerate GPUs' >&2; exit 1; }
  devices=()
  while IFS= read -r device; do [ -z "$device" ] || devices+=("$device"); done <<< "$gpu_list"
fi
available=${#devices[@]}
NG="${NUM_GPUS:-$(( available > 2 ? 2 : available ))}"
[[ "$NG" =~ ^[12]$ ]] && (( NG <= available )) || { echo "Need 1 or 2 visible GPUs; requested $NG, visible $available" >&2; exit 2; }
deadline=$(( T > 900 ? T - 900 : 1 ))
echo "datamate: $NG GPU(s), timeout ${T}s (includes model load), work $WORK"
for ((i=0; i<NG; i++)); do
  CUDA_VISIBLE_DEVICES="${devices[i]}" timeout --signal=TERM --kill-after=15s "${T}s" \
    python3 -m datamate.predict --data "$DATA" --output "$WORK/shard$i" --shard "$i/$NG" --deadline "$deadline" \
    --log-dir "$OUT/logs" --merge-into "$OUT" \
    --model /code/models/Qwen3.5-9B --fewshot /code/assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" "$@" > "$WORK/shard$i.log" 2>&1 &
  pids+=("$!")
done
status=0
for i in "${!pids[@]}"; do
  worker_status=0
  wait "${pids[i]}" || worker_status=$?
  echo "datamate: shard$i exited $worker_status"
  if (( worker_status != 0 )); then status=$worker_status; fi
done
pids=()
exit "$status"
