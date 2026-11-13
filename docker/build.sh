#!/usr/bin/env bash
# Build, check and export the submission image, step by step as the organizers' guide asks (sections 9, 10 and 11).
# Needs Docker with BuildKit and the NVIDIA Container Toolkit (sudo or the docker group). Run from anywhere:
#   TEAM=<team> bash docker/build.sh pack      # scripts/pack_training_code.py -> build/training_code
#   TEAM=<team> bash docker/build.sh build     # image zaic2026_datamate_<team>:final, then platform, size and /code layout
#   TEAM=<team> bash docker/build.sh test [N]  # scoring command (1/2 GPUs, no network, read-only /data, 16 GiB shm) on the
#                                              # public test -> result_docker/ (N = first N questions, a smoke test only)
#   TEAM=<team> bash docker/build.sh export    # .tar.gz, uncompressed .tar size in bytes, and the .md5 file to send
set -euo pipefail
cd "$(dirname "$0")/.."
TEAM=${TEAM:-locked-in1}
IMAGE=zaic2026_datamate_${TEAM,,}:final
ARCHIVE=zaic2026_datamate_${TEAM,,}.tar.gz
DOCKER=${DOCKER:-docker}
case "${1:-}" in
  pack)
    python3 scripts/pack_training_code.py
    ;;
  build)
    python3 docker/verify_context.py
    DOCKER_BUILDKIT=1 $DOCKER build --platform linux/amd64 -f docker/Dockerfile -t "$IMAGE" .
    $DOCKER image inspect -f '{{.Os}}/{{.Architecture}} {{.Size}}' "$IMAGE"
    $DOCKER run --rm --network none --entrypoint "" "$IMAGE" find /code -maxdepth 2
    ;;
  test)
    data=${DATA:-$PWD/data/public_test}
    result=${RESULT:-$PWD/result_docker}
    [ -s "$data/questions.jsonl" ] || { echo "missing $data/questions.jsonl" >&2; exit 1; }
    if [ -n "${2:-}" ] && ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then echo 'N must be positive' >&2; exit 2; fi
    ng=${NUM_GPUS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)}
    if [ -z "${NUM_GPUS:-}" ] && [[ "$ng" =~ ^[0-9]+$ ]] && (( ng > 2 )); then ng=2; fi
    [[ "$ng" =~ ^[12]$ ]] || { echo 'NUM_GPUS must be 1 or 2' >&2; exit 2; }
    gpu_request='"device=0"'
    [ "$ng" = 1 ] || gpu_request='"device=0,1"'
    mkdir -p "$result"
    [ -z "$(ls -A "$result")" ] || { echo "RESULT must be empty: $result (choose a new RESULT)" >&2; exit 1; }
    args=(); [ -z "${2:-}" ] || args=(--limit "$2")
    status=0
    $DOCKER run --rm --gpus "$gpu_request" --network none --shm-size=16g --entrypoint bash \
      -e "NUM_GPUS=$ng" -e "TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-21000}" \
      -v "$(realpath "$data")":/data:ro -v "$(realpath "$result")":/result "$IMAGE" /code/predict.sh "${args[@]}" || status=$?
    # Check inside the image too: no host .venv is required. Preserve worker errors.
    $DOCKER run --rm --network none --entrypoint python3 \
      -v "$(realpath "$data")":/data:ro -v "$(realpath "$result")":/result:ro "$IMAGE" \
      -m datamate.submission check /result/submission.jsonl --questions /data/questions.jsonl || status=1
    exit "$status"
    ;;
  export)
    set -o pipefail
    zip=$(command -v pigz >/dev/null && echo "pigz -p 8" || echo gzip)
    $DOCKER save "$IMAGE" | $zip > "$ARCHIVE"
    echo "uncompressed .tar bytes: $($zip -dc "$ARCHIVE" | wc -c)"
    md5sum "$ARCHIVE" > "$ARCHIVE.md5"
    cat "$ARCHIVE.md5"
    ;;
  *) echo "usage: TEAM=<team> bash docker/build.sh pack|build|test [N]|export" >&2; exit 2 ;;
esac
