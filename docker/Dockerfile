# DataMate Agent offline inference image. See docker/README.md.
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
# A plain `docker run` starts predict.sh, not the vLLM server of the base image
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 VLLM_NO_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONPATH=/code/src
WORKDIR /code
COPY docker/requirements.txt /code/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /code/requirements.txt
# Training pins its own torch/transformers: a separate offline-ready venv, so the vLLM environment stays untouched
COPY training/requirements.txt /code/training_code/requirements.txt
RUN python3 -m venv /opt/venv-train \
    && /opt/venv-train/bin/pip install --no-cache-dir -r /code/training_code/requirements.txt \
    && /opt/venv-train/bin/pip install --no-cache-dir fla-core==0.6.0 --no-deps
# Weights first, so a code change does not invalidate their layers
COPY models/Qwen3.5-9B /code/models/Qwen3.5-9B
COPY build/training_code/ /code/training_code/
COPY assets/fewshot_bank.jsonl /code/assets/fewshot_bank.jsonl
COPY docker/README.md /code/README.md
COPY docker/predict.sh scripts/flags.sh docker/model_manifest.json /code/
COPY datamate /code/src/datamate
COPY LICENSE NOTICE.md /code/
RUN chmod +x /code/predict.sh
CMD ["bash", "/code/predict.sh"]
