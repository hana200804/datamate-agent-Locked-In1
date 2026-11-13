"""CPU-only, offline preflight for the base-only submission's Docker COPY inputs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path = ROOT) -> None:
    manifest = json.loads((root / 'docker/model_manifest.json').read_text())
    if manifest['lora_adapter'] is not None:
        raise ValueError('submission requires lora_adapter=null')
    # Every explicit Dockerfile COPY source (directories checked separately below).
    for name in ('docker/requirements.txt', 'training/requirements.txt',
                 'assets/fewshot_bank.jsonl', 'docker/README.md', 'docker/predict.sh',
                 'scripts/flags.sh', 'docker/model_manifest.json', 'LICENSE', 'NOTICE.md',
                 'datamate/__init__.py', 'build/training_code/data_manifest.json'):
        if not (root / name).is_file() or not (root / name).stat().st_size:
            raise ValueError(f'missing {name}; download the pinned model and run docker/build.sh pack')
    model = root / 'models/Qwen3.5-9B'
    required = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    index = model / 'model.safetensors.index.json'
    if not index.is_file():
        raise ValueError(f'missing {index}; download the pinned model first')
    required.update(json.loads(index.read_text())['weight_map'].values())
    for name in sorted(required):
        path = model / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'missing/empty model file: {path}')
        # `hf download --revision ... --local-dir ...` records its commit here.
        metadata = model / '.cache/huggingface/download' / f'{name}.metadata'
        if not metadata.is_file():
            raise ValueError(f'missing download provenance: {metadata}; rerun the pinned hf download')
        lines = metadata.read_text().splitlines()
        if not lines or lines[0] != manifest['revision']:
            raise ValueError(f'model revision mismatch: {path}')
        # LFS etags are SHA-256; check complete weight contents, not just filenames.
        if name.endswith('.safetensors'):
            if len(lines) < 2 or len(lines[1]) != 64 or sha256(path) != lines[1]:
                raise ValueError(f'model weight checksum mismatch: {path}')
    package = root / 'build/training_code'
    packed = json.loads((package / 'data_manifest.json').read_text())
    if packed['model'] != manifest:
        raise ValueError('packed model manifest is stale; repack training code')
    if packed.get('own_run_evidence_included') or packed.get('train_data_included'):
        raise ValueError('submission requires the default source-only training package; repack without options')
    for item in packed['files']:
        path = package / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'training package checksum mismatch: {path}')
    for src, dst in (('scripts/flags.sh', 'configs/flags.sh'),
                     ('assets/fewshot_bank.jsonl', 'data/fewshot_bank.jsonl')):
        if sha256(root / src) != sha256(package / dst):
            raise ValueError(f'packed {src} is stale; repack training code')
    for directory in (model, package):
        for path in directory.rglob('*'):
            if path.name == 'adapter_config.json' or path.name.startswith('adapter_model.') or 'sql_rft_v3' in path.parts:
                raise ValueError(f'LoRA artifact must not ship: {path}')
    print(f'context OK: pinned {manifest["revision"]}, {len(required)} model files, '
          f'{len(packed["files"])} verified package files; no LoRA artifacts')


if __name__ == '__main__':
    try:
        verify()
    except (OSError, ValueError, KeyError) as exc:
        print(f'context ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
