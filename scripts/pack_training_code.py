"""Package training source and, optionally, evidence from your own Colab run.

    python scripts/pack_training_code.py
    python scripts/pack_training_code.py --run runs/from_base --train-data data/train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    else:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'build/training_code')
    p.add_argument('--run', type=Path, help='your own colab_train.py output directory')
    p.add_argument('--train-data', type=Path, help='organizer train split; never public/private')
    a = p.parse_args()
    out = a.out.resolve()
    # Never delete an arbitrary output path. Rebuilds use a new directory or remove the old package explicitly.
    if out.exists():
        p.error(f'{out} already exists; choose a new --out or move the previous package first')
    if a.run and not (a.run / 'flow.json').is_file():
        p.error('--run must contain flow.json from your own run')
    if a.train_data and not all((a.train_data / n).exists() for n in ('questions.jsonl', 'answers.jsonl', 'data')):
        p.error('--train-data must be a labeled train split')
    sources = [ROOT / 'training', ROOT / 'datamate', ROOT / 'docker', ROOT / 'assets', ROOT / 'scripts', ROOT / 'splits']
    if a.run:
        sources.append(a.run.resolve())
    if a.train_data:
        sources.append(a.train_data.resolve())
    if any(out.is_relative_to(src) for src in sources):
        p.error('--out must be outside the source directories being packaged')
    out.mkdir(parents=True)
    copy(ROOT / 'docker/training_code_README.md', out / 'README.md')
    copy(ROOT / 'training/requirements.txt', out / 'requirements.txt')
    copy(ROOT / 'docker/train.sh', out / 'train.sh')
    for name in ('LICENSE', 'NOTICE.md'):
        copy(ROOT / name, out / name)
    for name in ('train_lora.py', 'check_fla.py', 'merge_lora.py', 'no_fla'):
        copy(ROOT / 'training' / name, out / 'src' / name)
    for name in ('build_sft.py', 'add_notes.py', 'build_fewshot_bank.py', 'build_pool.py'):
        copy(ROOT / 'training' / name, out / 'data_generation' / name)
    copy(ROOT / 'datamate', out / 'data_generation/datamate')
    copy(ROOT / 'scripts/colab_train.py', out / 'data_generation/colab_train.py')
    copy(ROOT / 'scripts/flags.sh', out / 'configs/flags.sh')
    copy(ROOT / 'docker/model_manifest.json', out / 'configs/model_manifest.json')
    copy(ROOT / 'splits', out / 'configs/splits')
    copy(ROOT / 'assets/fewshot_bank.jsonl', out / 'data/fewshot_bank.jsonl')
    if a.run:
        for name in ('flow.json', 'state', 'logs', 'sft', 'lora/adapter'):
            src = a.run / name
            if src.exists():
                copy(src, out / 'evidence' / name)
    if a.train_data:
        copy(a.train_data, out / 'data/competition_train')
    files = [{'path': str(f.relative_to(out)), 'bytes': f.stat().st_size, 'sha256': sha256(f)}
             for f in sorted(out.rglob('*')) if f.is_file()]
    manifest = {'project': 'hana200804/datamate-agent-Locked-In1',
                'own_run_evidence_included': bool(a.run), 'train_data_included': bool(a.train_data),
                'model': json.loads((ROOT / 'docker/model_manifest.json').read_text()), 'files': files}
    (out / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'{out}: {len(files)} files; own run evidence: {bool(a.run)}')


if __name__ == '__main__':
    main()
