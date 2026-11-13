"""CPU checks for the real entrypoint shell, substituting only GPU discovery/model work.

Run: .venv/bin/python -m pytest -q docker/test_entrypoint.py
"""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

from docker.verify_context import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(tmp_path):
    code, data, out, bin_dir = (tmp_path / name for name in ('code', 'data', 'result', 'bin'))
    for path in (code, data, out, bin_dir):
        path.mkdir()
    (code / 'flags.sh').write_text((ROOT / 'scripts/flags.sh').read_text())
    # Only relocate /code; all process management/finalization runs unchanged.
    entrypoint = code / 'predict.sh'
    entrypoint.write_text((ROOT / 'docker/predict.sh').read_text().replace('cd /code\n', f'cd "{code}"\n', 1))
    qs = [{'id': f'q_{i}', 'business_id': f'shop_{i % 2}'} for i in range(4)]
    (data / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in qs))
    launcher = bin_dir / 'python3'
    launcher.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys, time
from pathlib import Path
if sys.argv[1:3] != ['-m', 'datamate.predict']:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
from datamate.predict import parse_args
from datamate.pipeline import load_questions
from datamate.submission import write_submission, merge_shards
a = parse_args(sys.argv[3:])
assert a.k == 8 and a.think_k == 4 and a.quant == 'fp8' and a.lora is None
assert a.debt_views and a.debt_route and a.canon_names and a.answer_unanswerable
mode = os.environ['FAKE_MODE']
if mode == 'failure':
    sys.exit(7)
qs, _ = load_questions(a)
a.output.mkdir(parents=True, exist_ok=True)
results = {q['id']: {'type': 'number', 'answer': 42} for q in (qs if mode == 'success' else qs[:1])}
write_submission(a.output / 'submission.jsonl', qs, results)
merge_shards(a.merge_into, a.data / 'questions.jsonl', shards=a.output.parent, copy_logs=False, quiet=True)
(a.output / 'mock.json').write_text(json.dumps({'shard': a.shard, 'device': os.environ['CUDA_VISIBLE_DEVICES']}))
if mode == 'success':
    sys.exit(0)
time.sleep(60)
''')
    launcher.chmod(0o755)
    smi = bin_dir / 'nvidia-smi'
    smi.write_text('#!/bin/bash\nprintf "%s" "${FAKE_GPUS}"\n')
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in (
        'NUM_GPUS', 'CUDA_VISIBLE_DEVICES', 'LORA_DIR', 'THINK_K', 'QUANT', 'ENGINE', 'GPU_MEM')}
    env.update(PATH=f'{bin_dir}:{env["PATH"]}', PYTHONPATH=str(ROOT), DATA_DIR=str(data), RESULT_DIR=str(out),
               TIMEOUT_SECONDS='10', FAKE_MODE='success', FAKE_GPUS='GPU-A\nGPU-B\n')
    return entrypoint, out, env


def validate(out, answered):
    rows = [json.loads(line) for line in (out / 'submission.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows] == ['q_0', 'q_1', 'q_2', 'q_3']
    assert sum(r['type'] != 'abstain' for r in rows) == answered
    assert len(list((out / 'logs').glob('q_*.json'))) == 4


@pytest.mark.parametrize('gpus', [1, 2])
def test_shards_and_devices(harness, gpus):
    script, out, env = harness
    env['NUM_GPUS'] = str(gpus)
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    validate(out, 4)
    records = [json.loads(p.read_text()) for p in sorted(out.glob('work/run.*/shard*/mock.json'))]
    assert [r['shard'] for r in records] == [f'{i}/{gpus}' for i in range(gpus)]
    assert [r['device'] for r in records] == ['GPU-A', 'GPU-B'][:gpus]


def test_masked_gpu_and_repeated_run(harness):
    script, out, env = harness
    env['CUDA_VISIBLE_DEVICES'] = 'GPU-X'
    for mode, answered, status in [('success', 4, 0), ('failure', 0, 7)]:
        env['FAKE_MODE'] = mode
        run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=20)
        assert run.returncode == status, run.stdout + run.stderr
        validate(out, answered)
    assert len(list(out.glob('work/run.*'))) == 2
    assert json.loads(next(out.glob('work/run.*/shard0/mock.json')).read_text())['device'] == 'GPU-X'


@pytest.mark.parametrize('setting,value,status', [('NUM_GPUS', '0', 2), ('NUM_GPUS', '3', 2),
    ('NUM_GPUS', 'invalid', 2), ('TIMEOUT_SECONDS', '21601', 2), ('LORA_DIR', '/forbidden', 2), ('FAKE_GPUS', '', 2)])
def test_startup_failure_leaves_submission(harness, setting, value, status):
    script, out, env = harness
    env[setting] = value
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == status, run.stdout + run.stderr
    validate(out, 0)


def test_timeout_keeps_partial_answers(harness):
    script, out, env = harness
    env.update(NUM_GPUS='2', TIMEOUT_SECONDS='2', FAKE_MODE='sleep')
    run = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 124, run.stdout + run.stderr
    validate(out, 2)


def test_term_keeps_partial_answers(harness):
    script, out, env = harness
    env.update(NUM_GPUS='1', FAKE_MODE='sleep')
    process = subprocess.Popen(['bash', str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not list(out.glob('work/run.*/shard0/mock.json')):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 143, stdout + stderr
        validate(out, 1)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.fixture
def context(tmp_path):
    for name in ('docker/requirements.txt', 'training/requirements.txt', 'assets/fewshot_bank.jsonl',
                 'docker/README.md', 'docker/predict.sh', 'scripts/flags.sh', 'docker/model_manifest.json',
                 'LICENSE', 'NOTICE.md', 'datamate/__init__.py'):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    subprocess.run([sys.executable, str(ROOT / 'scripts/pack_training_code.py'),
                    '--out', str(tmp_path / 'build/training_code')], check=True, capture_output=True)
    model = tmp_path / 'models/Qwen3.5-9B'
    metadata = model / '.cache/huggingface/download'
    metadata.mkdir(parents=True)
    files = {'config.json': '{}', 'tokenizer.json': '{}', 'tokenizer_config.json': '{}',
             'model.safetensors.index.json': json.dumps({'weight_map': {'test.weight': 'model-00001.safetensors'}}),
             'model-00001.safetensors': 'synthetic-test-weight'}
    revision = json.loads((ROOT / 'docker/model_manifest.json').read_text())['revision']
    import hashlib
    for name, content in files.items():
        (model / name).write_text(content)
        (metadata / f'{name}.metadata').write_text(f'{revision}\n{hashlib.sha256(content.encode()).hexdigest()}\n0\n')
    return tmp_path, model


def test_context_passes_then_rejects_wrong_revision(context):
    root, model = context
    verify(root)
    (model / '.cache/huggingface/download/config.json.metadata').write_text('wrong-revision\n')
    with pytest.raises(ValueError, match='revision mismatch'):
        verify(root)


def test_context_rejects_missing_weight(context):
    root, model = context
    (model / 'model-00001.safetensors').unlink()
    with pytest.raises(ValueError, match='missing/empty model file'):
        verify(root)


def test_context_rejects_truncated_weight(context):
    root, model = context
    (model / 'model-00001.safetensors').write_text('truncated')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(root)


def test_context_rejects_adapter(context):
    root, model = context
    (model / 'adapter_config.json').write_text('{}')
    with pytest.raises(ValueError, match='LoRA artifact must not ship'):
        verify(root)


def test_context_rejects_stale_flags(context):
    root, _ = context
    with (root / 'scripts/flags.sh').open('a') as stream:
        stream.write('\n# changed after packing\n')
    with pytest.raises(ValueError, match='is stale'):
        verify(root)


@pytest.mark.parametrize('gpus', [1, 2])
def test_build_helper_gpu_selection_and_error_status(harness, gpus):
    _, out, env = harness
    fake_docker = Path(env['PATH'].split(':')[0]) / 'docker-mock'
    fake_docker.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
record = Path(os.environ['DOCKER_CALLS'])
with record.open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
sys.exit(7 if '/code/predict.sh' in sys.argv else 0)
''')
    fake_docker.chmod(0o755)
    calls = out.parent / 'docker-calls.jsonl'
    env.update(DOCKER=str(fake_docker), DATA=env['DATA_DIR'], RESULT=str(out),
               NUM_GPUS=str(gpus), DOCKER_CALLS=str(calls))
    run = subprocess.run(['bash', str(ROOT / 'docker/build.sh'), 'test', '20'],
                         env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 7, run.stdout + run.stderr
    inference, check = [json.loads(line) for line in calls.read_text().splitlines()]
    assert inference[inference.index('--gpus') + 1] == ('"device=0"' if gpus == 1 else '"device=0,1"')
    assert inference[-3:] == ['/code/predict.sh', '--limit', '20']
    assert f'NUM_GPUS={gpus}' in inference
    assert '--network' in check and 'datamate.submission' in check
