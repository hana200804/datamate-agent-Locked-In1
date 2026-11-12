#!/usr/bin/env python3
"""Resumable base-model -> sampled SQL -> SFT -> own LoRA, without importing GPU libraries.

Run --dry-run to inspect every subprocess before spending Colab units. Runtime-only
venvs stay on local disk; model, logs, generated data and checkpoints live in --output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("setup", "preflight", "generate", "sft", "notes", "train", "public")
TRAIN_STAGES = STAGES[:-1]  # "all" = train flow; "public" only runs when named


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default=os.getenv("DATAMATE_STAGE", "all"),
                   help="all, or comma-separated setup,preflight,generate,sft,notes,train (in that order)")
    p.add_argument("--output", type=Path, default=Path(os.getenv("DATAMATE_OUTPUT", "/content/drive/MyDrive/datamate/from_base")))
    p.add_argument("--data", type=Path, default=Path(os.getenv("DATAMATE_DATA", str(ROOT / "data"))), help="parent of train/")
    p.add_argument("--env-dir", type=Path, default=Path(os.getenv("DATAMATE_ENV_DIR", "/content/datamate-venvs")), help="local, disposable venvs; rebuilt after session loss")
    p.add_argument("--gpu", choices=("auto", "a100", "h100", "g4"), default=os.getenv("DATAMATE_GPU", "auto"), help="auto detects compute capability; dry-run auto shows A100 commands")
    p.add_argument("--preflight-questions", type=int, default=10)
    p.add_argument("--startup-timeout", type=int, default=1200, help="seconds allowed for the inference engine to become ready")
    p.add_argument("--preflight-timeout", type=int, default=3600, help="total preflight seconds, including model startup")
    p.add_argument("--max-len", type=int, default=24576)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=-1, help="optional short training experiment; use a DIFFERENT output dir for a full run")
    p.add_argument("--max-rows", type=int, default=4000)
    p.add_argument("--easy", type=int, default=800)
    p.add_argument("--public-think-k", type=int, default=4, help="public stage: thinking samples per hard/debt question (0 = faster, weaker)")
    p.add_argument("--public-lora", type=Path, help="public stage: own adapter dir; default none = base model")
    p.add_argument("--dry-run", action="store_true", help="print all selected commands; no GPU checks, writes, installs or model imports")
    a = p.parse_args(argv)
    a.stages = list(TRAIN_STAGES) if a.stage == "all" else a.stage.split(",")
    if not a.stages or set(a.stages) - set(STAGES):
        p.error("--stage must be all or comma-separated names from " + ",".join(STAGES))
    if min(a.preflight_questions, a.save_steps, a.max_rows, a.grad_accum, a.max_len, a.startup_timeout, a.preflight_timeout) <= 0 or a.epochs <= 0 or a.max_steps == 0 or a.max_steps < -1:
        p.error("counts/timeouts/epochs must be positive; --max-steps is -1 or positive")
    a.output, a.data, a.env_dir = (x.expanduser().resolve() for x in (a.output, a.data, a.env_dir))
    return a


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def rows(path):
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a killed writer may leave its final line incomplete
                yield value


def selected_questions(a):
    ids = set(re.findall(r"q_[0-9a-f]+", (ROOT / "splits/rft_train24.txt").read_text()))
    qs = [q for q in rows(a.data / "train/questions.jsonl") if q["id"] in ids]
    if len(qs) != len(ids):
        raise RuntimeError(f"train/questions.jsonl must contain all {len(ids)} split IDs; found {len(qs)}")
    return qs


def inference_flags(a):
    # Read the shared shell flags without inheriting its HF_HUB_OFFLINE setting during setup.
    shell = 'source scripts/flags.sh; printf "%s\\n" "${DATAMATE_FLAGS[@]}"; printf "ENGINE=%s\\n" "$DATAMATE_ENGINE"'
    lines = subprocess.check_output(["bash", "-c", shell], cwd=ROOT, text=True).splitlines()
    flags = [x for x in lines if not x.startswith("ENGINE=")]
    engine = json.loads(next(x[7:] for x in lines if x.startswith("ENGINE=")))
    # Never forward an adapter path or either LoRA boolean, including environment defaults.
    for flag in ("--lora", "--lora-thinking"):
        while flag in flags:
            i = flags.index(flag)
            del flags[i:i + (2 if flag == "--lora" else 1)]
    gpu = a.gpu
    if not a.dry_run:
        try:
            info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], text=True).splitlines()[0]
            name, memory, cap = [s.strip() for s in info.split(",")]
            capability = tuple(map(int, cap.split(".")))
        except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as e:
            raise RuntimeError("No usable NVIDIA GPU; choose an A100 / H100 / G4 GPU runtime in Colab") from e
        if capability < (8, 0) or float(memory) < 22000:
            raise RuntimeError(f"{name}: need bf16 (Ampere+) and >=24 GB VRAM; T4/P100 do not qualify")
        gpu = "a100" if capability < (8, 9) else "h100"
        print(f"GPU: {info}; using {'bf16/auto KV' if gpu == 'a100' else 'fp8 weights/KV'}", flush=True)
    if gpu in ("auto", "a100"):
        i = flags.index("--quant")
        del flags[i:i + 2]
        flags[flags.index("--context") + 1] = "32768"
        engine.pop("kv_cache_dtype", None)
    return flags + ["--engine", json.dumps(engine, separators=(",", ":"))], "none" if gpu in ("auto", "a100") else "fp8"


class Flow:
    def __init__(self, a):
        self.a = a
        self.out = a.output
        self.py = a.env_dir / "inference/bin/python"
        self.train_py = a.env_dir / "training/bin/python"
        self.model = self.out / "models/Qwen3.5-9B"
        self.pool = self.out / "pool"
        self.sft = self.out / "sft/sql_rft.jsonl"
        self.notes = self.out / "sft/sql_rft_notes.jsonl"
        self.lora = self.out / "lora"
        self.manifest = json.loads((ROOT / "docker/model_manifest.json").read_text())
        self.flags = self.quant = None

    def gpu_flags(self):
        if self.flags is None:
            self.flags, self.quant = inference_flags(self.a)
        return self.flags

    def run(self, cmd, label, monitor=None, startup=False, timeout=None):
        cmd = [str(x) for x in cmd]
        print("$ " + shlex.join(cmd), flush=True)
        if self.a.dry_run:
            return
        log = self.out / "logs" / (label + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "DATAMATE_DATA": str(self.a.data),
               "HF_HOME": str(self.out / "cache/huggingface"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "VLLM_NO_USAGE_STATS": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "0",
               "PIP_CACHE_DIR": str(self.a.env_dir / "cache/pip"), "TORCHINDUCTOR_CACHE_DIR": str(self.out / "cache/torchinductor"),
               "TRITON_CACHE_DIR": str(self.out / "cache/triton")}
        if label == "download":
            env.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
        q = queue.Queue()
        started = time.monotonic()
        ready = not startup
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, errors="replace", start_new_session=True)

        def read_output():
            for line in proc.stdout:
                q.put(line)
            q.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        try:
            with log.open("a", encoding="utf-8") as f:
                f.write("\n$ " + shlex.join(cmd) + "\n")
                while True:
                    now = time.monotonic()
                    if not ready and now - started > self.a.startup_timeout:
                        raise RuntimeError(f"Engine did not start in {self.a.startup_timeout}s; inspect {log}; check GPU, CUDA and vLLM installation")
                    if timeout and now - started > timeout:
                        raise RuntimeError(f"Preflight exceeded {timeout}s; stop and inspect {log} before spending more units")
                    try:
                        line = q.get(timeout=1)
                    except queue.Empty:
                        continue
                    if line is None:
                        break
                    f.write(line)
                    f.flush()
                    print(line, end="", flush=True)
                    if "model ready in " in line:
                        ready = True
                    if monitor:
                        monitor(line, now - started)
            code = proc.wait()
            if code:
                raise RuntimeError(f"{label} exited {code}; see {log}; do not proceed until this is fixed")
        finally:
            # Stop the entire GPU subprocess tree on timeout / notebook interruption.
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
        print(f"{label}: elapsed {time.monotonic() - started:.1f}s", flush=True)

    def setup_env(self, training=False):
        py = self.train_py if training else self.py
        signature = hashlib.sha256((ROOT / ("training/requirements.txt" if training else "pyproject.toml")).read_bytes()).hexdigest()
        marker = py.parents[1] / ".installed.json"
        if not self.a.dry_run and marker.exists() and py.exists() and json.loads(marker.read_text()).get("sha256") == signature:
            print(f"skip installed environment: {py}")
            return
        # Colab's python3 has no ensurepip, so `python -m venv` fails; uv builds the venv and seeds pip into it.
        self.run([sys.executable, "-m", "pip", "install", "-q", "uv"], "install-uv")
        self.run([sys.executable, "-m", "uv", "venv", "--clear", "--seed", "--python", sys.executable, py.parents[1]],
                 "venv-training" if training else "venv-inference")
        self.run([py, "-m", "pip", "install", "--upgrade", "pip"], "pip-training" if training else "pip-inference")
        if training:
            self.run([py, "-m", "pip", "install", "-r", ROOT / "training/requirements.txt"], "install-training")
            self.run([py, "-m", "pip", "install", "--no-deps", "fla-core==0.6.0"], "install-fla")
        else:
            self.run([py, "-m", "pip", "install", "-e", f"{ROOT}[llm]"], "install-inference")
        if not self.a.dry_run:
            atomic_json(marker, {"sha256": signature})

    def setup(self):
        self.setup_env()
        marker = self.model / ".downloaded.json"
        expected = {k: self.manifest[k] for k in ("repo_id", "revision")}
        if not self.a.dry_run and marker.exists() and json.loads(marker.read_text()) == expected:
            print("skip pinned model download")
            return
        code = "from huggingface_hub import snapshot_download; snapshot_download(repo_id=" + repr(expected["repo_id"]) + ", revision=" + repr(expected["revision"]) + ", local_dir=" + repr(str(self.model)) + ")"
        self.run([self.py, "-c", code], "download")
        if not self.a.dry_run:
            atomic_json(marker, expected)

    def predict_cmd(self, output, ids, limit=None, split="train", flags=None):
        return [self.py, "-m", "datamate.predict", "--data", self.a.data / split, "--ids", ids,
                "--output", output, "--model", self.model, "--fewshot", ROOT / "assets/fewshot_bank.jsonl",
                *(flags if flags is not None else self.gpu_flags()), "--save-prompts", *(["--limit", limit] if limit else [])]

    def preflight(self):
        marker = self.out / "state/preflight.json"
        if not self.a.dry_run and marker.exists():
            print("skip completed preflight")
            return
        dest = self.out / "preflight"
        start = time.monotonic()
        self.run(self.predict_cmd(dest, ROOT / "splits/rft_train24.txt", self.a.preflight_questions), "preflight",
                 startup=True, timeout=self.a.preflight_timeout)
        if self.a.dry_run:
            return
        trace = list(rows(dest / "trace.jsonl"))
        executions = sum(bool(e.get("executed")) and e.get("error") is None and e.get("output") is not None
                         for p in dest.glob("logs/q_*.json") for r in rows(p) for e in r.get("executions", []))
        if len(trace) != self.a.preflight_questions or not executions:
            raise RuntimeError(f"Preflight FAILED: {len(trace)} questions, {executions} successful SQL executions; inspect {dest}/logs before generation")
        rate = len(trace) * 60 / (time.monotonic() - start)
        print(f"preflight: {rate:.3f} questions/min including startup; {executions} successful SQL executions", flush=True)
        atomic_json(marker, {"questions": len(trace), "questions_per_min": rate, "successful_executions": executions})

    def collect(self, qs, pool=None):
        """Keep only full shops, with their per-question logs, from immutable attempts."""
        pool = pool or self.pool
        expected = defaultdict(set)
        for q in qs:
            expected[q["business_id"]].add(q["id"])
        complete = {}
        for attempt in sorted(pool.glob("attempt-*")):
            shops = defaultdict(dict)
            for r in rows(attempt / "trace.jsonl"):
                if r.get("prompt_user") and (attempt / "logs" / f"{r['id']}.json").exists():
                    shops[r["business_id"]][r["id"]] = r
            for shop, records in shops.items():
                if set(records) == expected.get(shop):
                    complete.update(records)
        return complete

    def generate(self):
        marker = self.out / "state/generate.json"
        if not self.a.dry_run and marker.exists() and (self.pool / "trace.jsonl").exists():
            print("skip completed labelled train run")
            return
        if self.a.dry_run:
            self.run(self.predict_cmd(self.pool / "attempt-0001", self.pool / "pending.txt"), "generate-0001", startup=True)
            print("# pending.txt = split IDs minus complete shops in previous attempts; atomically consolidate pool/trace.jsonl")
            return
        preflight = self.out / "state/preflight.json"
        if not preflight.exists():
            raise RuntimeError("Run --stage setup,preflight first; generation is gated on successful SQL execution")
        qs = selected_questions(self.a)
        complete = self.collect(qs)
        pending = [q["id"] for q in qs if q["id"] not in complete]
        self.pool.mkdir(parents=True, exist_ok=True)
        (self.pool / "pending.txt").write_text("\n".join(pending) + "\n")
        rate = json.loads(preflight.read_text())["questions_per_min"]
        print(f"generate: {len(complete)}/{len(qs)} preserved; {len(pending)} pending; preflight {rate:.3f} questions/min, rough {len(pending) / max(rate, 1e-9) / 60:.2f} hours (small-sample estimate)", flush=True)
        print(f"prompt budget: {len(qs)} x 8 x ~9k-19k = {len(qs)*8*9000/1e6:.1f}-{len(qs)*8*19000/1e6:.1f}M input tokens; up to 4 extra thinking samples, plus drafts/repairs; prefix caching may reduce compute", flush=True)
        if pending:
            n = max([int(p.name.split("-")[-1]) for p in self.pool.glob("attempt-*")], default=0) + 1
            count = 0

            def progress(line, elapsed):
                nonlocal count
                m = re.search(r"\] (\d+) q,", line)
                if m:
                    count += int(m[1])
                    print(f"generate throughput: {count*60/max(elapsed, 1e-9):.3f} questions/min ({count} new; startup/drafts included)", flush=True)

            self.run(self.predict_cmd(self.pool / f"attempt-{n:04d}", self.pool / "pending.txt"), f"generate-{n:04d}", progress, startup=True)
        complete = self.collect(qs)
        if len(complete) != len(qs):
            raise RuntimeError(f"Only {len(complete)}/{len(qs)} questions in full shops; retry generate to finish remaining shops")
        temp = self.pool / "trace.jsonl.tmp"
        with temp.open("w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(complete[q["id"]], ensure_ascii=False) + "\n")
        temp.replace(self.pool / "trace.jsonl")
        atomic_json(marker, {"questions": len(qs), "shops": len({q['business_id'] for q in qs})})

    def public(self):
        """1000 public questions -> public/submission.jsonl for the portal; resumes by shop like generate."""
        dest, split = self.out / "public", "public_test"
        qfile = self.a.data / split / "questions.jsonl"
        flags = list(self.gpu_flags())
        flags[flags.index("--think-k") + 1] = str(self.a.public_think_k)
        lora = []
        if self.a.public_lora:
            lora = ["--lora", self.a.public_lora, "--lora-thinking"]
        if self.a.dry_run:
            self.run(self.predict_cmd(dest / "attempt-0001", dest / "pending.txt", split=split, flags=[*flags, *lora]), "public-0001", startup=True)
            return
        if not qfile.exists():
            raise RuntimeError(f"Missing {qfile}; upload public_test.zip next to train.zip")
        qs = list(rows(qfile))
        complete = self.collect(qs, dest)
        pending = [q["id"] for q in qs if q["id"] not in complete]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "pending.txt").write_text("\n".join(pending) + "\n")
        print(f"public: {len(complete)}/{len(qs)} done in full shops; {len(pending)} pending; think-k {self.a.public_think_k}; "
              f"{'LoRA ' + str(self.a.public_lora) if lora else 'base model, no LoRA'}", flush=True)
        if pending:
            n = max([int(p.name.split("-")[-1]) for p in dest.glob("attempt-*")], default=0) + 1
            self.run(self.predict_cmd(dest / f"attempt-{n:04d}", dest / "pending.txt", split=split, flags=[*flags, *lora]),
                     f"public-{n:04d}", startup=True)
            complete = self.collect(qs, dest)
        with (dest / "submission.jsonl.tmp").open("w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps({"id": q["id"], **(complete[q["id"]]["final"] if q["id"] in complete else {"type": "abstain"})},
                                   ensure_ascii=False, default=str) + "\n")
        (dest / "submission.jsonl.tmp").replace(dest / "submission.jsonl")
        answered = sum(complete[q["id"]]["final"].get("type") != "abstain" for q in qs if q["id"] in complete)
        print(f"public: {len(complete)}/{len(qs)} questions run, {answered} answered -> {dest / 'submission.jsonl'}", flush=True)
        self.run([self.py, "-m", "datamate.submission", "check", dest / "submission.jsonl", "--questions", qfile], "public-check")
        # the notebook releases the runtime when it ends (and /content with it): keep a copy in the cell output too.
        # decode: the lines between the markers -> base64 -d | gunzip; md5 is of the plain file
        import base64, gzip, textwrap
        raw = (dest / "submission.jsonl").read_bytes()
        b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
        nl = raw.count(b"\n")
        print(f"SUBMISSION-B64-BEGIN lines={nl} md5={hashlib.md5(raw).hexdigest()}\n"
              + "\n".join(textwrap.wrap(b64, 120)) + "\nSUBMISSION-B64-END", flush=True)
        if len(complete) != len(qs):
            raise RuntimeError(f"Only {len(complete)}/{len(qs)} public questions finished; rerun the public stage to resume")

    def sft_stage(self):
        if not self.a.dry_run and self.sft.exists() and self.sft.stat().st_size:
            print("skip completed SFT file")
            return
        if not self.a.dry_run and not (self.out / "state/generate.json").exists():
            raise RuntimeError("Complete generate before building SFT")
        temp = self.sft.with_suffix(".partial.jsonl")
        self.run([self.py, ROOT / "training/build_sft.py", self.pool, "--fewshot", ROOT / "assets/fewshot_bank.jsonl",
                  "--out", temp, "--easy", self.a.easy, "--max", self.a.max_rows], "sft")
        if not self.a.dry_run:
            if not temp.stat().st_size:
                raise RuntimeError("No correct samples produced SFT rows; inspect logs/sft.log")
            temp.replace(self.sft)

    def notes_stage(self):
        if not self.a.dry_run and self.notes.exists() and self.notes.stat().st_size:
            print("skip completed notes file")
            return
        self.gpu_flags()
        if not self.a.dry_run and not self.sft.exists():
            raise RuntimeError("Complete sft before notes")
        temp = self.notes.with_suffix(".partial.jsonl")
        self.run([self.py, ROOT / "training/add_notes.py", self.sft, temp, "--model", self.model,
                  "--rulebook", self.a.data / "train/rulebook.md", "--quant", self.quant], "notes")
        if not self.a.dry_run:
            if not temp.exists() or not temp.stat().st_size:
                raise RuntimeError("Notes output is empty")
            temp.replace(self.notes)

    def train(self):
        marker = self.out / "state/train.json"
        weights = self.lora / "adapter/adapter_model.safetensors"
        if not self.a.dry_run and marker.exists() and weights.exists() and weights.stat().st_size > 1000000:
            print("skip completed own adapter")
            return
        if not self.a.dry_run and not self.notes.exists():
            raise RuntimeError("Complete notes before train")
        self.gpu_flags()
        self.setup_env(training=True)
        # check_fla.py prints FAIL but does not exit nonzero, so explicitly check its log.
        self.run([self.train_py, ROOT / "training/check_fla.py", "--T", "2048"], "check-fla")
        if not self.a.dry_run and "fla check PASS" not in (self.out / "logs/check-fla.log").read_text().split("$ ")[-1]:
            raise RuntimeError("fla kernel check failed; aborting rather than spending the budget on the slow reference kernel")
        initial = 0
        if not self.a.dry_run:
            # A session can die while Trainer is writing to Drive. Its resume glob
            # otherwise selects that half-written checkpoint forever.
            required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                        "adapter_model.safetensors", "adapter_config.json")
            for checkpoint in sorted(self.lora.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
                valid = all((checkpoint / name).exists() and (checkpoint / name).stat().st_size for name in required)
                try:
                    state = json.loads((checkpoint / "trainer_state.json").read_text()) if valid else {}
                    valid = valid and state.get("global_step") == int(checkpoint.name.split("-")[-1])
                except (OSError, ValueError):
                    valid = False
                if valid:
                    initial = state["global_step"]
                else:
                    incomplete = self.lora / "incomplete" / f"{checkpoint.name}-{time.time_ns()}"
                    incomplete.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint.rename(incomplete)
                    print(f"Preserved incomplete checkpoint at {incomplete}; resuming previous complete checkpoint", flush=True)
        train_started = None
        last = initial

        def progress(line, elapsed):
            nonlocal train_started, last
            if "precision:" in line or "resuming from " in line:
                train_started = elapsed
            m = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
            if m:
                step = int(m[1])
                if train_started is None:
                    train_started = elapsed
                if step > last and step > initial:
                    seconds = max(elapsed - train_started, 1e-9)
                    print(f"train throughput: {(step-initial)*60/seconds:.3f} steps/min; {step}/{m[2]} steps; excludes setup/model load", flush=True)
                    last = step

        self.run([self.train_py, ROOT / "training/train_lora.py", self.notes, "--out", self.lora, "--model", self.model,
                  "--max-len", self.a.max_len, "--r", "16", "--alpha", "32", "--lr", "1e-4", "--epochs", self.a.epochs,
                  "--grad-accum", self.a.grad_accum, "--end", "<|im_end|>", "--no-thinking", "--completion-logits",
                  "--max-steps", self.a.max_steps, "--save-steps", self.a.save_steps, "--resume"], "train", progress)
        if not self.a.dry_run:
            if not weights.exists() or weights.stat().st_size < 1000000:
                raise RuntimeError("Training exited without valid own-adapter weights")
            atomic_json(marker, {"adapter": str(weights.parent), "initial_step": initial})


def main(argv=None):
    a = arguments(argv)
    if not a.dry_run and sys.version_info < (3, 12):
        raise SystemExit("Python >=3.12 is required; select a current Colab runtime")
    flow = Flow(a)
    if not a.dry_run:
        a.output.mkdir(parents=True, exist_ok=True)
        # Changing training/sample settings must not silently reuse stale artifacts.
        settings = {k: getattr(a, k) for k in ("preflight_questions", "max_len", "epochs", "grad_accum", "save_steps", "max_steps", "max_rows", "easy")}
        settings["base_revision"] = flow.manifest["revision"]
        settings["split_sha256"] = hashlib.sha256((ROOT / "splits/rft_train24.txt").read_bytes()).hexdigest()
        config = a.output / "flow.json"
        if config.exists() and json.loads(config.read_text()) != settings:
            raise RuntimeError("Parameters differ from flow.json; restore the original parameters to resume, or choose a new --output")
        atomic_json(config, settings)
    print(f"output: {a.output}; stages: {','.join(a.stages)}; base revision: {flow.manifest['revision']}", flush=True)
    methods = {"setup": flow.setup, "preflight": flow.preflight, "generate": flow.generate,
               "sft": flow.sft_stage, "notes": flow.notes_stage, "train": flow.train, "public": flow.public}
    for stage in STAGES:
        if stage in a.stages:
            start = time.monotonic()
            print(f"\n=== {stage} ===", flush=True)
            try:
                methods[stage]()
            finally:
                print(f"stage {stage}: elapsed {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print(f"STOP: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
