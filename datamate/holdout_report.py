"""Score a run's trace against train gold and say where the points go: PA, the vote threshold sweep, per shop and per type.

    python -m datamate.holdout_report --trace RUN/trace.jsonl --data data/train [--label think4]

Prints one JSON summary and a gzip+base64 block of compact per-question rows between HOLDOUT-B64-BEGIN/END (Colab wipes /content
when the notebook ends; the cell output survives). Evaluation only: inference never imports this module.
"""
import argparse
import base64
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

from .scoring import read_jsonl, score
from .voting import final_answer


def oor_of(final):
    return final.get("type") == "unanswerable" and "outside" in str(final.get("reason", ""))


def pa(scores):
    n = len(scores)
    c, w = sum(s == 1 for s in scores), sum(s == -1 for s in scores)
    return {"n": n, "PA": round((c - w) / n, 4) if n else None, "correct": c, "wrong": w, "abstain": n - c - w}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--label", default="run")
    p.add_argument("--threshold", type=float, default=0.5, help="the run's --threshold")
    a = p.parse_args(argv)
    gold = read_jsonl(a.data / "answers.jsonl")
    recs = [json.loads(l) for l in a.trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for r in recs:
        g = gold[r["id"]]
        preds = [s.get("pred") for s in r.get("samples", [])]
        rows.append({"id": r["id"], "biz": r["business_id"], "gtype": g["type"], "final": score(r["final"], g),
                     "vote": score(r["vote"], g) if r.get("vote") else 0, "conf": round(float(r.get("conf") or 0), 4),
                     "n": len(preds), "any": int(any(pr and score(pr, g) == 1 for pr in preds)),
                     "ftype": r["final"].get("type"), "vtype": (r.get("vote") or {}).get("type"), "oor": int(oor_of(r["final"]))})
    by_id = {r["id"]: r for r in recs}
    sweep = {}
    for t in [0.25, 0.3, 0.375, 0.4, 0.45, 0.5, 0.55, 0.6, 0.625, 0.65, 0.7, 0.75, 0.8, 0.875, 0.9, 1.0]:
        s = [score(final_answer(by_id[x["id"]].get("vote") or {"type": "abstain"}, x["conf"], t, bool(x["oor"]), True), gold[x["id"]])
             for x in rows]
        sweep[str(t)] = pa(s)
    buckets = defaultdict(list)  # vote correctness by vote share: is answering this bucket worth it (2p-1 > 0)?
    for x in rows:
        if x["vtype"] not in (None, "abstain", "unanswerable"):
            buckets[f"{x['conf']:.3f}"].append(x["vote"])
    by = lambda k: {v: pa([x["final"] for x in rows if x[k] == v]) for v in sorted({x[k] for x in rows})}
    summary = {"label": a.label, "overall": pa([x["final"] for x in rows]),
               "oracle_any_sample_correct": sum(x["any"] for x in rows),
               "thinking_used": sum(x["n"] > 8 for x in rows),
               "threshold_sweep": sweep,
               "vote_share_buckets": {k: {"n": len(v), "right": v.count(1), "wrong": v.count(-1)} for k, v in sorted(buckets.items())},
               "by_shop": by("biz"), "by_gold_type": by("gtype")}
    print("HOLDOUT-SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    raw = "\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in rows).encode() + b"\n"
    b64 = base64.b64encode(gzip.compress(raw, 9)).decode()
    print(f"HOLDOUT-B64-BEGIN label={a.label} lines={len(rows)} md5={hashlib.md5(raw).hexdigest()}\n"
          + "\n".join(textwrap.wrap(b64, 120)) + "\nHOLDOUT-B64-END", flush=True)


if __name__ == "__main__":
    main()
