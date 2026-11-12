"""Label open-ended benchmark answers with a text LLM judge (default Qwen3-32B), for every run at once.

benchmark.py --no_score writes predictions per run. MCQ answers and exact matches need no judge; this
labels the rest as correct / false_claim / wrong (qa_prompts.GRADE_SYSTEM) and writes grades.jsonl next
to each run's predictions, with one model load for all runs. Then: benchmark.py --score_only.

  python grade.py --runs 'runs/bench_*/*'           # every run folder under runs/
  python grade.py --mock --runs 'out_bench/*'
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import glob
import os
import re

import benchmark as B
import qa_common as C
import qa_prompts as P
from llm import LazyChat, add_llm_args, make_backend, reasons, resolve_defaults


def mock_grade(msgs):
    u = msgs[-1]["content"]
    get = lambda k: (re.search(rf"^{k}: (.*)$", u, re.M) or [None, ""])[1]
    lab = B.match_label(get("Model's answer"), get("True answer"), get("Known false answer"))
    return f'{{"reason": "mock", "label": "{lab}"}}'


def run_dirs(spec):
    out = []
    for pat in spec.split(","):
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "predictions*.jsonl")) and d not in out:
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="benchmark run folders (globs, comma separated)")
    ap.add_argument("--mock", action="store_true", help="word matching instead of a model, no GPU")
    ap.add_argument("--batch", type=int, default=2048, help="answers per write to disk")
    add_llm_args(ap, "Qwen/Qwen3-32B", 256)
    C.add_gpu_args(ap)
    C.add_shard_args(ap)
    a = ap.parse_args()
    if a.mock:
        a.backend, a.model = "mock", "mock"
    resolve_defaults(a, 256)
    if not a.mock:
        C.setup_caches()

    # one judge call per distinct (question, truth, claim, answer), shared by every run that has it
    todo = {}
    for d in run_dirs(a.runs):
        done = {r["qa_id"] for r in C.read_shards(os.path.join(d, "grades.jsonl")) if r.get("label")}
        for p in C.read_shards(os.path.join(d, "predictions.jsonl")):
            if p["format"] == "mcq" or p["qa_id"] in done or not C.in_shard(a, p["example_id"]):
                continue
            if B.quick_label(p["pred"], p["truth"], p["false"]) is None:
                todo.setdefault((p["question"], p["truth"], p["false"] or "", p["pred"]), []).append((d, p["qa_id"]))
    print(f"[grade] {sum(map(len, todo.values()))} answers to label ({len(todo)} distinct) in "
          f"{len(run_dirs(a.runs))} runs", flush=True)
    if not todo:
        return

    def load():
        if a.backend == "vllm":
            C.pick_gpus(a)
        return make_backend(a, mock_fn=mock_grade)
    chat = LazyChat(load)
    keys = list(todo)
    for s in range(0, len(keys), a.batch):
        chunk = keys[s:s + a.batch]
        convs = [P.grade_messages(q.split("\n")[0].replace(P.OPEN_SUFFIX, ""), t, f, pred) for q, t, f, pred in chunk]
        outs = chat(convs, schema=P.GRADE_SCHEMA, temperature=None if reasons(a) else 0.0)
        per_run = {}
        for k, o in zip(chunk, outs):
            obj = C.parse_json(o) or {}
            lab = obj.get("label") if obj.get("label") in ("correct", "false_claim", "wrong") else None
            for d, qa_id in todo[k]:
                per_run.setdefault(d, []).append(dict(qa_id=qa_id, label=lab, reason=str(obj.get("reason", "")),
                                                      model=a.model))
        for d, rows in per_run.items():
            C.append_jsonl(C.shard_path(a, os.path.join(d, "grades.jsonl")), rows)
        print(f"[grade] {min(s + a.batch, len(keys))}/{len(keys)}", flush=True)


if __name__ == "__main__":
    main()
