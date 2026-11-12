"""Everything in one command, spread over every free GPU: the captioner (Step 1 score checks, Step 2 GRPO
training and its before/after caption evaluation), then Caption 2 on DOCCI from that captioner, then the
Positive / Negative / Hallucinated QA (generate + check, build), the benchmark, grading and one report.

  A. step1    run.py: pick and check the FreeRet score (SugarCrepe, DOCCI checks, gate)  -> <out_root>/step1/
  B. grpo     grpo.py: GRPO training of the Qwen2.5-VL captioner with that score as reward (captions up
              to --grpo_max_new_tokens 4096), then the before/after caption evaluation on COCO val and
              DOCCI                                                                     -> <out_root>/grpo/
              (A and B are skipped when their results are already there; --step1_dir / --grpo_dir can
              point at runs you already have, e.g. out_7b and out_grpo. A GRPO folder trained with another
              caption length, like the old 128-token out_grpo, is not reused: B runs again in
              <grpo_dir>_tok4096 and the old folder is left as it is)
  C. caption  caption_docci.py: Caption 2 for every DOCCI image by the same captioner and the same
              generation code as grpo.py, before RL (LoRA off) and after RL (GRPO LoRA on), once per
              caption prompt (simple = "Describe this image in detail."; argmax = "Describe this image.", the GRPO prompt;
              up to --caption_max_tokens)
                                                       -> <out_root>/caption2_<before|after>__<prompt>/
              (--captioners vllm adds any VLM through caption.py; --caption2_file uses your own captions)
  D. gen, verify  for each Caption 2 and each generator: P/N/H QA from Caption 1 (DOCCI) + Caption 2,
              checked by that same model; build: qa_dataset.json (per image, with which sets it has)
              and qa_final.jsonl                       -> <out_root>/qa_<gen>__<captioner>__<prompt>/
  E. bench, grade  every VLM answers every dataset, with and without the image; --grader labels the
              open-ended answers; scores                -> <out_root>/bench_<gen>__<captioner>__<prompt>/
  F. summary.md / summary.html: the captioner results (A, B), datasets, coverage, every benchmark, and
              sample images for every coverage case in one layout (rebuild alone with python report.py)

Multi-GPU: C, D and E look for free GPUs among --gpus, load one model copy per --gpus_per_model GPUs and
split the images between the copies (data parallel); A and B use one free GPU each (as run.py / grpo.py
always did). GPUs 0, 1, 2, 3 are never used. Each step resumes from what is on disk: after a crash, run
the same command again.

Results and logs go to runs/ in the folder you run from (--out_root); they are small (the GRPO adapter,
~0.2 GB, is the largest). Big files go to /mnt/data2/mchung45: DOCCI and vLLM / torch caches and temp
files under qa_common.WORK_ROOT (docci_qa_work), COCO / SugarCrepe downloads for A and B under
argmax_caption_work/cache (as before), and A / B's embedding cache and GRPO checkpoints (several GB)
under argmax_caption_work/outputs (--big_root). Model weights stay in HF_HOME if it is set.

  python run_all.py --mock                                   # smoke test: tiny models / fake data, CPU
  python run_all.py --n_images 200 --out_root runs_try200
  python run_all.py --step1_dir out_7b --grpo_dir out_grpo   # reuse Step 1; out_grpo (128 tokens) -> out_grpo_tok4096
  nohup python run_all.py > /dev/null 2>&1 &                 # DOCCI test, all 5,000 images
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import time

import qa_common as C
import qa_prompts as P
import report

HERE = os.path.dirname(os.path.abspath(__file__))
# Step 1 / GRPO's own big-disk folder (its download cache is <this>/cache, as in argmax_caption)
ARGMAX_WORK = os.environ.get("ARGMAX_WORK_DIR", "/mnt/data2/mchung45/argmax_caption_work")

# Rough download size (GB) of the weights, used only for the disk check at start-up.
MODEL_GB = {"qwen3-32b": 66, "gpt-oss-120b": 65, "gpt-oss-20b": 14, "qwen2.5-vl-7b": 17,
            "qwen2.5-vl-72b": 146, "qwen3-vl-8b": 18, "qwen3-vl-32b": 66, "gemma-3-27b": 55,
            "mistral-small-3.2-24b": 48, "llama-3.3-70b": 141}


def tag(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(model.rstrip("/")))


# ------------------------------------------------------------------ disk
def setup_cache_root(a):
    C.setup_caches(a.cache_root, a.models_in_root)


def hf_hub_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def model_cached(m):
    return os.path.isdir(os.path.join(hf_hub_dir(), "models--" + m.replace("/", "--")))


def big_dir(a, path, sub):
    """Where the heavy part of a Step 1 / GRPO folder goes (embedding cache, checkpoints): on --big_root,
    in a folder named after the results folder. An older run that already has <path>/<sub> keeps it."""
    if os.path.isdir(os.path.join(path, sub)):
        return os.path.join(path, sub)
    return os.path.join(a.big_root, re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/")), sub)


def trained_tokens(d):
    """Caption length limit a GRPO folder was trained with; None when nothing was trained there yet.
    Folders from before adapter/grpo_train.json existed fall back to args.json, then to 128 (the old default)."""
    for p in ("adapter/grpo_train.json", "args.json"):
        j = report.load_json(os.path.join(d, p))
        if j:
            return j.get("max_new_tokens", 128)
    return None


def adapter_id(adapter):
    """Which adapter made the 'after' captions: path + trained length + file time (a retrain changes it)."""
    f = os.path.join(adapter, "adapter_model.safetensors")
    return dict(adapter=adapter, trained_tokens=trained_tokens(os.path.dirname(adapter)),
                mtime=int(os.path.getmtime(f)) if os.path.exists(f) else None)


def check_disk(a, models):
    need = 0
    lines = []
    for m in models:
        gb = next((v for k, v in MODEL_GB.items() if k in m.lower()), None)
        cached = model_cached(m)
        lines.append(f"  {m}: {'already downloaded' if cached else f'~{gb} GB to download' if gb else 'size unknown'}")
        if not cached and gb:
            need += gb
    hub = hf_hub_dir()
    os.makedirs(hub, exist_ok=True)
    places = {"model weights (HF_HOME)": hub, "DOCCI (--cache_dir)": a.cache_dir, "results (--out_root)": a.out_root,
              "temp files (TMPDIR)": os.environ.get("TMPDIR", "/tmp"),
              "Step 1 embeddings, GRPO checkpoints (--big_root)": a.big_root}
    print("[disk] models:\n" + "\n".join(lines))
    # (path, GB needed): models; DOCCI images for 5,000 test images are a few GB; results are small
    need_gb = {"model weights (HF_HOME)": need + 20, "DOCCI (--cache_dir)": 10, "results (--out_root)": 1,
               "temp files (TMPDIR)": 5, "Step 1 embeddings, GRPO checkpoints (--big_root)": a.big_need}
    problems = []
    for name, p in places.items():
        os.makedirs(p, exist_ok=True)
        free = shutil.disk_usage(p).free / 1e9
        print(f"[disk] {name}: {p}  ->  {free:.0f} GB free")
        if free < need_gb[name]:
            problems.append(f"{name} at {p} has {free:.0f} GB free, needs ~{need_gb[name]} GB")
    if problems:
        msg = ("[disk] not enough space:\n  " + "\n  ".join(problems) +
               "\n  Move it to a disk with room: --cache_dir / --out_root for DOCCI and results, "
               "--cache_root for everything (an HF_HOME already set is kept), or pass --skip_disk_check.")
        if a.skip_disk_check:
            print(msg + " (continuing because of --skip_disk_check)")
        else:
            raise SystemExit(msg)


# ------------------------------------------------------------------ running steps
class Runner:
    def __init__(self, a):
        self.a = a
        self.log_dir = os.path.join(a.out_root, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.failed = []
        self.n = 0

    def model_args(self):
        a = self.a
        out = ["--seed", str(a.seed)]
        if a.mock:
            out += ["--mock"]
        if a.backend != "vllm":
            out += ["--backend", a.backend, "--base_url", a.base_url]
        return out

    def common(self):
        a = self.a
        out = ["--cache_dir", a.cache_dir] + self.model_args()
        if a.docci_dir:
            out += ["--docci_dir", a.docci_dir]
        return out

    def gpu_groups(self):
        """Free GPUs among --gpus, in groups of --gpus_per_model; waits until there is at least one group."""
        a = self.a
        cands = C.gpu_candidates(a.gpus, mock=a.mock or a.backend != "vllm")
        k = a.gpus_per_model
        while True:
            if a.mock or a.backend != "vllm":
                free = cands
            else:
                g = C._query_gpus()
                free = [i for i in cands if i in g and g[i]["used"] <= a.gpu_max_mem_mb
                        and g[i]["util"] <= a.gpu_max_util]
                print("[gpu] " + " | ".join(f"{i}: {g[i]['used']}/{g[i]['total']} MiB, {g[i]['util']}%"
                                             for i in cands if i in g), flush=True)
            groups = [free[i:i + k] for i in range(0, len(free) - k + 1, k)]
            if a.max_parallel:
                groups = groups[:a.max_parallel]
            if groups:
                return groups
            if a.no_gpu_wait:
                return []
            print(f"[gpu] no {k} free GPU(s) among {cands}, checking again in 60s ...", flush=True)
            time.sleep(60)

    def run(self, name, script, args, parallel=False):
        """One step. parallel=True: one process per GPU group, each with --shard k/N."""
        self.n += 1
        groups = self.gpu_groups() if parallel else [None]
        if not groups:
            print(f"===== [{self.n}] {name}: no free GPU, skipped", flush=True)
            self.failed.append((name, "no free GPU"))
            return False
        N = len(groups)
        print(f"\n===== [{self.n}] {name}" + (f" on {N} GPU group(s): {groups}" if parallel else ""), flush=True)
        t0 = time.time()
        procs = []
        for k, grp in enumerate(groups):
            cmd = [sys.executable, os.path.join(HERE, script)] + args
            env = dict(os.environ)
            if parallel:
                cmd += ["--shard", f"{k}/{N}", "--n_gpus", str(len(grp)), "--no_gpu_pick"]
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, grp))
            log = os.path.join(self.log_dir, f"{self.n:02d}_{name}" + (f".{k}-of-{N}" if N > 1 else "") + ".log")
            f = open(log, "a")
            f.write(f"$ CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')} {' '.join(cmd)}\n")
            f.flush()
            procs.append((subprocess.Popen(cmd, cwd=self.a.out_root, env=env, stdout=f, stderr=subprocess.STDOUT), f, log))
        print(f"  logs: {procs[0][2]}" + (" (+ one per shard)" if N > 1 else ""), flush=True)
        last = 0
        while any(p.poll() is None for p, _, _ in procs):
            time.sleep(1 if self.a.mock else 5)
            if time.time() - last > 120:  # heartbeat: last line of each shard's log
                last = time.time()
                for p, _, log in procs:
                    print(f"  [{os.path.basename(log)}] {tail(log)}", flush=True)
        ok = True
        for p, f, log in procs:
            f.close()
            if p.returncode != 0:
                ok = False
                self.failed.append((name, log))
                print(f"  FAILED (exit {p.returncode}), last lines of {log}:\n{tail(log, 15)}", flush=True)
        print(f"===== [{self.n}] {name}: {'done' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        if parallel and not self.a.mock:
            time.sleep(self.a.gpu_settle_s)  # let the driver release the model's memory
        return ok


class Tee:
    """Console output also goes to <out_root>/run_all.log, so nohup's own file is not needed."""

    def __init__(self, path):
        self.f = open(path, "a")
        self.out = sys.__stdout__

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def tail(path, n=1):
    try:
        with open(path, errors="replace") as f:
            lines = [l.rstrip() for l in f.read().replace("\r", "\n").splitlines() if l.strip()]
        return "\n".join(lines[-n:])
    except OSError:
        return ""


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="step1,grpo,caption,gen,verify,bench,grade",
                    help="any of step1,grpo,caption,gen,verify,bench,grade (build, scores and the report always run)")
    c = ap.add_argument_group("captioner (Step 1 + GRPO)")
    c.add_argument("--captioner_model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="model run.py / grpo.py train")
    c.add_argument("--step1_dir", default=None, help="Step 1 folder (default <out_root>/step1; skipped if done)")
    c.add_argument("--grpo_dir", default=None, help="GRPO folder with adapter/ (default <out_root>/grpo; skipped if done)")
    c.add_argument("--step1_args", default="", help='extra arguments for run.py, e.g. "--n_docci 200"')
    c.add_argument("--grpo_max_new_tokens", type=int, default=4096,
                   help="caption length limit for GRPO training and its evaluation. A --grpo_dir trained with "
                        "another limit (e.g. the old 128) is not reused: GRPO runs again in <grpo_dir>_tok<N>")
    c.add_argument("--grpo_args", default="", help='extra arguments for grpo.py, e.g. "--steps 300 --rollouts 4"')
    k = ap.add_argument_group("Caption 2")
    k.add_argument("--captioners", default="before,after",
                   help="before = captioner with LoRA off, after = GRPO LoRA on (both via caption_docci.py); "
                        "vllm = --caption_model through caption.py")
    k.add_argument("--caption_prompts", default="simple",
                   help="caption prompts (qa_prompts.CAPTION_PROMPTS); each gives its own Caption 2 and QA")
    k.add_argument("--caption_max_tokens", type=int, default=4096, help="max new tokens per Caption 2")
    k.add_argument("--caption_batch", type=int, default=8, help="images per generate call in caption_docci.py")
    k.add_argument("--caption_model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLM for --captioners vllm")
    k.add_argument("--caption2_file", default=None,
                   help="use these Caption 2 instead (JSONL: example_id, caption); skips step C")
    q = ap.add_argument_group("QA and benchmark")
    q.add_argument("--generators", default="Qwen/Qwen3-32B,openai/gpt-oss-120b",
                   help="text LLMs that write the QA, comma separated")
    q.add_argument("--vlms", default="Qwen/Qwen2.5-VL-7B-Instruct", help="VLMs to benchmark, comma separated")
    q.add_argument("--grader", default="Qwen/Qwen3-32B",
                   help="text LLM that labels open-ended answers; none = word matching only")
    q.add_argument("--skip_blind", action="store_true", help="do not run the no-image baseline")
    q.add_argument("--qa_format", default="open-ended", choices=["open-ended", "mcq", "mixed"])
    q.add_argument("--hal_mcq_truth", action="store_true",
                   help="hallucinated MCQ: include the Caption 1 truth among the options")
    q.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high"])
    q.add_argument("--backend", default="vllm", choices=["vllm", "openai"],
                   help="openai = one running server that serves every QA model named here")
    q.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--n_images", type=int, default=0, help="0 = every image in --splits")
    ap.add_argument("--mock", action="store_true",
                    help="smoke test: tiny random captioner for A/B, fake captions and QA models, CPU")
    g = ap.add_argument_group("GPUs")
    g.add_argument("--gpus", default="auto",
                   help="GPUs this run may use (nvidia-smi indices); auto = every GPU except 0,1,2,3")
    g.add_argument("--gpus_per_model", type=int, default=1,
                   help="GPUs per model copy (tensor parallel). 1 fits every default model on an H200")
    g.add_argument("--max_parallel", type=int, default=0, help="at most this many model copies (0 = no limit)")
    g.add_argument("--gpu_max_mem_mb", type=int, default=1000, help="a GPU counts as free below this memory use")
    g.add_argument("--gpu_max_util", type=int, default=10, help="... and below this utilization %%")
    g.add_argument("--no_gpu_wait", action="store_true", help="skip a step instead of waiting for free GPUs")
    g.add_argument("--gpu_settle_s", type=int, default=15)
    d = ap.add_argument_group("disk")
    d.add_argument("--cache_root", default=os.path.join(C.WORK_ROOT, "caches"),
                   help="vLLM/torch caches, temp files, .pyc, and model weights unless HF_HOME is already set")
    d.add_argument("--big_root", default=None,
                   help="Step 1 embedding cache and GRPO checkpoints (several GB; default "
                        f"{ARGMAX_WORK}/outputs, with --mock <out_root>/_big)")
    d.add_argument("--models_in_root", action="store_true",
                   help="download model weights under --cache_root even if HF_HOME is set (re-downloads them)")
    d.add_argument("--skip_disk_check", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--samples_per_case", type=int, default=2,
                    help="sample images per coverage case (P+N+H, P+N, P+H, P, dropped) in summary.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"), help="DOCCI files")
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--out_root", default="runs", help="results and logs (default: runs/ in the current folder)")
    a = ap.parse_args()
    setup_cache_root(a)
    os.makedirs(a.out_root, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(a.out_root, "run_all.log"))
    print(f"\n[run_all] {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(sys.argv)}\n"
          f"[run_all] everything is written under {a.out_root}, {a.cache_dir}, {a.cache_root}")
    a.cache_dir = os.path.abspath(a.cache_dir)
    a.out_root = os.path.abspath(a.out_root)
    if a.docci_dir:
        a.docci_dir = os.path.abspath(a.docci_dir)
    gens = [m for m in a.generators.split(",") if m]
    vlms = [m for m in a.vlms.split(",") if m]
    steps = set(a.steps.split(","))
    grader = None if a.grader.lower() == "none" else a.grader
    captioners = [None] if a.caption2_file else [c for c in a.captioners.split(",") if c]
    versions = [None] if a.caption2_file else [v for v in a.caption_prompts.split(",") if v]
    if set(versions) - set(P.CAPTION_PROMPTS) - {None}:
        ap.error(f"--caption_prompts: choose from {sorted(P.CAPTION_PROMPTS)}")
    if set(captioners) - {"before", "after", "vllm", None}:
        ap.error("--captioners: choose from before, after, vllm")
    step1_dir = os.path.abspath(a.step1_dir or os.path.join(a.out_root, "step1"))
    grpo_dir = os.path.abspath(a.grpo_dir or os.path.join(a.out_root, "grpo"))
    had = trained_tokens(grpo_dir)
    if had is not None and had != a.grpo_max_new_tokens and not a.mock:  # mock trains with 12 tokens on purpose
        new = f"{grpo_dir}_tok{a.grpo_max_new_tokens}"
        print(f"[run_all] {grpo_dir} was trained with captions of up to {had} tokens, not {a.grpo_max_new_tokens}: "
              f"GRPO runs again in {new} (the old folder is left as it is)", flush=True)
        grpo_dir = new
    adapter = os.path.join(grpo_dir, "adapter")
    a.big_root = os.path.abspath(a.big_root or (os.path.join(a.out_root, "_big") if a.mock
                                                else os.path.join(ARGMAX_WORK, "outputs")))
    step1_done = os.path.exists(os.path.join(step1_dir, "results.md"))
    grpo_done = os.path.exists(os.path.join(adapter, "adapter_config.json")) \
        and os.path.exists(os.path.join(grpo_dir, "results.md"))
    a.big_need = (4 if "step1" in steps and not step1_done else 0) + (4 if "grpo" in steps and not grpo_done else 0)
    emb_dir, ckpt_dir = big_dir(a, step1_dir, "emb"), big_dir(a, grpo_dir, "ckpt")
    if a.big_need:
        print(f"[run_all] big files of A / B: Step 1 embedding cache {emb_dir}, GRPO checkpoints {ckpt_dir}", flush=True)
    if not a.mock and a.backend == "vllm":
        need = ([a.captioner_model] if steps & {"step1", "grpo", "caption"} else []) \
            + ([a.caption_model] if "vllm" in captioners and "caption" in steps else []) + gens \
            + (vlms if "bench" in steps else []) + ([grader] if grader and "grade" in steps else [])
        check_disk(a, list(dict.fromkeys(need)))
    R = Runner(a)
    data_args = ["--splits", a.splits, "--n_images", str(a.n_images)]

    # DOCCI once, before everything: descriptions, and the images Caption 2 needs (one pass over the tar)
    recs = None
    if not a.mock:
        recs = C.load_docci(a.cache_dir, a.splits.split(","), a.n_images, a.seed, a.docci_dir)
        if "caption" in steps and not a.caption2_file:
            C.docci_image_paths([r["image_file"] for r in recs], a.cache_dir, a.docci_dir)
    img_dir = os.path.join(a.docci_dir, "images") if a.docci_dir else os.path.join(a.cache_dir, "docci", "images")
    # with every test image on disk, Step 1 / GRPO read their 200 DOCCI test images from there too
    # (same files, same seeded sample) instead of streaming the tar a second time into their own cache
    all_test = recs is not None and a.n_images == 0 and "test" in a.splits.split(",") \
        and "caption" in steps and not a.caption2_file
    docci_for_argmax = a.docci_dir or (os.path.dirname(img_dir) if all_test else None)

    # A, B) the captioner: Step 1 score checks, then GRPO training + its before/after caption evaluation
    argmax_args = ["--model", a.captioner_model, "--seed", str(a.seed), "--gpus", a.gpus,
                   "--gpu_max_mem_mb", str(a.gpu_max_mem_mb), "--gpu_max_util", str(a.gpu_max_util)] \
        + ([] if a.no_gpu_wait else ["--gpu_wait"]) + (["--mock"] if a.mock else []) \
        + (["--docci_dir", docci_for_argmax] if docci_for_argmax else [])
    if "step1" in steps:
        if step1_done:
            print(f"[run_all] Step 1 already done: {step1_dir}", flush=True)
        else:
            R.run("step1_scores", "run.py", ["--out_dir", step1_dir, "--emb_dir", emb_dir] + argmax_args
                  + shlex.split(a.step1_args))
    gate = (report.load_json(os.path.join(step1_dir, "results.json")) or {}).get("gate")
    if gate and not gate.get("passed"):
        print(f"[run_all] note: the Step 1 gate is not passed ({step1_dir}/results.md); GRPO and Caption 2 "
              f"after RL still run, read their results with that in mind", flush=True)
    if "grpo" in steps:
        have = os.path.exists(os.path.join(adapter, "adapter_config.json"))
        if grpo_done:
            print(f"[run_all] GRPO already done: {grpo_dir}", flush=True)
        else:
            resume = ["--eval_only"] if have else ["--resume"] if glob.glob(os.path.join(ckpt_dir, "step_*")) else []
            R.run("grpo_train_eval", "grpo.py", ["--step1_dir", step1_dir, "--out_dir", grpo_dir, "--ckpt_dir", ckpt_dir,
                                                 "--max_new_tokens", str(a.grpo_max_new_tokens)]
                  + resume + argmax_args + shlex.split(a.grpo_args))

    # C) Caption 2, once per (captioner, caption prompt)
    images_file = os.path.join(a.out_root, "docci_images.jsonl")
    C.write_jsonl(images_file, [dict(example_id=r["example_id"], split=r["split"], image_file=r["image_file"])
                                for r in recs or []])
    if "after" in captioners and not a.mock and not os.path.exists(os.path.join(adapter, "adapter_config.json")):
        print(f"[run_all] no GRPO adapter at {adapter}: skipping Caption 2 after RL", flush=True)
        R.failed.append(("caption2 after RL", f"no GRPO adapter at {adapter} (see the grpo step)"))
        captioners = [c for c in captioners if c != "after"]
    # 'after' captions on disk from another adapter (e.g. the old 128-token one) must not be mixed with new ones
    id_file = os.path.join(a.out_root, "caption2_after_adapter.json")
    if "after" in captioners and "caption" in steps and not a.caption2_file:
        old, cur = report.load_json(id_file), adapter_id(adapter)
        made = {d: C.read_shards(os.path.join(d, "caption2.jsonl")) for d in glob.glob(os.path.join(a.out_root, "caption2_after__*"))}
        made = {d: rows for d, rows in made.items() if rows}
        # no record yet (captions from before this check existed): trust the adapter path in their model label
        same = old == cur or (old is None and (a.mock or all(f"({adapter})" in r.get("model", "")
                                                             for rows in made.values() for r in rows)))
        if made and not same:
            raise SystemExit(f"[run_all] {', '.join(map(os.path.basename, made))} in {a.out_root} were made with another "
                             f"GRPO adapter ({(old or {}).get('adapter', 'unknown')}), not {adapter}. Use a new --out_root "
                             "so captions, QA and benchmarks from the two adapters are not mixed.")
        with open(id_file, "w") as f:
            json.dump(cur, f, indent=2)
    cap2 = {}
    for cap in captioners:
        for v in versions:
            if a.caption2_file:
                cap2[(cap, v)] = os.path.abspath(a.caption2_file)
                continue
            name = f"caption2_{cap}__{v}"
            out = os.path.join(a.out_root, name)
            cap2[(cap, v)] = os.path.join(out, "caption2.jsonl")
            if "caption" not in steps:
                continue
            if cap == "vllm" or a.mock:  # mock: fake captions with known mistakes (a tiny random model's are noise)
                R.run(name, "caption.py", ["--model", a.caption_model, "--prompt_version", v, "--max_tokens",
                                           str(a.caption_max_tokens), "--out_dir", out] + data_args + R.common()
                      + (["--mock_captioner", cap] if a.mock and cap != "vllm" else []), parallel=True)
            else:
                R.run(name, "caption_docci.py",
                      ["--which", cap, "--model", a.captioner_model, "--caption_prompt", P.CAPTION_PROMPTS[v],
                       "--prompt_version", v, "--max_new_tokens", str(a.caption_max_tokens),
                       "--batch", str(a.caption_batch), "--images", images_file, "--image_dir", img_dir,
                       "--out_dir", out] + (["--adapter", adapter] if cap == "after" else []), parallel=True)

    # one dataset per (Caption 2, generator): qa_<gen>__<captioner>__<prompt>/
    datasets = []
    for (cap, v), path in cap2.items():
        if not C.read_shards(path):
            print(f"[run_all] no Caption 2 in {path}: no QA from it", flush=True)
            continue
        datasets += [(tag(g) + (f"__{cap}__{v}" if v else ""), g, path) for g in gens]
    qa_dir = {k: os.path.join(a.out_root, f"qa_{k}") for k, _, _ in datasets}

    def gen_args(path):
        return data_args + ["--caption2", path, "--qa_format", a.qa_format, "--reasoning_effort", a.reasoning_effort] \
            + (["--hal_mcq_truth"] if a.hal_mcq_truth else [])

    # D) generate + self-check with one model load per GPU, then build with that check only
    ok_gen = {}
    for k, g, path in datasets:
        if "gen" in steps:
            stage = "gen,verify" if "verify" in steps else "gen"
            ok_gen[k] = R.run(f"gen_{k}", "generate.py",
                              ["--stage", stage, "--model", g, "--verify_tag", tag(g), "--out_dir", qa_dir[k]]
                              + gen_args(path) + R.common(), parallel=True)
        else:
            ok_gen[k] = bool(C.read_shards(os.path.join(qa_dir[k], "gen_raw.jsonl")))
    for k, g, path in datasets:
        if not ok_gen[k]:
            print(f"[run_all] skipping build and benchmark for {k}: generation failed or missing", flush=True)
            continue
        own = bool(C.read_shards(os.path.join(qa_dir[k], "verify", tag(g) + ".jsonl")))
        R.run(f"build_{k}", "generate.py", ["--stage", "build", "--out_dir", qa_dir[k],
                                            "--verifiers", tag(g) if own else "none"] + gen_args(path) + R.common())

    # E) every VLM answers every dataset, with and without the image; one grader load; scores
    runs = []  # (name, run folder, bench args)
    for k, _, _ in datasets:
        qa = os.path.join(qa_dir[k], "qa_final.jsonl")
        if not os.path.exists(qa):
            continue
        if "bench" in steps and not a.mock:  # fetch the images once here, not once per shard
            C.docci_image_paths(sorted({q["image_file"] for q in C.read_jsonl(qa)}), a.cache_dir, a.docci_dir)
        for v in vlms:
            for blind in ([False] if a.skip_blind else [False, True]):
                t = tag(v) + ("__noimage" if blind else "")
                out = os.path.join(a.out_root, f"bench_{k}")
                args = ["--qa", qa, "--model", v, "--tag", t, "--bootstrap", str(a.bootstrap), "--out_dir", out] \
                    + (["--no_image"] if blind else []) + R.common()
                if "bench" in steps:
                    R.run(f"bench_{k}_{t}", "benchmark.py", args + ["--no_score"], parallel=True)
                if C.read_shards(os.path.join(out, t, "predictions.jsonl")):
                    runs.append((f"{k}_{t}", os.path.join(out, t), args))
    if runs and grader and "grade" in steps:
        R.run(f"grade_{tag(grader)}", "grade.py", ["--runs", ",".join(d for _, d, _ in runs), "--model", grader,
                                                   "--reasoning_effort", a.reasoning_effort] + R.model_args(),
              parallel=True)
    for name, _, args in runs:
        R.run(f"score_{name}", "benchmark.py", args + ["--score_only", "--grading", "judge" if grader else "match"])

    write_summary(a, [k for k, _, _ in datasets], R, [("Step 1: score checks", step1_dir),
                                                      ("Step 2: GRPO, captions before / after RL", grpo_dir)])


def write_summary(a, order, R, captioner_dirs=()):
    path = report.write_report(a.out_root, order, a.cache_dir, a.docci_dir, R.failed,
                               a.samples_per_case, a.seed, a.mock, captioner_dirs=captioner_dirs)
    with open(path) as f:
        text = f.read()
    print("\n" + text.split("## Benchmark per coverage case")[0] + f"[run_all] summary with samples: {path}", flush=True)
    if R.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
