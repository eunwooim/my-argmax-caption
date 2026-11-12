"""Helpers for the QA part (generate / benchmark / grade / report / run_all): DOCCI loading, JSONL I/O,
JSON parsing of model output, GPU picking, bootstrap.

DOCCI loading follows data.py (same URLs, same --docci_dir layout, same test-split sampling: --splits test
--n_images 200 --seed 0 picks the same 200 images as grpo.py's DOCCI evaluation).
"""
import base64
import glob
import io
import json
import os
import random
import re
import subprocess
import tarfile
import time
import urllib.request
import zlib

import numpy as np

DOCCI_JSONL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DOCCI_IMAGES_TAR = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
SPLITS = ["train", "test", "qual_dev", "qual_test"]

# Everything big this code writes (DOCCI, caches, temp files) goes under this folder.
# Results and logs go to the current folder (runs/, out_qa/, out_bench/).
# Change it here or with the env var DOCCI_QA_ROOT.
WORK_ROOT = os.environ.get("DOCCI_QA_ROOT", "/mnt/data2/mchung45/docci_qa_work")


# ------------------------------------------------------------------ files
def _download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[download] {url}", flush=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def read_jsonl(path):
    """Rows of a JSONL file. A line cut off by a crash (e.g. disk full) is skipped and redone on resume."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for i, l in enumerate(f):
            if not l.strip():
                continue
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                print(f"[jsonl] skipping broken line {i + 1} of {path}", flush=True)
    return rows


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_newline = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"  # last write was cut off: start a fresh line
    with open(path, "a") as f:
        if needs_newline:
            f.write("\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


CACHE_ROOT = os.path.join(WORK_ROOT, "caches")


def setup_caches(root=CACHE_ROOT, models_in_root=False):
    """Every cache and temp file under root: vLLM/torch compile caches, temp files, .pyc files, pip,
    and model weights (unless HF_HOME is already set, e.g. to a shared folder that has the models).
    Must run before vLLM / torch are imported."""
    root = os.path.abspath(root)
    env = {"HF_HOME": "hf", "VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton",
           "TORCHINDUCTOR_CACHE_DIR": "inductor", "TMPDIR": "tmp", "PYTHONPYCACHEPREFIX": "pycache",
           "PIP_CACHE_DIR": "pip", "XDG_CACHE_HOME": "xdg"}
    for k, sub in env.items():
        if k == "HF_HOME" and os.environ.get("HF_HOME") and not models_in_root:
            continue  # keep a shared model folder that is already set
        os.environ[k] = os.path.join(root, sub)
        os.makedirs(os.environ[k], exist_ok=True)
    if models_in_root:
        os.environ.pop("HF_HUB_CACHE", None)
    print(f"[paths] caches + temp: {root}   model weights: {os.environ['HF_HOME']}", flush=True)


# ------------------------------------------------------------------ shards (data parallel over GPUs)
_SHARD_RE = re.compile(r"\.\d+-of-\d+$")


def add_shard_args(ap):
    ap.add_argument("--shard", default="0/1",
                    help="k/N: this process does part k of N (one process per GPU; run_all.py sets it)")


def shard_kn(a):
    k, n = (int(x) for x in a.shard.split("/"))
    assert 0 <= k < n, f"bad --shard {a.shard}"
    return k, n


def in_shard(a, example_id):
    """Shards split by image, so every question about one image lands in the same shard."""
    k, n = shard_kn(a)
    return n == 1 or zlib.crc32(example_id.encode()) % n == k


def shard_path(a, path):
    """x.jsonl for a single process, x.<k>-of-<N>.jsonl for shard k: shards never write the same file."""
    k, n = shard_kn(a)
    return path if n == 1 else f"{path[:-len('.jsonl')]}.{k}-of-{n}.jsonl"


def read_shards(path):
    """Rows of x.jsonl and of every x.<k>-of-<N>.jsonl, so a rerun with another N still resumes."""
    rows = read_jsonl(path)
    for p in sorted(glob.glob(glob.escape(path[:-len(".jsonl")]) + ".*-of-*.jsonl")):
        rows += read_jsonl(p)
    return rows


def unshard_name(stem):
    return _SHARD_RE.sub("", stem)


# ------------------------------------------------------------------ DOCCI
def load_docci(cache_dir, splits=("test",), n=0, seed=0, docci_dir=None):
    """DOCCI records dict(example_id, split, image_file, description), sorted by id.

    n > 0 keeps a seeded random sample of n records (after the split filter); n = 0 keeps all.
    """
    if docci_dir:
        jl = os.path.join(docci_dir, "docci_descriptions.jsonlines")
    else:
        jl = _download(DOCCI_JSONL, os.path.join(cache_dir, "docci", "docci_descriptions.jsonlines"))
    recs = sorted((r for r in read_jsonl(jl) if r["split"] in splits), key=lambda r: r["example_id"])
    if n and n < len(recs):
        idx = sorted(random.Random(seed).sample(range(len(recs)), n))
        recs = [recs[i] for i in idx]
    return recs


def docci_image_paths(image_files, cache_dir, docci_dir=None):
    """Local paths for the given DOCCI image files, streaming missing ones out of the image tar.

    The tar (3.8 GB) is not in id order, so a first run usually reads most of it once.
    """
    img_dir = os.path.join(docci_dir, "images") if docci_dir else os.path.join(cache_dir, "docci", "images")
    os.makedirs(img_dir, exist_ok=True)
    missing = {f for f in image_files if not os.path.exists(os.path.join(img_dir, f))}
    if missing:
        print(f"[docci] streaming {len(missing)} images out of the image tar (one pass) ...", flush=True)
        with urllib.request.urlopen(DOCCI_IMAGES_TAR) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if m.isfile() and name in missing:
                    with open(os.path.join(img_dir, name), "wb") as g:
                        g.write(tf.extractfile(m).read())
                    missing.discard(name)
                    if not missing:
                        break
        if missing:
            raise SystemExit(f"[docci] {len(missing)} images not found in the tar, e.g. {sorted(missing)[:3]}")
    return {f: os.path.join(img_dir, f) for f in image_files}


def image_data_url(path_or_pil, max_side=1280):
    """JPEG data URL, longest side capped at max_side (DOCCI photos are large)."""
    from PIL import Image
    img = path_or_pil if isinstance(path_or_pil, Image.Image) else Image.open(path_or_pil)
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ model output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_HARMONY_FINAL = "<|channel|>final<|message|>"


def strip_think(text):
    """Reply without its reasoning: Qwen3 <think>...</think>, or gpt-oss (harmony) analysis channel."""
    text = text or ""
    if _HARMONY_FINAL in text:
        text = re.split(r"<\|(?:return|end|call)\|>", text.rsplit(_HARMONY_FINAL, 1)[1])[0]
    elif "<|channel|>" in text:  # reasoning only, no final answer (ran out of tokens)
        return ""
    elif re.match(r"\s*analysis", text) and "assistantfinal" in text:  # special tokens already stripped
        text = text.rsplit("assistantfinal", 1)[1]
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt (chat template), only the close is here
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_json(text):
    """First JSON object in a model reply (after removing <think> and code fences), or None."""
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


# ------------------------------------------------------------------ GPUs (same rule as argmax_caption/gpu.py)
def _query_gpus():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    gpus = {}
    for line in out.strip().splitlines():
        i, used, total, util = [x.strip() for x in line.split(",")]
        gpus[int(i)] = dict(used=int(used), total=int(total), util=int(util) if util.isdigit() else 0)
    return gpus


# GPUs we may never use (other groups' GPUs), whatever --gpus says.
FORBIDDEN_GPUS = {0, 1, 2, 3}


def gpu_candidates(spec, mock=False):
    """'auto' = every GPU nvidia-smi lists; or '4,5,6'. FORBIDDEN_GPUS are always dropped."""
    if spec == "auto":
        ids = list(range(8)) if mock else sorted(_query_gpus())
    else:
        ids = [int(x) for x in spec.split(",") if x.strip()]
    dropped = [i for i in ids if i in FORBIDDEN_GPUS]
    if dropped and spec != "auto":
        print(f"[gpu] ignoring GPU {dropped}: not ours to use", flush=True)
    return [i for i in ids if i not in FORBIDDEN_GPUS]


def add_gpu_args(ap, default="auto", n_gpus=1):
    ap.add_argument("--gpus", default=default,
                    help="candidate GPUs (nvidia-smi indices); auto = every GPU except 0,1,2,3. "
                         "The free ones are picked automatically")
    ap.add_argument("--n_gpus", type=int, default=n_gpus,
                    help="how many free GPUs to use (= vLLM tensor parallel size)")
    ap.add_argument("--gpu_max_mem_mb", type=int, default=1000, help="a GPU counts as free below this memory use")
    ap.add_argument("--gpu_max_util", type=int, default=10, help="... and below this utilization %%")
    ap.add_argument("--gpu_wait", action="store_true", help="wait until enough GPUs are free instead of exiting")
    ap.add_argument("--no_gpu_pick", action="store_true",
                    help="do not touch CUDA_VISIBLE_DEVICES (use it as already set)")


def pick_gpus(a, poll_s=60):
    """Set CUDA_VISIBLE_DEVICES to n_gpus free GPUs among --gpus. Must run before vLLM is imported."""
    if a.no_gpu_pick:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cands = gpu_candidates(a.gpus)
    while True:
        gpus = _query_gpus()
        free = sorted((g["used"], i) for i, g in gpus.items()
                      if i in cands and g["used"] <= a.gpu_max_mem_mb and g["util"] <= a.gpu_max_util)
        print("[gpu] " + " | ".join(f"{i}: {gpus[i]['used']}/{gpus[i]['total']} MiB, {gpus[i]['util']}%"
                                     for i in cands if i in gpus), flush=True)
        if len(free) >= a.n_gpus:
            ids = sorted(i for _, i in free[:a.n_gpus])
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))
            print(f"[gpu] using GPUs {ids}", flush=True)
            return
        if not a.gpu_wait:
            raise SystemExit(f"[gpu] need {a.n_gpus} free GPUs among {cands}, found {len(free)}. "
                             f"Use --gpu_wait, change --gpus, or loosen --gpu_max_mem_mb / --gpu_max_util.")
        print(f"[gpu] not enough free GPUs, checking again in {poll_s}s ...", flush=True)
        time.sleep(poll_s)


# ------------------------------------------------------------------ stats (same as argmax_caption/checks.py)
def bootstrap(outcomes, groups, B=10000, seed=0, chunk=500):
    """Mean of 0/1 outcomes with a 95% interval from B resamples over groups (images)."""
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, n_images=0)
    _, gidx = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    G = gidx.max() + 1
    s = np.bincount(gidx, weights=outcomes, minlength=G)
    c = np.bincount(gidx, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    stats = []
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = rng.integers(0, G, size=(b, G))
        stats.append(s[idx].sum(1) / np.maximum(c[idx].sum(1), 1))
    stats = np.concatenate(stats)
    return dict(mean=float(outcomes.mean()), lo=float(np.percentile(stats, 2.5)),
                hi=float(np.percentile(stats, 97.5)), n=int(len(outcomes)), n_images=int(G))
