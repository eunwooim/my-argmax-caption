"""Pick a free GPU from a candidate list (default: every GPU except 0, 1, 2, 3) using nvidia-smi.

"Free" = memory used <= --gpu_max_mem_mb and utilization <= --gpu_max_util. Among free GPUs the
one with the least memory used wins. With --gpu_wait, poll until one frees up.
Indices are nvidia-smi (PCI bus) indices; run.py sets CUDA_DEVICE_ORDER=PCI_BUS_ID so torch
numbers GPUs the same way.
"""
import os
import subprocess
import time

# GPUs we may never use (other groups' GPUs), whatever --gpus / --device say.
FORBIDDEN = {0, 1, 2, 3}
# what "free" means by default (env, so rl_paper_commands.sh can loosen it for a shared server where a small
# idle process often sits on an otherwise empty 143 GB card)
FREE_MEM_MB = int(os.environ.get("GPU_FREE_MEM_MB", 1000))
FREE_UTIL = int(os.environ.get("GPU_FREE_UTIL", 10))


def query():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    gpus = {}
    for line in out.strip().splitlines():
        i, used, total, util = [x.strip() for x in line.split(",")]
        gpus[int(i)] = dict(used=int(used), total=int(total), util=int(util) if util.isdigit() else 0)
    return gpus


def candidates(spec):
    """'auto' = every GPU nvidia-smi lists; or '4,5,6'. GPUs in FORBIDDEN are always dropped."""
    ids = sorted(query()) if spec == "auto" else [int(x) for x in spec.split(",") if x.strip()]
    dropped = [i for i in ids if i in FORBIDDEN]
    if dropped and spec != "auto":
        print(f"[gpu] ignoring GPU {dropped}: not ours to use", flush=True)
    return [i for i in ids if i not in FORBIDDEN]


def check_device(dev):
    if dev.startswith("cuda:") and not os.environ.get("CUDA_VISIBLE_DEVICES") and int(dev[5:]) in FORBIDDEN:
        raise SystemExit(f"[gpu] {dev} is one of GPUs {sorted(FORBIDDEN)}, which we may not use")
    return dev


def pick(candidates, max_mem_mb=None, max_util=None, wait=False, poll_s=60):
    max_mem_mb = FREE_MEM_MB if max_mem_mb is None else max_mem_mb
    max_util = FREE_UTIL if max_util is None else max_util
    candidates = [i for i in candidates if i not in FORBIDDEN]
    while True:
        gpus = query()
        status = []
        free = []
        for i in candidates:
            g = gpus.get(i)
            if g is None:
                status.append(f"{i}: not found")
                continue
            ok = g["used"] <= max_mem_mb and g["util"] <= max_util
            status.append(f"{i}: {g['used']}/{g['total']} MiB, {g['util']}%{' (free)' if ok else ''}")
            if ok:
                free.append((g["used"], i))
        print("[gpu] " + " | ".join(status), flush=True)
        if free:
            return min(free)[1]
        if not wait:
            raise SystemExit(f"[gpu] no free GPU among {candidates}. "
                             f"Use --gpu_wait to wait, or loosen --gpu_max_mem_mb / --gpu_max_util.")
        print(f"[gpu] none free, checking again in {poll_s}s ...", flush=True)
        time.sleep(poll_s)


def to_torch_device(physical):
    """Map a physical index to a torch device string, honoring CUDA_VISIBLE_DEVICES if set."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        ids = [int(x) for x in vis.split(",") if x.strip().isdigit()]
        if physical not in ids:
            raise SystemExit(f"[gpu] GPU {physical} is hidden by CUDA_VISIBLE_DEVICES={vis}")
        return f"cuda:{ids.index(physical)}"
    return f"cuda:{physical}"


def add_args(ap, default="auto"):
    ap.add_argument("--device", default="auto",
                    help="auto = free GPU among --gpus (else CPU); or e.g. cuda:5, cpu")
    ap.add_argument("--gpus", default=default,
                    help="candidate GPUs (nvidia-smi indices) for --device auto; auto = all but 0,1,2,3")
    ap.add_argument("--gpu_max_mem_mb", type=int, default=1000, help="a GPU counts as free below this memory use")
    ap.add_argument("--gpu_max_util", type=int, default=FREE_UTIL,
                    help="... and below this utilization %% (default: GPU_FREE_UTIL, else 10)")
    ap.add_argument("--gpu_wait", action="store_true", help="wait until one of --gpus is free instead of exiting")


def resolve(a, mock=False):
    """Turn a.device == 'auto' into a concrete device string."""
    import shutil
    if a.device != "auto":
        return check_device(a.device)
    if mock:
        return "cpu"
    if not shutil.which("nvidia-smi"):
        # a 7B model on CPU fills system RAM and can freeze the whole machine; never do it silently
        raise SystemExit("[gpu] nvidia-smi not found, so no GPU here. Run on the GPU server, "
                         "or pass --device cpu explicitly (needs 30+ GB RAM for 7B and is very slow).")
    idx = pick(candidates(a.gpus), a.gpu_max_mem_mb, a.gpu_max_util, a.gpu_wait)
    dev = to_torch_device(idx)
    print(f"[gpu] using GPU {idx} ({dev})")
    return dev
