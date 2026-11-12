"""Step 1: pick the image-caption score before GRPO.

Stages (run in this order; --stage all runs the three):
  means     doc 3  image / caption averages over COCO train2017, per score, every layer
  select    doc 6  best of S0-S3 x {raw, centered} on SugarCrepe dev, then a layer sweep for it
  evaluate  doc 5  checks 1-4 on held-out data with bootstrap intervals, tables, curve, gate

Examples
  python run.py --mock --out_dir out_mock                       # CPU smoke test, no downloads
  python run.py --model Qwen/Qwen2.5-VL-7B-Instruct --out_dir out_7b
  python run.py --model Qwen/Qwen2.5-VL-3B-Instruct --out_dir out_3b   # T4 fallback (doc 1)
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch GPU numbers == nvidia-smi numbers

import torch

import checks as C
import data as D
import prompts as P

VARIANTS = [(s, c) for s in ["S0", "S1", "S2", "S3"] for c in (False, True)]


def vname(score, centered, layer=None, L=None):
    name = f"{score}-{'centered' if centered else 'raw'}"
    return name if layer is None or layer == L else f"{name}@L{layer}"


# ------------------------------------------------------------------ embedding cache
class Cache:
    """Embeddings of one score for a set of keys, keeping only the layers in `keep` (0-based)."""

    def __init__(self, emb, score, image_fn, keep, path=None):
        self.emb, self.score, self.image_fn = emb, score, image_fn
        self.keep = sorted(keep)
        self.row = {li: r for r, li in enumerate(self.keep)}
        self.data, self.path = {}, path
        if path and os.path.exists(path):
            saved = torch.load(path)
            if saved["keep"] == self.keep:
                self.data = saved["data"]

    def vec(self, key, li):
        return self.data[key][self.row[li]].float()

    def ensure(self, img_keys, captions, label=""):
        todo = [k for k in sorted(img_keys, key=str) if k not in self.data]
        todo += [("cap", c) for c in sorted(captions) if ("cap", c) not in self.data]
        t0 = time.time()
        for i, k in enumerate(todo):
            if k[0] == "cap":
                e = self.emb.embed(self.score, caption=k[1])
            else:
                e = self.emb.embed(self.score, image=self.image_fn(k))
            self.data[k] = e[self.keep].float()  # fp32: fp16 rounding can create ties between near-equal cosines
            if (i + 1) % 200 == 0:
                print(f"  [{self.score} {label}] {i + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
                self.save()
        self.save()

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            torch.save(dict(keep=self.keep, data=self.data), self.path)


# ------------------------------------------------------------------ setup
def load_embedder(a):
    import embedder as E
    if a.mock:
        return E.load_mock(a.max_pixels, device=a.device)
    return E.load_real(a.model, a.max_pixels, a.device, a.dtype)


def load_sugarcrepe(a):
    if a.mock:
        import mock_data as M
        items = M.sugarcrepe()
    else:
        items = D.load_sugarcrepe(a.cache_dir)
    dev, test = D.split_by_image(items, dev_frac=0.1, seed=a.seed)
    if a.limit_images:  # quick partial runs only; the doc's numbers use the full split
        keep = sorted({it["filename"] for it in test})[:a.limit_images]
        test = [it for it in test if it["filename"] in set(keep)]
    print(f"SugarCrepe: {len(items)} items on {len({i['filename'] for i in items})} images; "
          f"dev {len(dev)} items / test {len(test)} items")
    return dev, test


def make_image_fn(a, docci=None):
    if a.mock:
        import mock_data as M
        sc_img = M.image
    else:
        sc_img = lambda fn: D.coco_val_image(fn, a.cache_dir)
    docci_by_id = {r["example_id"]: r["image"] for r in (docci or [])}

    def fn(key):
        if key[0] == "img":
            return sc_img(key[1])
        if key[0] == "noise":
            return D.noise_image(sc_img(key[1]).size, seed=key[2])
        if key[0] == "docci":
            return docci_by_id[key[1]]
        raise KeyError(key)
    return fn


def means_path(a, score):
    return os.path.join(a.out_dir, "means", f"{score}.pt")


def load_means(a, score):
    p = means_path(a, score)
    return torch.load(p) if os.path.exists(p) else None


# ------------------------------------------------------------------ stage: means
def stage_means(a, emb):
    if a.mock:
        import mock_data as M
        imgs, caps = M.centering_sets()
        img_fn = lambda info: M.image(info["file_name"])
    else:
        imgs, caps = D.centering_sets(a.cache_dir, n=a.n_center, seed=a.seed, ann_path=a.coco_ann)
        img_fn = lambda info: D.coco_train_image(info, a.cache_dir)
    for score in a.scores:
        if os.path.exists(means_path(a, score)):
            print(f"[means] {score}: exists, skipping")
            continue
        t0 = time.time()
        s_img = sum(emb.embed(score, image=img_fn(info)).double() for info in _progress(imgs, f"{score} img"))
        s_cap = sum(emb.embed(score, caption=c).double() for c in _progress(caps, f"{score} cap"))
        out = dict(image=(s_img / len(imgs)).float(), caption=(s_cap / len(caps)).float(),
                   n_images=len(imgs), n_captions=len(caps))
        os.makedirs(os.path.dirname(means_path(a, score)), exist_ok=True)
        torch.save(out, means_path(a, score))
        print(f"[means] {score}: done in {time.time() - t0:.0f}s")


def _progress(seq, label, every=500):
    for i, x in enumerate(seq):
        if i and i % every == 0:
            print(f"  [{label}] {i}/{len(seq)}", flush=True)
        yield x


# ------------------------------------------------------------------ stage: select
def stage_select(a, emb):
    dev, _ = load_sugarcrepe(a)
    image_fn = make_image_fn(a)
    L = emb.L
    imgs, caps = C.sugarcrepe_keys(dev)
    table, caches = {}, {}
    for score in a.scores:
        cache = Cache(emb, score, image_fn, keep=range(L),
                      path=os.path.join(a.emb_dir, f"dev_{score}.pt"))
        cache.ensure(imgs, caps, "dev")
        caches[score] = cache
        means = load_means(a, score)
        for centered in (False, True):
            if centered and means is None:
                print(f"[select] no means for {score}; skipping centered (run --stage means)")
                continue
            sim = C.Scorer(cache, L, means if centered else None)
            table[vname(score, centered)] = C.check1_sugarcrepe(sim, dev, a.bootstrap)["overall"]
    best = max(table, key=lambda k: table[k]["mean"])
    score, centered = best.split("-")[0], best.endswith("centered")
    means = load_means(a, score) if centered else None
    sweep = {}
    for layer in range(1, L + 1):
        sim = C.Scorer(caches[score], layer, means)
        sweep[layer] = C.check1_sugarcrepe(sim, dev, a.bootstrap)["overall"]
    best_layer = max(sweep, key=lambda l: sweep[l]["mean"])
    sel = dict(score=score, centered=centered, layer=best_layer, num_layers=L,
               dev_variants=table, dev_layer_sweep=sweep)
    with open(os.path.join(a.out_dir, "selection.json"), "w") as f:
        json.dump(sel, f, indent=2)
    print("\n[select] SugarCrepe dev, last layer:")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {k:14s} {_fmt(v)}")
    print(f"[select] chosen: {best}; layer sweep best = {best_layer}/{L} ({_fmt(sweep[best_layer])})")
    return sel


# ------------------------------------------------------------------ stage: evaluate
def stage_evaluate(a, emb):
    sel_path = os.path.join(a.out_dir, "selection.json")
    if not os.path.exists(sel_path):
        raise SystemExit("run --stage select first (selection.json missing)")
    with open(sel_path) as f:
        sel = json.load(f)
    L = emb.L
    _, test = load_sugarcrepe(a)
    if a.mock:
        import mock_data as M
        docci = M.docci()
    else:
        docci = D.load_docci(a.cache_dir, n=a.n_docci, seed=a.seed, docci_dir=a.docci_dir)
    image_fn = make_image_fn(a, docci)
    dplan, nplan = C.docci_plan(docci), C.noise_plan(test, seed=a.seed)
    print(f"DOCCI: {len(dplan)} images, {sum(len(p['prefixes']) for p in dplan)} prefixes, "
          f"{sum(p['swapped'] is not None for p in dplan)} with a color swap")
    k1, k2, k3 = C.sugarcrepe_keys(test), C.docci_keys(dplan), C.noise_keys(nplan)
    imgs, caps = k1[0] | k2[0] | k3[0], k1[1] | k2[1] | k3[1]

    results, curves = {}, {}
    chosen_name = vname(sel["score"], sel["centered"], sel["layer"], L)
    for score in a.scores:
        keep = {L - 1} | ({sel["layer"] - 1} if score == sel["score"] else set())
        cache = Cache(emb, score, image_fn, keep=keep,
                      path=os.path.join(a.emb_dir, f"test_{score}.pt"))
        cache.ensure(imgs, caps, "test")
        means = load_means(a, score)
        rows = [(False, L), (True, L)]
        if score == sel["score"] and sel["layer"] != L:
            rows.append((sel["centered"], sel["layer"]))
        for centered, layer in rows:
            if centered and means is None:
                continue
            name = vname(score, centered, layer, L)
            sim = C.Scorer(cache, layer, means if centered else None)
            r = dict(check1=C.check1_sugarcrepe(sim, test, a.bootstrap))
            r["check2"], curves[name] = C.check2_docci_more(sim, dplan, a.bootstrap)
            r["check3"] = C.check3_docci_wrong(sim, dplan, a.bootstrap)
            r["check4"] = C.check4_noise(sim, nplan, a.bootstrap)
            results[name] = r
            print(f"[eval] {name:16s} SugarCrepe {_fmt(r['check1']['overall'])}")

    g = C.gate(results[chosen_name], a.gate) if chosen_name in results else None
    out = dict(selection={k: sel[k] for k in ("score", "centered", "layer", "num_layers")},
               chosen=chosen_name, gate=g, results=results, docci_curves=curves)
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    write_report(a, out, sel)
    plot_curves(a, curves, chosen_name)
    print(f"\nwrote {a.out_dir}/results.md, results.json, docci_curve.png")
    if g:
        print("GATE:", "PASSED" if g["passed"] else "NOT PASSED", f"({chosen_name})")


# ------------------------------------------------------------------ report
def _fmt(r):
    return f"{100 * r['mean']:.1f} [{100 * r['lo']:.1f}, {100 * r['hi']:.1f}]"


def write_report(a, out, sel):
    res = out["results"]
    types = D.SUGARCREPE_TYPES
    lines = ["# Step 1 results", "",
             f"Chosen on SugarCrepe dev: **{out['chosen']}** (layer {sel['layer']} of {sel['num_layers']}).",
             "", "Accuracy % with 95% bootstrap interval over images (10,000 resamples by default). Chance is 50%.", "",
             "| score | SugarCrepe overall | DOCCI more detail (rises) | DOCCI wrong color | real > noise | real > wrong image | real > both |",
             "|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        c4 = r["check4"]
        lines.append(f"| {name} | {_fmt(r['check1']['overall'])} | {_fmt(r['check2'])} | {_fmt(r['check3'])} | "
                     f"{_fmt(c4['vs_noise'])} | {_fmt(c4['vs_wrong'])} | {_fmt(c4['vs_both'])} |")
    lines += ["", "## SugarCrepe by type", "", "| score | " + " | ".join(types) + " |",
              "|---|" + "---|" * len(types)]
    for name, r in res.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(r["check1"][t]) if t in r["check1"] else "-" for t in types) + " |")
    lines += ["", "## SugarCrepe dev (selection)", "", "| variant | dev accuracy |", "|---|---|"]
    for k, v in sorted(sel["dev_variants"].items(), key=lambda kv: -kv[1]["mean"]):
        lines.append(f"| {k} | {_fmt(v)} |")
    lines += ["", "Layer sweep for the chosen score on dev:", "", "| layer | dev accuracy |", "|---|---|"]
    for l, v in sel["dev_layer_sweep"].items():
        lines.append(f"| {l} | {_fmt(v)} |")
    g = out["gate"]
    lines += ["", f"## Gate for GRPO ({out['chosen']})", ""]
    if g:
        for cond, ok in g["conditions"].items():
            lines.append(f"- {'PASS' if ok else 'FAIL'}: {cond}")
        lines += ["", f"**{'Passed. GRPO may start with this score.' if g['passed'] else 'Not passed. Do not start GRPO; next step is per-word matching (doc section 8).'}**"]
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_curves(a, curves, chosen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw_c, cen_c = "#2a78d6", "#eb6834"
    scores = [s for s in a.scores if any(n.startswith(s + "-") for n in curves)]
    fig, axes = plt.subplots(1, len(scores) + 1, figsize=(4 * (len(scores) + 1), 3.4), squeeze=False)
    for ax, s in zip(axes[0], scores):
        for centered, col in ((False, raw_c), (True, cen_c)):
            n = f"{s}-{'centered' if centered else 'raw'}"
            if n in curves:
                ks = [p["k"] for p in curves[n]]
                ax.plot(ks, [p["mean"] for p in curves[n]], color=col, lw=2, marker="o", ms=4,
                        label="centered" if centered else "raw")
        ax.set_title(s)
        ax.legend(frameon=False, fontsize=8)
    ax = axes[0][-1]
    ks = [p["k"] for p in curves[chosen]]
    ax.plot(ks, [p["mean"] for p in curves[chosen]], color=raw_c, lw=2, marker="o", ms=4)
    ax.set_title(f"chosen: {chosen}")
    for ax in axes[0]:
        ax.set_xlabel("first k sentences")
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("mean cos(image, caption)")
    fig.suptitle("DOCCI: average score by number of true sentences", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "docci_curve.png"), dpi=150)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "means", "select", "evaluate"])
    ap.add_argument("--mock", action="store_true", help="tiny random Qwen2.5-VL + synthetic data, CPU, no downloads")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--device", default="auto",
                    help="auto = free GPU among --gpus (else CPU); or e.g. cuda:5, cpu")
    ap.add_argument("--gpus", default="auto",
                    help="candidate GPUs (nvidia-smi indices) for --device auto; auto = all but 1,2,3")
    ap.add_argument("--gpu_max_mem_mb", type=int, default=int(os.environ.get("GPU_FREE_MEM_MB", 1000)), help="a GPU counts as free below this memory use")
    ap.add_argument("--gpu_max_util", type=int, default=int(os.environ.get("GPU_FREE_UTIL", 10)), help="... and below this utilization %%")
    ap.add_argument("--gpu_wait", action="store_true", help="wait until one of --gpus is free instead of exiting")
    ap.add_argument("--dtype", default="bfloat16", help="use float16 on a T4 (no bf16)")
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--scores", nargs="+", default=["S0", "S1", "S2", "S3"], choices=["S0", "S1", "S2", "S3"])
    ap.add_argument("--n_center", type=int, default=10000)
    ap.add_argument("--n_docci", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--gate", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_images", type=int, default=0, help="cap held-out SugarCrepe images (quick runs only)")
    ap.add_argument("--cache_dir", default=os.environ.get("ARGMAX_CACHE_DIR", "/mnt/data2/mchung45/argmax_caption_work/cache"),
                    help="downloads (SugarCrepe, DOCCI; COCO only if no local copy). Results go to --out_dir")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--emb_dir", default=None, help="embedding cache, a few GB (default <out_dir>/emb)")
    ap.add_argument("--coco_ann", default=None, help="path to captions_train2017.json (else downloaded)")
    ap.add_argument("--docci_dir", default=None, help="folder with docci_descriptions.jsonlines and images/")
    a = ap.parse_args()
    a.emb_dir = a.emb_dir or os.path.join(a.out_dir, "emb")
    if a.device.startswith("cuda:"):
        import gpu
        gpu.check_device(a.device)
    if a.mock and a.device == "auto":
        a.device = "cpu"
    elif a.device == "auto":
        import shutil
        if shutil.which("nvidia-smi"):
            import gpu
            idx = gpu.pick(gpu.candidates(a.gpus), a.gpu_max_mem_mb, a.gpu_max_util, a.gpu_wait)
            a.device = gpu.to_torch_device(idx)
            print(f"[gpu] using GPU {idx} ({a.device})")
        else:
            a.device = "cpu"
            print("[gpu] nvidia-smi not found; running on CPU")
    os.makedirs(a.out_dir, exist_ok=True)
    emb = load_embedder(a)
    print(f"model ready: {emb.L} decoder layers; S3 averages the last {emb.lens_n} tokens")
    if a.stage in ("all", "means"):
        stage_means(a, emb)
    if a.stage in ("all", "select"):
        stage_select(a, emb)
    if a.stage in ("all", "evaluate"):
        stage_evaluate(a, emb)


if __name__ == "__main__":
    main()
