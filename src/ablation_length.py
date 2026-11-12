"""Ablation: is the GRPO reward gain just "shorter caption -> higher cosine"?

GRPO made captions shorter and the reward (S2-centered@L24) went up. If shortness alone raised the score,
cutting the *before* caption down to the *after* caption's length (no new model, no RL) would score as
high as *after*. This script scores, per image, with the exact training reward (LoRA off):

  before            the base model's caption (as in grpo.py's eval)
  after             the GRPO model's caption
  before_lenmatch   before, cut at a sentence end once it reaches after's word count
  before_words      before, cut to exactly after's word count (mid-sentence allowed)
  before_k1/k2/k3   the first 1 / 2 / 3 sentences of before
  short_*           the base model asked for a short caption (--short_prompts), LoRA off

Reads <grpo_dir>/eval_coco_val.jsonl and eval_docci_ood.jsonl (written by grpo.py), so the images and
before/after captions are the same ones grpo.py evaluated. The hypothesis "length is not the cause"
holds if before_lenmatch / short_* are not above before, and after stays above all of them.

  python ablation_length.py --step1_dir out_7b --grpo_dir out_grpo_tok4096
  python ablation_length.py --mock --grpo_dir runs_mock/grpo     # CPU smoke test
Writes <grpo_dir>/ablation_length/{results.md, results.html, per_image_<set>.jsonl}.
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import json
import re
import time
from types import SimpleNamespace

import numpy as np

import checks as C
import gpu
import grpo as G
import report as RP

SETS = ["coco_val", "docci_ood"]
SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(t):
    return [s for s in SENT.split(t.strip()) if s]


def first_k(t, k):
    return " ".join(sentences(t)[:k])


def len_match(t, n_words):
    """Whole sentences of t until at least n_words words (never longer than t)."""
    out, n = [], 0
    for s in sentences(t):
        if n >= n_words:
            break
        out.append(s)
        n += len(s.split())
    return " ".join(out)


def cut_words(t, n_words):
    return " ".join(t.split()[:n_words])


def words(t):
    return len(t.split())


def grpo_args(a):
    """The grpo.py settings the reward / generation helpers read."""
    ga = SimpleNamespace(**vars(a))
    tj = os.path.join(a.grpo_dir, "adapter", "grpo_train.json")
    trained = {}
    if os.path.exists(os.path.join(a.grpo_dir, "args.json")):
        with open(os.path.join(a.grpo_dir, "args.json")) as f:
            trained = json.load(f)
    for k in ("model", "max_pixels", "n_eval_coco", "n_eval_docci", "n_center", "reward_score", "reward_layer",
              "reward_raw", "caption_prompt", "temperature", "lora_r"):
        if getattr(ga, k, None) is None and k in trained:
            setattr(ga, k, trained[k])
    if os.path.exists(tj) and ga.max_new_tokens is None:
        with open(tj) as f:
            ga.max_new_tokens = json.load(f).get("max_new_tokens")
    ga.max_new_tokens = ga.max_new_tokens or trained.get("max_new_tokens", 4096)
    ga.step1_dir = ga.step1_dir or trained.get("step1_dir")
    ga.out_dir, ga.eval_only, ga.n_train_pool = a.grpo_dir, True, 0
    has_adapter = os.path.exists(os.path.join(a.grpo_dir, "adapter"))
    ga.adapter = None
    ga.eval_only = has_adapter  # build() then loads <grpo_dir>/adapter; it is never switched on here
    return ga


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grpo_dir", required=True, help="grpo.py out_dir (has eval_*.jsonl)")
    ap.add_argument("--step1_dir", default=None, help="Step 1 out_dir (means/, selection.json); default from args.json")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="for short_* captions; default as GRPO")
    gpu.add_args(ap)
    ap.add_argument("--short_prompts", default="Describe this image in one sentence.|Briefly describe this image.",
                    help="'|'-separated prompts for base-model short captions; empty = skip generation")
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N images per set (0 = all)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.environ.get("ARGMAX_CACHE_DIR", "/mnt/data2/mchung45/argmax_caption_work/cache"))
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="default <grpo_dir>/ablation_length")
    for k in ("n_eval_coco", "n_eval_docci", "n_center", "reward_layer", "lora_r", "temperature"):
        ap.add_argument(f"--{k}", type=float if k == "temperature" else int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reward_score", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reward_raw", action="store_true", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--caption_prompt", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.out_dir = a.out_dir or os.path.join(a.grpo_dir, "ablation_length")
    os.makedirs(a.out_dir, exist_ok=True)

    rows = {}
    for s in SETS:
        p = os.path.join(a.grpo_dir, f"eval_{s}.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                rows[s] = [json.loads(l) for l in f if l.strip()]
            if a.limit:
                rows[s] = rows[s][:a.limit]
    if not rows:
        raise SystemExit(f"no eval_coco_val.jsonl / eval_docci_ood.jsonl in {a.grpo_dir}: run grpo.py first")

    ga = grpo_args(a)
    if ga.mock:
        ga.max_new_tokens = min(ga.max_new_tokens, 12)
        ga.bootstrap = min(ga.bootstrap, 500)
    ga.device = gpu.resolve(ga, mock=ga.mock)
    emb, policy = G.build(ga)
    layer, means, reward_name = G.load_reward_config(ga, emb.L)
    reward = G.Reward(emb, policy, ga.reward_score, layer, means)
    data = G.Images(ga)
    items = {tuple(it["key"]): it for it in data.eval_coco + data.eval_docci}
    shorts = [p.strip() for p in a.short_prompts.split("|") if p.strip()]

    md = ["# Ablation: is the reward gain just shorter captions?", "",
          f"Reward **{reward_name}** (the GRPO training reward; LoRA off), source `{a.grpo_dir}`. "
          "Each row scores a different caption of the same images. *before* / *after* are grpo.py's base and "
          "GRPO captions; the *before_* rows are the base caption cut shorter, with no model change; *short_* rows "
          "are the base model asked for a short caption. Differences are per image, 95% bootstrap over images.", "",
          "If length alone explained the gain, the shortened rows would reach *after*. They don't if "
          "`after − row` stays above 0.", ""]
    summary = {}
    for s, rs in rows.items():
        rs = [r for r in rs if tuple(r["key"]) in items]
        ids = [str(r["key"]) for r in rs]
        caps = {"before": [r["before"] for r in rs], "after": [r["after"] for r in rs]}
        caps["before_lenmatch"] = [len_match(r["before"], words(r["after"])) for r in rs]
        caps["before_words"] = [cut_words(r["before"], words(r["after"])) for r in rs]
        for k in (1, 2, 3):
            caps[f"before_k{k}"] = [first_k(r["before"], k) for r in rs]
        old = ga.caption_prompt
        for i, p in enumerate(shorts):
            ga.caption_prompt = p
            imgs = [items[tuple(r["key"])] for r in rs]
            caps[f"short_{i + 1}"] = G.captions_for(ga, policy, emb.processor, data, imgs, use_adapter=False,
                                                    label=f"{s} short_{i + 1}")
        ga.caption_prompt = old
        sc = {v: [] for v in caps}
        t0 = time.time()
        for j, r in enumerate(rs):
            it = items[tuple(r["key"])]
            names = list(caps)
            vals = reward(it["key"], data.image(it), [caps[v][j] or "." for v in names])
            for v, x in zip(names, vals):
                sc[v].append(x)
            if (j + 1) % 50 == 0 or j + 1 == len(rs):
                print(f"  [{s}] scored {j + 1}/{len(rs)}  {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(a.out_dir, f"per_image_{s}.jsonl"), "w") as f:
            for j, r in enumerate(rs):
                f.write(json.dumps(dict(key=r["key"], **{v: caps[v][j] for v in caps},
                                        **{f"{v}_score": sc[v][j] for v in caps}), ensure_ascii=False) + "\n")
        before, after = np.array(sc["before"]), np.array(sc["after"])
        stored = [r.get("reward_before") for r in rs]
        drift = float(np.nanmax(np.abs(before - np.array(stored, float)))) if None not in stored else float("nan")
        bw = np.array([words(t) for t in caps["before"]], float)
        corr = float(np.corrcoef(bw, before)[0, 1]) if len(rs) > 2 and bw.std() > 0 else float("nan")
        md += [f"## {s} (n={len(rs)})", "",
               "| caption | words | score | row − before | after − row | after beats row |",
               "|---|---|---|---|---|---|"]
        summary[s] = {}
        for v in caps:
            x = np.array(sc[v])
            d1 = C.bootstrap(x - before, ids, ga.bootstrap)
            d2 = C.bootstrap(after - x, ids, ga.bootstrap)
            w = C.bootstrap(after > x, ids, ga.bootstrap)
            nw = float(np.mean([words(t) for t in caps[v]]))
            label = v if not v.startswith("short_") else f"{v} (`{shorts[int(v[6:]) - 1]}`)"
            md.append(f"| {label} | {nw:.1f} | {x.mean():.4f} | {d1['mean']:+.4f} [{d1['lo']:+.4f}, {d1['hi']:+.4f}] | "
                      + ("—" if v == "after" else f"{d2['mean']:+.4f} [{d2['lo']:+.4f}, {d2['hi']:+.4f}]")
                      + " | " + ("—" if v == "after" else f"{100 * w['mean']:.1f}%") + " |")
            summary[s][v] = dict(words=nw, score=float(x.mean()), minus_before=d1, after_minus=d2, after_wins=w)
        md += ["", f"Within the *before* captions, Pearson r(words, score) = {corr:+.3f} "
                   "(negative would mean shorter base captions already score higher).",
               f"Re-scored *before* vs grpo.py's stored reward_before: max |difference| {drift:.2e}.", ""]
        summary[s]["_corr_words_score_before"] = corr
        # verdict
        worst = max((v for v in caps if v != "after"), key=lambda v: summary[s][v]["score"])
        lo = summary[s][worst]["after_minus"]["lo"]
        md += [f"**Verdict ({s}):** the best non-RL caption is `{worst}` "
               f"({summary[s][worst]['score']:.4f} vs after {summary[s]['after']['score']:.4f}). "
               + ("After stays significantly above every shortened caption, so shortness alone does not explain "
                  "the gain." if lo > 0 else
                  "After is **not** significantly above it, so length may explain part or all of the gain."), ""]
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(dict(reward=reward_name, results=summary), f, indent=2)
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    RP.write_html("\n".join(md), os.path.join(a.out_dir, "results.html"), a.out_dir, title="Length ablation")
    print(f"wrote {a.out_dir}/results.md and results.html")


if __name__ == "__main__":
    main()
