"""Test-time scaling with each reward as a selector (README "Reward comparison", Use 1 and the combination).

The candidates are sampled inside lmms-eval (prompting_baselines.py tts_sample: N = 16 per image with
Self-Validation's sampling, the exact image bytes saved), so the prompts and images are the benchmark's own. This
script does the rest offline:

  score   every candidate, by every selector:
            self      the model's own S2 score (frozen base model, Step 1 layer and centering means; grpo.Reward)
            selfcert  self-certainty of the base model over the caption's tokens (Kang et al., arXiv 2502.18581):
                      mean over tokens of KL(uniform || p(. | image, prompt, caption so far)), the generation
                      prompt and system prompt of lmms-eval's Qwen2.5-VL
            lpfv      Self-Validation's object confidence (its BoN score f(y): mean LPFV of the caption's MSCOCO
                      objects, 1.0 without objects; prompting_baselines.lpfv / caption_score, the fixed forward)
            longclip  Long-CLIP-L cosine (rewards.LongCLIPReward, the official code + longclip-L.pt)
            jina      jina-clip-v2 cosine (rewards.ExternalReward, the embedder of the external GRPO run)
          an item with several images: the mean of the per-image scores (self / longclip / jina)
  select  for N in 2, 4, 8, 16, each selector keeps one of the first N candidates (the first N of the same 16, so
          the N points are nested): highest score; shortest = fewest words; random = seeded uniform pick.
          N = 1 is candidate 0 for every selector, which is what the sampling cell itself answered and scored.
          Writes <out_dir>/<selector>_n<N>.jsonl ({task, doc_id, caption, index}) for lmms-eval's replay cells.
  vqa     the captioning model answers every CaptionQA question itself, seeing the image (greedy, the options in
          CaptionQA's own shuffled order, without its "cannot answer from the caption" option): the reference
          for the agreement metric (tts_report.py), written as {image_id, question_idx, chosen}.
  args    the --models / --prompting / --prompting_args of lmms_eval_run.py for the replay cells of a selection.

  python tts.py score  --sample_dir lmms_eval_tts/base-sample/caption_long --out tts/base/scores.jsonl
  python tts.py select --sample_dir lmms_eval_tts/base-sample/caption_long --scores tts/base/scores.jsonl --out_dir tts/base/selected
  python tts.py vqa    --sample_dir lmms_eval_tts/base-sample/caption_long --out tts/vqa/base.jsonl [--adapter out_grpo/adapter]
  python tts.py args   --selected_dir tts/base/selected --prefix base --model Qwen/Qwen2.5-VL-7B-Instruct --field models
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import glob
import json
import math
import random
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompting_baselines as PB  # noqa: E402

SELECTORS = ["self", "random", "shortest", "selfcert", "longclip", "lpfv", "jina"]
SCORED = ["self", "selfcert", "lpfv", "longclip", "jina"]  # need a model; random / shortest do not
LMMS_SYSTEM_PROMPT = "You are a helpful assistant."  # lmms-eval Qwen2_5_VL default (system_prompt)
VQA_POST = "Answer with the option's letter from the given choices directly."


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def samples(sample_dir):
    """The sampling cell's items (prompting_trace.jsonl); a resumed cell may hold an item twice: the last wins."""
    path = os.path.join(sample_dir, "prompting_trace.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"[tts] {path} not found: run the sampling cell first (rl_paper_commands.sh tts_sample)")
    recs = {}
    for r in read_jsonl(path):
        recs[PB.item_key(r["task"], r["doc_id"])] = r
    return list(recs.values())


# ------------------------------------------------------------------ models
def resolve_model(name):
    if os.path.exists(name) or name.count("/") != 1:
        return name
    from huggingface_hub import snapshot_download
    caches = [None] + [os.path.join(d, "models", "hub") for d in filter(None, [os.environ.get("LMMS_HF_DIR")])]
    for cache in caches:
        try:
            return snapshot_download(name, cache_dir=cache, local_files_only=True)
        except Exception:
            continue
    return name


def pixel_limits(plan_path):
    if plan_path and os.path.exists(plan_path):
        with open(plan_path) as f:
            m = dict(kv.split("=", 1) for kv in json.load(f)["model_args"].split(","))
        return int(m["min_pixels"]), int(m["max_pixels"])
    return 3136, 802816


class LocalLM:
    """What prompting_baselines.VLM reads from lmms-eval's Qwen2_5_VL, for a model loaded here. Images are given
    as the saved files (the exact bytes the sampling cell's model saw), so _encode_image_data_url passes them on."""

    def __init__(self, model, processor, min_px, max_px, system_prompt=LMMS_SYSTEM_PROMPT):
        self.model, self.processor, self.tokenizer = model, processor, processor.tokenizer
        self.image_min_pixels, self.image_max_pixels, self.system_prompt = min_px, max_px, system_prompt

    @staticmethod
    def _encode_image_data_url(path):
        return path


def load_models(a, need_embedder):
    """One Qwen2.5-VL for everything: S2 reads it through embedder.Embedder (its own processor, the training
    one), the other selectors through prompting_baselines.VLM with lmms-eval's processor settings."""
    import torch

    import embedder as E
    min_px, max_px = pixel_limits(a.plan)
    if a.mock:
        emb = E.load_mock(max_px, device=a.device)
        processor = emb.processor
        lm = LocalLM(emb.model, processor, min_px, max_px, system_prompt=None)
    else:
        from transformers import AutoProcessor
        path = resolve_model(a.model)
        emb = E.load_real(path, max_px, a.device, a.dtype)
        processor = AutoProcessor.from_pretrained(path, max_pixels=max_px, min_pixels=min_px)
        lm = LocalLM(emb.model, processor, min_px, max_px)
    if getattr(a, "adapter", None):
        from peft import PeftModel
        emb.model = lm.model = PeftModel.from_pretrained(emb.model, a.adapter).merge_and_unload().eval()
        print(f"[tts] merged {a.adapter}", flush=True)
    torch.set_grad_enabled(False)
    vlm = PB.VLM(lm)
    vlm.batch = a.batch
    return (emb if need_embedder else None), vlm


# ------------------------------------------------------------------ selectors
class FrozenS2:
    """grpo.Reward without an adapter: cos(img - mu_img, cap - mu_cap) at the Step 1 layer."""

    def __init__(self, emb, a):
        import grpo
        cfg = types.SimpleNamespace(reward_layer=a.reward_layer, step1_dir=a.step1_dir, reward_score="S2",
                                    reward_raw=a.reward_raw, mock=a.mock)
        layer, means, self.name = grpo.load_reward_config(cfg, emb.L)
        self.emb, self.li = emb, layer - 1
        self.mu_i = means["image"][self.li].float() if means else None
        self.mu_c = means["caption"][self.li].float() if means else None

    def _vec(self, image=None, caption=None):
        v = self.emb.embed("S2", image=image, caption=caption)[self.li].float()
        mu = self.mu_i if image is not None else self.mu_c
        return v - mu if mu is not None else v

    def __call__(self, key, image, captions):
        import torch
        iv = self._vec(image=image)
        return [float(torch.nn.functional.cosine_similarity(iv, self._vec(caption=c), dim=0)) for c in captions]


def self_certainty(vlm, images, prompt, captions, batch):
    """Per caption: mean over its tokens (+ <|im_end|>) of KL(uniform || p) = logsumexp - mean(logits) - log V,
    teacher-forced after the generation prompt (right padding; image tokens marked for the 3-D positions)."""
    import torch
    base = vlm.inputs([vlm.messages(images, prompt)], padding_side="right")
    p_ids = base["input_ids"][0]
    P = len(p_ids)
    end = vlm.tok.convert_tokens_to_ids("<|im_end|>")
    toks = [vlm.tok(c, add_special_tokens=False).input_ids + [end] for c in captions]
    pad = vlm.tok.pad_token_id if vlm.tok.pad_token_id is not None else 0
    out = []
    for i in range(0, len(toks), batch):
        chunk = toks[i:i + batch]
        L = P + max(len(t) for t in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=p_ids.dtype, device=vlm.device)
        att = torch.zeros((len(chunk), L), dtype=base["attention_mask"].dtype, device=vlm.device)
        for b, t in enumerate(chunk):
            ids[b, :P + len(t)] = torch.cat([p_ids, torch.tensor(t, dtype=p_ids.dtype, device=vlm.device)])
            att[b, :P + len(t)] = 1
        kw = {}
        if "pixel_values" in base:
            kw["pixel_values"] = base["pixel_values"].repeat(len(chunk), 1)
            kw["image_grid_thw"] = base["image_grid_thw"].repeat(len(chunk), 1)
        if "mm_token_type_ids" in base:
            tt = torch.zeros((len(chunk), L), dtype=base["mm_token_type_ids"].dtype, device=vlm.device)
            tt[:, :P] = base["mm_token_type_ids"][0]
            kw["mm_token_type_ids"] = tt
        keep = L - (P - 1)  # positions P-1 .. L-1: position P-1+k predicts token k of the caption
        logits = vlm.model(input_ids=ids, attention_mask=att, logits_to_keep=keep, **kw).logits
        for b, t in enumerate(chunk):
            lg = logits[b, :len(t)].float()
            sc = torch.logsumexp(lg, -1) - lg.mean(-1) - math.log(lg.shape[-1])
            out.append(float(sc.mean()))
    return out


def per_image_mean(fn, key, pils, captions):
    vals = [fn((key, k), im, captions) for k, im in enumerate(pils)]
    return [sum(v[j] for v in vals) / len(vals) for j in range(len(captions))]


def cmd_score(a):
    from PIL import Image
    recs = samples(a.sample_dir)
    sel = [s for s in a.selectors.split(",") if s]
    unknown = set(sel) - set(SCORED)
    if unknown:
        raise SystemExit(f"[tts] score: {sorted(unknown)} need no scoring or are unknown (scored: {SCORED})")
    done = {}
    if os.path.exists(a.out):
        for r in read_jsonl(a.out):
            done[PB.item_key(r["task"], r["doc_id"])] = r
    if a.num_shards > 1:  # several processes / GPUs, each a fixed part of the items (its own --out file)
        recs = [r for i, r in enumerate(sorted(recs, key=lambda r: PB.item_key(r["task"], r["doc_id"])))
                if i % a.num_shards == a.shard]
    todo = [r for r in recs if PB.item_key(r["task"], r["doc_id"]) not in done
            or any(s not in done[PB.item_key(r["task"], r["doc_id"])] for s in sel)]
    print(f"[tts] score {sel}: {len(recs)} items (shard {a.shard}/{a.num_shards}), {len(recs) - len(todo)} done "
          f"already", flush=True)
    if not todo:
        return
    emb, vlm = load_models(a, need_embedder="self" in sel)
    scorers = {}
    if "self" in sel:
        scorers["self"] = FrozenS2(emb, a)
    if "longclip" in sel:
        import rewards as R
        scorers["longclip"] = R.LongCLIPReward(a.longclip_repo, a.longclip_ckpt, a.device, mock=a.mock or a.mock_scorers)
    if "jina" in sel:
        import rewards as R
        scorers["jina"] = R.ExternalReward(a.jina_model, a.device, mock=a.mock or a.mock_scorers)
    s_lpfv = PB.settings("selfval_bon")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    with open(a.out, "a") as fh:
        for n, r in enumerate(todo, 1):
            key = PB.item_key(r["task"], r["doc_id"])
            row = dict(done.get(key, {}), task=r["task"], doc_id=r["doc_id"],
                       words=[len(c.split()) for c in r["candidates"]])
            cands = r["candidates"]
            pils = [Image.open(p).convert("RGB") for p in r["images"]]
            for s in sel:
                if s in row:
                    continue
                if s == "selfcert":
                    row[s] = self_certainty(vlm, r["images"], r["prompt"], cands, a.sc_batch)
                elif s == "lpfv":
                    words = [PB.chair_words(c) for c in cands]
                    probs = PB.lpfv(vlm, r["images"], [w for ws in words for w in ws], s_lpfv)
                    row[s] = [PB.caption_score(ws, probs) for ws in words]
                else:
                    row[s] = per_image_mean(scorers[s], key, pils, cands)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"[tts] scored {n}/{len(todo)} items ({time.time() - t0:.0f}s)", flush=True)
    for s, sc in scorers.items():
        if hasattr(sc, "truncated_frac"):
            print(f"[tts] {s}: {100 * sc.truncated_frac():.1f}% of the captions were cut to its text limit", flush=True)


# ------------------------------------------------------------------ selection
def pick(selector, row, n, seed, task, doc_id):
    if selector == "random":
        return random.Random(PB.item_seed(seed, task, doc_id) * 31 + n).randrange(n)
    if selector == "shortest":
        w = row["words"][:n]
        return min(range(n), key=lambda i: w[i])  # first minimum
    s = row[selector][:n]
    return max(range(n), key=lambda i: s[i])  # first maximum (np.argmax)


def cmd_select(a):
    recs = samples(a.sample_dir)
    scores = {}
    for pat in [p for p in (a.scores or "").split(",") if p]:  # files or globs (one per scoring shard)
        for path in sorted(glob.glob(pat)):
            for r in read_jsonl(path):
                scores.setdefault(PB.item_key(r["task"], r["doc_id"]), {}).update(r)
    sel = [s for s in a.selectors.split(",") if s]
    ns = [int(x) for x in a.ns.split(",") if x]
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    for s in sel:
        for n in ns:
            rows, picked = [], []
            for r in recs:
                key = PB.item_key(r["task"], r["doc_id"])
                row = scores.get(key, {"words": [len(c.split()) for c in r["candidates"]]})
                if n > len(r["candidates"]):
                    raise SystemExit(f"[tts] {key}: {len(r['candidates'])} candidates, N {n} asked")
                if s in SCORED and (s not in row or len(row[s]) < n):
                    raise SystemExit(f"[tts] {key}: no {s} scores (run tts.py score --selectors {s})")
                i = pick(s, row, n, a.seed, r["task"], r["doc_id"])
                rows.append(dict(task=r["task"], doc_id=r["doc_id"], caption=r["candidates"][i], index=i))
                picked.append(i)
            path = os.path.join(a.out_dir, f"{s}_n{n}.jsonl")
            with open(path, "w") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
            summary[f"{s}_n{n}"] = dict(file=path, items=len(rows),
                                        words=sum(len(x["caption"].split()) for x in rows) / max(len(rows), 1),
                                        first_pick=sum(i == 0 for i in picked) / max(len(picked), 1))
    # N = 1: candidate 0, the same for every selector (its own replay cell, <prefix>-first-n1)
    with open(os.path.join(a.out_dir, "first_n1.jsonl"), "w") as f:
        f.writelines(json.dumps(dict(task=r["task"], doc_id=r["doc_id"], caption=r["candidates"][0], index=0),
                                ensure_ascii=False) + "\n" for r in recs)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[tts] select: {len(sel)} selectors x N {ns} over {len(recs)} items -> {a.out_dir}", flush=True)


# ------------------------------------------------------------------ the model's own answers (agreement)
def captionqa_ext(tasks_dir):
    sys.path.insert(0, tasks_dir)
    import captionqa_ext as X  # the task's own question / option / permutation logic
    return X


def cmd_vqa(a):
    X = captionqa_ext(a.tasks_dir)
    recs = [r for r in samples(a.sample_dir) if r["task"].startswith("captionqa_x")]
    if not recs:
        raise SystemExit(f"[tts] vqa: no CaptionQA items in {a.sample_dir}")
    _, vlm = load_models(a, need_embedder=False)
    items, meta = [], []
    for r in recs:
        for q_idx, question, choices, _, opts, perm, _ in X.prepared(r["doc"]):
            order = [i for i in perm if i < len(choices)]  # CaptionQA's order, its "cannot answer" option left out
            lines = "\n".join(f"{X.U.LETTER_ALPH[k]}. {choices[i]}" for k, i in enumerate(order))
            items.append((r["images"], f"{question}\n{lines}\n{VQA_POST}"))
            meta.append((r["doc"].get("id", "unknown"), q_idx, order))
    t0 = time.time()
    answers = []
    for i in range(0, len(items), 256):
        answers += vlm.generate(items[i:i + 256], a.max_new_tokens)
        print(f"[tts] vqa {len(answers)}/{len(items)} questions ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for (image_id, q_idx, order), ans in zip(meta, answers):
            letter = X.U.extract_letter(ans, len(order))
            k = X.U.LETTER_ALPH.find(letter) if letter else -1
            chosen = order[k] if 0 <= k < len(order) else None
            f.write(json.dumps(dict(image_id=image_id, question_idx=q_idx, chosen=chosen, answer=ans)) + "\n")
    print(f"[tts] vqa: {len(answers)} answers -> {a.out}", flush=True)


# ------------------------------------------------------------------ replay cells for lmms_eval_run.py
def cmd_args(a):
    files = sorted(p for p in glob.glob(os.path.join(a.selected_dir, "*_n*.jsonl")))
    if a.selectors:
        keep = set(a.selectors.split(","))
        files = [p for p in files if os.path.basename(p).rsplit("_n", 1)[0] in keep]
    if a.ns:
        keep_n = set(a.ns.split(","))
        files = [p for p in files if os.path.basename(p)[:-6].rsplit("_n", 1)[1] in keep_n]
    if not files:
        raise SystemExit(f"[tts] args: no <selector>_n<N>.jsonl in {a.selected_dir} (run tts.py select)")
    names = {f"{a.prefix}-{os.path.basename(p)[:-6].replace('_n', '-n')}": os.path.abspath(p) for p in files}
    if a.field == "models":
        print(",".join(f"{n}={a.model}" for n in names))
    elif a.field == "prompting":
        print(",".join(f"{n}=replay" for n in names))
    else:
        print(json.dumps({n: {"file": p} for n, p in names.items()}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="hub id (local copy used) or folder")
        p.add_argument("--plan", default=os.path.join(HERE, "lmms_eval_out", "plan.json"),
                       help="plan.json whose min/max pixels lmms-eval used (the processor here uses the same)")
        p.add_argument("--dtype", default="bfloat16")
        import gpu
        gpu.add_args(p)
        p.add_argument("--batch", type=int, default=16, help="sequences per generate / LPFV forward")
        p.add_argument("--mock", action="store_true", help="tiny random Qwen2.5-VL on CPU (smoke test)")

    p = sub.add_parser("score")
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selectors", default="self,selfcert,lpfv,longclip,jina")
    p.add_argument("--step1_dir", default=os.path.join(HERE, "out_7b"))
    p.add_argument("--reward_layer", type=int, default=None)
    p.add_argument("--reward_raw", action="store_true")
    p.add_argument("--sc_batch", type=int, default=4, help="captions per self-certainty forward (full-vocab logits)")
    p.add_argument("--longclip_repo", default=os.environ.get("LONGCLIP_REPO", "/mnt/data2/mchung45/Long-CLIP"))
    p.add_argument("--longclip_ckpt", default=os.environ.get("LONGCLIP_CKPT", "/mnt/data2/mchung45/Long-CLIP/checkpoints/longclip-L.pt"))
    p.add_argument("--jina_model", default="jinaai/jina-clip-v2")
    p.add_argument("--mock_scorers", action="store_true", help="tiny random CLIP for longclip / jina (smoke test)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    model_args(p)
    p = sub.add_parser("select")
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--scores", default=None, help="score files or globs, comma list (all shards)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--selectors", default=",".join(SELECTORS))
    p.add_argument("--ns", default="2,4,8,16")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("vqa")
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None, help="GRPO LoRA adapter merged into --model (that model's own answers)")
    p.add_argument("--tasks_dir", default="/mnt/data2/mchung45/lmms_eval_work/custom_tasks")
    p.add_argument("--max_new_tokens", type=int, default=8)
    model_args(p)
    p = sub.add_parser("args")
    p.add_argument("--selected_dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--selectors", default="")
    p.add_argument("--ns", default="", help="only these N (comma list), e.g. 4 for a quick run")
    p.add_argument("--field", choices=["models", "prompting", "prompting_args"], required=True)
    a = ap.parse_args()
    if a.cmd in ("score", "vqa"):
        import gpu
        a.device = gpu.resolve(a, mock=a.mock)
    {"score": cmd_score, "select": cmd_select, "vqa": cmd_vqa, "args": cmd_args}[a.cmd](a)


if __name__ == "__main__":
    main()
