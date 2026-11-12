"""Step 2: GRPO on COCO with the FreeRet (S2) score as reward, single GPU.

  BeforeRlModelCaption = Qwen2.5-VL(image, caption prompt)              (LoRA off)
  Argmax*              = Qwen2.5-VL + GRPO LoRA(image, caption prompt)  (LoRA on)
  Target result        = FreeRet(Argmax*, image) vs FreeRet(BeforeRlModelCaption, image)
                         on COCO val2017 (in-distribution) and DOCCI test (OOD)

One copy of the model on one GPU does three jobs:
  policy     = model + LoRA (trainable)
  reference  = model with LoRA disabled (KL term)
  reward     = model with LoRA disabled, read exactly as Step 1's S2 (frozen FreeRet embedding)

--reward picks the reward (the comparison runs of the RL paper; rewards.py):
  self      the model's own S2 score above (default)
  external  cos(image, caption) from a separate embedding model (--external_model)
  length    -|words - target| / target, target = the self run's mean rollout length at the same step
            (--length_target_log <self out_dir>/train_log.jsonl): length-matched control, no image signal
  random    uniform noise per caption (rewards.RandomReward): the no-signal control
  selfcert  the policy's own self-certainty over its sampled tokens (Intuitor, arXiv 2505.19590; SelfCertainty)
  longclip  cos(image, caption) from Long-CLIP-L (rewards.LongCLIPReward; --longclip_repo / --longclip_ckpt)
With any reward but self, the self score (S2, Step 1 layer and means) is still reported as a held-out scorer.

Examples
  python grpo.py --mock --out_dir out_grpo_mock                       # CPU smoke test, no downloads
  python grpo.py --step1_dir out_7b --out_dir out_grpo                # real run, free GPU (never 0, 1, 2, 3)
  python grpo.py --step1_dir out_7b --out_dir out_grpo --eval_only     # re-run the before/after evaluation
  python grpo.py --step1_dir out_7b --out_dir out_grpo_ext --reward external --external_model jinaai/jina-clip-v2
  python grpo.py --step1_dir out_7b --out_dir out_grpo_len --reward length --length_target_log out_grpo/train_log.jsonl
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch GPU numbers == nvidia-smi numbers

import argparse
import glob
import hashlib
import json
import random
import time

import numpy as np
import torch

import checks as C
import data as D
import gpu
import prompts as P

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


# ------------------------------------------------------------------ reward (frozen FreeRet / S2)
class Reward:
    """cos(img - mu_img, cap - mu_cap) read like Step 1, at one layer, with LoRA disabled."""

    def __init__(self, emb, policy, score, layer, means):
        self.emb, self.policy, self.score, self.li = emb, policy, score, layer - 1
        self.name = f"{score}-{'centered' if means else 'raw'}@L{layer}"
        self.mu_i = means["image"][self.li].float() if means else None
        self.mu_c = means["caption"][self.li].float() if means else None
        self.img_cache = {}

    def _vec(self, image=None, caption=None):
        with self.policy.disable_adapter():
            v = self.emb.embed(self.score, image=image, caption=caption)[self.li].float()
        mu = self.mu_i if image is not None else self.mu_c
        return v - mu if mu is not None else v

    def __call__(self, key, image, captions):
        if key not in self.img_cache:
            self.img_cache[key] = self._vec(image=image)
        iv = self.img_cache[key]
        return [float(torch.nn.functional.cosine_similarity(iv, self._vec(caption=c), dim=0)) for c in captions]


def load_reward_config(a, L):
    """Defaults from Step 1: S2, centered, layer from selection.json (24 in the 7B run)."""
    means, layer = None, a.reward_layer
    sel_path = os.path.join(a.step1_dir, "selection.json") if a.step1_dir else None
    if layer is None and sel_path and os.path.exists(sel_path):
        with open(sel_path) as f:
            sel = json.load(f)
        if sel["score"] == a.reward_score:
            layer = sel["layer"]
    layer = layer or L
    if not a.reward_raw:
        mp = os.path.join(a.step1_dir or "", "means", f"{a.reward_score}.pt")
        if not os.path.exists(mp):
            if a.mock:
                print("[reward] mock: no Step 1 means, using raw score")
            else:
                raise SystemExit(f"[reward] {mp} not found. Pass --step1_dir <Step 1 out_dir> "
                                 f"(it holds means/{a.reward_score}.pt), or --reward_raw.")
        else:
            means = torch.load(mp)
    name = f"{a.reward_score}-{'centered' if means else 'raw'}@L{layer}"
    print(f"[reward] {name}")
    return layer, means, name


# ------------------------------------------------------------------ model
def build(a):
    import embedder as E
    from peft import LoraConfig, PeftModel, get_peft_model

    if a.mock:
        emb = E.load_mock(a.max_pixels, device=a.device)
    else:
        emb = E.load_real(a.model, a.max_pixels, a.device, a.dtype)
    base = emb.model
    adapter = a.adapter or (os.path.join(a.out_dir, "adapter") if a.eval_only else None)
    if adapter:
        policy = PeftModel.from_pretrained(base, adapter, is_trainable=not a.eval_only)
        print(f"[model] loaded adapter {adapter}")
    else:
        cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules=LORA_TARGETS)
        policy = get_peft_model(base, cfg)
    policy.print_trainable_parameters()
    return emb, policy


def caption_text(a):
    return P.qwen_chat(P.IMAGE_TOKEN + a.caption_prompt)


def special_ids(tok):
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
    return eos, pad


def completion_mask(comp, eos):
    """1 up to and including the first <|im_end|>, 0 after."""
    is_eos = comp == eos
    after = (is_eos.cumsum(1) - is_eos.long()) > 0
    return (~after).long()


def decode(tok, comp, mask):
    return [tok.decode(c[m.bool()], skip_special_tokens=True).strip() for c, m in zip(comp, mask)]


@torch.no_grad()
def generate(a, policy, processor, images, n, sample, use_adapter=True):
    """Captions for a list of images; n per image. Returns (inputs, prompt_len, completions, mask, texts)."""
    tok = processor.tokenizer
    eos, pad = special_ids(tok)
    inp = processor(text=[caption_text(a)] * len(images), images=images, padding=True,
                    return_tensors="pt").to(a.device)
    # never emit vision placeholders: they would break the log-prob forward (image token count)
    banned = [[tok.convert_tokens_to_ids(t)] for t in
              ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")]
    # repetition_penalty is set explicitly: Qwen's generation_config ships 1.05, and with it the sampled
    # distribution would differ from the one the loss scores (on-policy needs plain softmax sampling)
    kw = dict(max_new_tokens=a.max_new_tokens, eos_token_id=eos, pad_token_id=pad, num_return_sequences=n,
              bad_words_ids=banned, repetition_penalty=1.0)
    # remove_invalid_values: a NaN / inf logit (bf16 overflow) would otherwise crash multinomial sampling
    kw.update(dict(do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0, remove_invalid_values=True)
              if sample else dict(do_sample=False))
    policy.eval()
    if use_adapter:
        out = policy.generate(**inp, **kw)
    else:
        with policy.disable_adapter():
            out = policy.generate(**inp, **kw)
    plen = inp["input_ids"].shape[1]
    comp = out[:, plen:]
    mask = completion_mask(comp, eos)
    return inp, plen, out, mask, decode(tok, comp, mask)


def token_logprobs(model, inp, seqs, plen, mask, n, with_sc=False):
    """Log-prob of each completion token. Sequences are ordered [img0 x n, img1 x n, ...].
    with_sc: also each token's self-certainty, KL(uniform || p) = -log V - mean_j log p_j (no gradient)."""
    thw = inp["image_grid_thw"]
    chunks = inp["pixel_values"].split(thw.prod(-1).tolist())
    pv = torch.cat([c for c in chunks for _ in range(n)])
    attn = torch.cat([inp["attention_mask"].repeat_interleave(n, 0), mask], 1)
    T = seqs.shape[1] - plen
    extra = {}
    if "mm_token_type_ids" in inp:  # without it transformers falls back to 1-D positions for the image tokens
        tt = inp["mm_token_type_ids"].repeat_interleave(n, 0)
        extra["mm_token_type_ids"] = torch.cat([tt, torch.zeros_like(seqs[:, plen:], dtype=tt.dtype)], 1)
    logits = model(input_ids=seqs, attention_mask=attn, pixel_values=pv,
                   image_grid_thw=thw.repeat_interleave(n, 0), use_cache=False,
                   logits_to_keep=T + 1, **extra).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, seqs[:, plen:].unsqueeze(-1)).squeeze(-1)
    if not with_sc:
        return lp
    with torch.no_grad():
        lg = logits.detach()
        sc = torch.logsumexp(lg, -1) - lg.mean(-1) - float(np.log(lg.shape[-1]))
    return lp, sc


class SelfCertainty:
    """Intuitor's reward (arXiv 2505.19590): self-certainty (Kang et al., arXiv 2502.18581), the mean over the
    caption's tokens of KL(uniform || p_policy(. | image, prompt, caption so far)), V = the logits' vocabulary.
    In training it is read from the policy's own forward over the sampled tokens (grpo_micro, `fused`), as in
    Intuitor (online, the current policy). As a scorer (before / after evaluation) it is a teacher-forced forward
    of the caption's tokens + <|im_end|> with the adapter on."""
    fused = True

    def __init__(self, a, policy, processor):
        self.a, self.policy, self.processor = a, policy, processor
        self.name = "self-certainty(policy)"

    @torch.no_grad()
    def __call__(self, key, image, captions):
        a, tok = self.a, self.processor.tokenizer
        eos, pad = special_ids(tok)
        inp = self.processor(text=[caption_text(a)], images=[image], return_tensors="pt").to(a.device)
        plen = inp["input_ids"].shape[1]
        comps = [tok(c, add_special_tokens=False).input_ids[:a.max_new_tokens] + [eos] for c in captions]
        comp = torch.full((len(comps), max(map(len, comps))), pad, dtype=inp["input_ids"].dtype, device=a.device)
        for i, c in enumerate(comps):
            comp[i, :len(c)] = torch.tensor(c, dtype=comp.dtype, device=a.device)
        mask = completion_mask(comp, eos)
        self.policy.eval()
        _, sc = token_logprobs(self.policy, inp, torch.cat([inp["input_ids"].repeat(len(comps), 1), comp], 1),
                               plen, mask, len(comps), with_sc=True)
        m = mask.float()
        return ((sc * m).sum(1) / m.sum(1).clamp(min=1)).tolist()


# ------------------------------------------------------------------ data
class Images:
    """COCO train pool (minus the Step 1 centering images), COCO val eval set, DOCCI eval set."""

    def __init__(self, a):
        self.a = a
        if a.mock:
            import mock_data as M
            self.train = [dict(key=("coco", i), file=f"synth_{i}.png") for i in range(100, 100 + a.n_train_pool)]
            self.eval_coco = [dict(key=("coco", i), file=f"synth_{i}.png") for i in range(5000, 5000 + a.n_eval_coco)]
            self.eval_docci = [dict(key=("docci", r["example_id"]), pil=r["image"]) for r in M.docci(a.n_eval_docci)]
            return
        self.train = []
        if not a.eval_only:  # the train2017 caption file is large; skip it when only evaluating
            imgs, caps = D.load_coco_captions(a.cache_dir, "train2017")
            imgs = [im for im in imgs if im["id"] in caps]
            excl = set(random.Random(0).sample(range(len(imgs)), 2 * a.n_center))  # Step 1 centering images
            pool = [im for i, im in enumerate(imgs) if i not in excl]
            random.Random(a.seed + 1).shuffle(pool)
            self.train = [dict(key=("coco", im["id"]), info=im, split="train2017") for im in pool[:a.n_train_pool]]
        vimgs, self.val_caps = D.load_coco_captions(a.cache_dir, "val2017")
        vsel = random.Random(a.seed).sample(vimgs, a.n_eval_coco)
        self.eval_coco = [dict(key=("coco", im["id"]), info=im, split="val2017") for im in vsel]
        self.eval_docci = []
        if a.n_eval_docci:
            for r in D.load_docci(a.cache_dir, n=a.n_eval_docci, seed=a.seed, docci_dir=a.docci_dir):
                self.eval_docci.append(dict(key=("docci", r["example_id"]), pil=r["image"]))

    def image(self, item):
        if "pil" in item:
            return item["pil"]
        if self.a.mock:
            import mock_data as M
            return M.image(item["file"])
        return D.coco_image(item["info"], self.a.cache_dir, item["split"])


# ------------------------------------------------------------------ one GRPO step
def grpo_micro(a, policy, processor, reward, items, data, accum):
    """Rollouts, rewards, advantages and one backward pass for a few images."""
    G = a.rollouts
    images = [data.image(it) for it in items]
    inp, plen, seqs, mask, texts = generate(a, policy, processor, images, G, sample=True)
    fused = getattr(reward, "fused", False)  # self-certainty: read from the policy forward below
    if not fused:
        r = torch.tensor([v for i, it in enumerate(items)
                          for v in reward(it["key"], images[i], texts[i * G:(i + 1) * G])])

    with torch.no_grad(), policy.disable_adapter():
        ref = token_logprobs(policy, inp, seqs, plen, mask, G)
    policy.train()
    m = mask.float()
    if fused:
        lp, sc = token_logprobs(policy, inp, seqs, plen, mask, G, with_sc=True)
        r = ((sc * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
    else:
        lp = token_logprobs(policy, inp, seqs, plen, mask, G)
    if not torch.isfinite(r).all():  # a NaN reward makes every advantage NaN, so every update gets skipped
        bad = [t[:80] for t, v in zip(texts, r.tolist()) if not np.isfinite(v)]
        raise SystemExit(f"[reward] {reward.name} returned {len(bad)}/{len(r)} non-finite values, e.g. {bad[:2]}")
    rg = r.view(len(items), G)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1).to(a.device)
    d = (ref - lp).clamp(-20, 20)  # exp of a very unlikely token's gap overflows to inf -> NaN gradients
    kl = torch.exp(d) - d - 1  # k3 estimator, >= 0
    per_tok = -torch.exp(lp - lp.detach()) * adv[:, None] + a.beta * kl
    if a.loss_norm == "seq":   # original GRPO: mean over each caption's tokens (length-biased)
        per_seq = (per_tok * m).sum(1) / m.sum(1).clamp(min=1)
    else:                      # Dr. GRPO: sum over tokens / a fixed constant, no bias toward short captions
        per_seq = (per_tok * m).sum(1) / a.loss_const
    loss = per_seq.mean() * len(items) / accum
    loss.backward()
    return dict(reward=r.tolist(), group_std=rg.std(1).mean().item(),
                kl=((kl * m).sum(1) / m.sum(1).clamp(min=1)).mean().item(),
                words=[len(t.split()) for t in texts], loss=loss.item() * accum / len(items),
                sample=texts[0])


def train(a, emb, policy, processor, reward, data):
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    start, log_path = 0, os.path.join(a.out_dir, "train_log.jsonl")
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "step_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    if a.resume and ckpts:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(policy, load_file(os.path.join(ckpts[-1], "adapter_model.safetensors")))
        # map_location: the optimizer state was saved on the GPU of the earlier run; loading it there would open a
        # CUDA context on that GPU (another one than this run's, maybe someone else's)
        opt.load_state_dict(torch.load(os.path.join(ckpts[-1], "optimizer.pt"), map_location=a.device))
        start = int(ckpts[-1].rsplit("_", 1)[1])
        print(f"[train] resumed from {ckpts[-1]}")
    if a.device.startswith("cuda"):
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    micro = a.batch_images if a.micro_images == "auto" else int(a.micro_images)
    B = a.batch_images
    n_skipped = 0
    for step in range(start, a.steps):
        batch = [data.train[(step * B + j) % len(data.train)] for j in range(B)]
        if hasattr(reward, "set_step"):  # length control: target length of this step
            reward.set_step(step + 1)
        torch.manual_seed(a.seed * 100003 + step)  # same rollouts for a step after --resume
        t0 = time.time()
        while True:
            try:
                opt.zero_grad(set_to_none=True)
                stats = [grpo_micro(a, policy, processor, reward, batch[j:j + micro], data, B)
                         for j in range(0, B, micro)]
                break
            except torch.cuda.OutOfMemoryError:
                if a.micro_images != "auto" or micro == 1:
                    raise
                micro = max(1, micro // 2)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[train] OOM -> {micro} image(s) x {a.rollouts} rollouts per forward", flush=True)
        gn = torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
        skipped = not torch.isfinite(gn)
        if not skipped:  # never let one bad batch put NaN into the LoRA weights
            backup = [p.detach().clone() for p in params]
            opt.step()
            if not all(torch.isfinite(p).all() for p in params):
                with torch.no_grad():
                    for p, b in zip(params, backup):
                        p.copy_(b)
                opt.state.clear()
                skipped = True
            del backup
        if skipped:
            n_skipped += 1
            print(f"[train] step {step + 1}: non-finite gradients / weights, update skipped ({n_skipped} so far)", flush=True)
            if n_skipped > max(10, a.steps // 10):
                raise SystemExit("[train] too many non-finite steps; lower --lr or --max_new_tokens")
        rw = [v for s in stats for v in s["reward"]]
        words = [v for s in stats for v in s["words"]]
        rec = dict(step=step + 1, reward=float(np.mean(rw)), group_std=float(np.mean([s["group_std"] for s in stats])),
                   kl=float(np.mean([s["kl"] for s in stats])), loss=float(np.mean([s["loss"] for s in stats])),
                   words=float(np.mean(words)), grad_norm=float(gn), skipped=skipped, micro_images=micro,
                   vram_peak_gb=round(torch.cuda.max_memory_allocated(a.device) / 2**30, 1) if a.device.startswith("cuda") else 0.0,
                   sec=round(time.time() - t0, 1), sample=stats[0]["sample"])
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[step {step + 1}/{a.steps}] reward {rec['reward']:.4f}  kl {rec['kl']:.4f}  "
              f"words {rec['words']:.1f}  vram {rec['vram_peak_gb']}GB  {rec['sec']}s | {rec['sample'][:80]!r}", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            d = os.path.join(a.ckpt_dir, f"step_{step + 1}")
            policy.save_pretrained(d)
            torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 != a.steps:
            q = evaluate_set(a, policy, processor, reward, data, data.eval_coco[:a.n_eval_quick], "coco_quick")
            with open(os.path.join(a.out_dir, "quick_eval.jsonl"), "a") as f:
                f.write(json.dumps(dict(step=step + 1, **q)) + "\n")
    policy.save_pretrained(os.path.join(a.out_dir, "adapter"))
    with open(os.path.join(a.out_dir, "adapter", "grpo_train.json"), "w") as f:  # what this adapter was trained with
        json.dump(dict(max_new_tokens=a.max_new_tokens, caption_prompt=a.caption_prompt, steps=a.steps,
                       loss_norm=a.loss_norm, loss_const=a.loss_const, model=a.model, reward=a.reward,
                       reward_name=reward.name, max_pixels=a.max_pixels), f, indent=2)
    if a.device.startswith("cuda"):
        policy.gradient_checkpointing_disable()


# ------------------------------------------------------------------ before / after evaluation
def captions_for(a, policy, processor, data, items, use_adapter, label=""):
    out, t0 = [], time.time()
    for j in range(0, len(items), a.eval_batch):
        chunk = items[j:j + a.eval_batch]
        *_, texts = generate(a, policy, processor, [data.image(it) for it in chunk], 1,
                             sample=False, use_adapter=use_adapter)
        out += texts
        print(f"  [{label}] captions {len(out)}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return out


def evaluate_set(a, policy, processor, reward, data, items, name, write=False, extra=None):
    """Before/after captions on `items`, scored by the training reward and by extra (held-out) scorers."""
    tag = hashlib.md5(f"{a.caption_prompt}|{a.max_new_tokens}".encode()).hexdigest()[:8]  # cache depends on these
    base_path = os.path.join(a.out_dir, f"captions_before_{name}_{tag}.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            before = json.load(f)
    else:
        before = captions_for(a, policy, processor, data, items, use_adapter=False, label=f"{name} before")
        with open(base_path, "w") as f:
            json.dump(before, f)
    after = captions_for(a, policy, processor, data, items, use_adapter=True, label=f"{name} after")
    ids = [str(it["key"]) for it in items]
    scorers = {"reward": reward, **(extra or {})}
    scores, per_item = {}, {}
    for sname, sc in scorers.items():
        rb, ra, t0 = [], [], time.time()
        for k, (it, cb, ca) in enumerate(zip(items, before, after)):
            b, c = sc(it["key"], data.image(it), [cb, ca])
            rb.append(b)
            ra.append(c)
            if (k + 1) % 50 == 0 or k + 1 == len(items):
                print(f"  [{name}] scoring with {sc.name}: {k + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        diff = np.array(ra) - np.array(rb)
        scores[sname] = dict(name=sc.name, before=float(np.mean(rb)), after=float(np.mean(ra)),
                             diff=C.bootstrap(diff, ids, a.bootstrap), win=C.bootstrap(diff > 0, ids, a.bootstrap))
        per_item[sname] = (rb, ra)
    res = dict(n=len(items), scores=scores,
               words_before=float(np.mean([len(t.split()) for t in before])),
               words_after=float(np.mean([len(t.split()) for t in after])))
    if write:
        with open(os.path.join(a.out_dir, f"eval_{name}.jsonl"), "w") as f:
            for i, (it, cb, ca) in enumerate(zip(items, before, after)):
                rec = dict(key=list(it["key"]), before=cb, after=ca)
                for sname, (rb, ra) in per_item.items():
                    rec[f"{sname}_before"], rec[f"{sname}_after"] = rb[i], ra[i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return res


def write_report(a, reward_name, res):
    lines = [f"# Step 2: GRPO with {a.reward} reward", "",
             f"Reward: **{reward_name}** (frozen). Policy: {a.model if not a.mock else 'mock'} + LoRA r={a.lora_r}. "
             f"Caption prompt: `{a.caption_prompt}`, up to {a.max_new_tokens} new tokens. Rollouts {a.rollouts}, batch {a.batch_images} images, "
             f"{a.steps} steps, lr {a.lr}, KL beta {a.beta}, loss norm {a.loss_norm}.", "",
             "Before = BeforeRlModelCaption (base model). After = Argmax* (GRPO-trained). Greedy decoding for both. "
             "Intervals: 95% bootstrap over images.", "",
             "`reward` is the score that was optimised, so it should go up by construction. The other rows were "
             "never trained on: if they do not move with it, the gain is specific to the reward (reward hacking), "
             "not better captions.", "",
             "| set | scorer | before | after | after - before | after wins |",
             "|---|---|---|---|---|---|"]
    for name, r in res.items():
        for sname, sc in r["scores"].items():
            d, w = sc["diff"], sc["win"]
            label = f"{sname} ({sc['name']})" if sname != "reward" else f"reward ({sc['name']})"
            lines.append(f"| {name} (n={r['n']}) | {label} | {sc['before']:.4f} | {sc['after']:.4f} | "
                         f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] | "
                         f"{100 * w['mean']:.1f}% [{100 * w['lo']:.1f}, {100 * w['hi']:.1f}] |")
    lines += ["", "| set | words before | words after |", "|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {name} | {r['words_before']:.1f} | {r['words_after']:.1f} |")
    lines += ["", "Step 1 found this score does not rise with more true detail, so shrinking captions are a warning sign.", ""]
    for name in res:
        p = os.path.join(a.out_dir, f"eval_{name}.jsonl")
        if os.path.exists(p):
            lines += [f"## Examples ({name})", ""]
            with open(p) as f:
                for line in list(f)[:5]:
                    e = json.loads(line)
                    lines += [f"- **{e['key']}** before ({e['reward_before']:.3f}): {e['before']}",
                              f"  after ({e['reward_after']:.3f}): {e['after']}"]
            lines.append("")
    with open(os.path.join(a.out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_training(a):
    p = os.path.join(a.out_dir, "train_log.jsonl")
    if not os.path.exists(p):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(p) as f:
        log = [json.loads(l) for l in f]
    steps = [r["step"] for r in log]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, key, title in zip(axes, ["reward", "words", "kl"],
                              ["mean reward (rollouts)", "caption length (words)", "KL to base"]):
        ax.plot(steps, [r[key] for r in log], color="#2a78d6", lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "train_curves.png"), dpi=150)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="tiny random Qwen2.5-VL + synthetic images, CPU, no downloads")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    gpu.add_args(ap)
    # reward
    ap.add_argument("--step1_dir", default=None, help="Step 1 out_dir (means/ and selection.json)")
    ap.add_argument("--reward", default="self", choices=["self", "external", "length", "random", "selfcert", "longclip"],
                    help="self: own S2 score (below); external: --external_model; length: length-matched control; "
                         "random: uniform noise; selfcert: the policy's self-certainty (Intuitor); longclip: Long-CLIP")
    ap.add_argument("--longclip_repo", default=os.environ.get("LONGCLIP_REPO", "/mnt/data2/mchung45/Long-CLIP"),
                    help="clone of github.com/beichenzbc/Long-CLIP (its model/ package)")
    ap.add_argument("--longclip_ckpt", default=os.environ.get("LONGCLIP_CKPT", "/mnt/data2/mchung45/Long-CLIP/checkpoints/longclip-L.pt"),
                    help="Long-CLIP checkpoint (BeichenZhang/LongCLIP-L longclip-L.pt)")
    ap.add_argument("--external_model", default="jinaai/jina-clip-v2",
                    help="image-text embedder for --reward external / --extra_external (CLIP / SigLIP family, "
                         "or remote code with encode_image / encode_text)")
    ap.add_argument("--extra_external", action="store_true",
                    help="also report --external_model as a held-out scorer when it is not the reward")
    ap.add_argument("--length_target_log", default=None, help="--reward length: the self run's train_log.jsonl")
    ap.add_argument("--length_target_words", type=float, default=None,
                    help="--reward length without a log: one fixed target length in words")
    ap.add_argument("--reward_score", default="S2", choices=["S0", "S1", "S2", "S3"])
    ap.add_argument("--reward_layer", type=int, default=None, help="default: Step 1 selection (24), else last")
    ap.add_argument("--reward_raw", action="store_true", help="no centering")
    # policy / GRPO
    ap.add_argument("--caption_prompt", default="Describe this image.")
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="caption length limit for training rollouts and evaluation (a cap, not a target)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--rollouts", type=int, default=4, help="captions sampled per image (GRPO group size)")
    ap.add_argument("--batch_images", type=int, default=8, help="images per optimizer step")
    ap.add_argument("--micro_images", default="auto",
                    help="images per forward pass; 'auto' starts at batch_images and halves on OOM")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL weight to the base model")
    ap.add_argument("--loss_norm", default="const", choices=["const", "seq"],
                    help="const = Dr. GRPO (sum / --loss_const, no length bias); seq = original GRPO per-caption mean")
    ap.add_argument("--loss_const", type=float, default=128,
                    help="Dr. GRPO's constant divisor. 128 = the old max_new_tokens, so the loss scale (and how often "
                         "--grad_clip bites) stays what it was when the length limit was 128")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--adapter", default=None, help="LoRA adapter to load (eval_only default: out_dir/adapter)")
    # data / eval
    ap.add_argument("--n_train_pool", type=int, default=20000)
    ap.add_argument("--n_center", type=int, default=10000, help="Step 1 centering size (those images are excluded)")
    ap.add_argument("--n_eval_coco", type=int, default=500)
    ap.add_argument("--n_eval_docci", type=int, default=200)
    ap.add_argument("--n_eval_quick", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--extra_scores", default="S1,S3,S0",
                    help="held-out scorers for the final eval (last layer, centered if Step 1 means exist)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.environ.get("ARGMAX_CACHE_DIR", "/mnt/data2/mchung45/argmax_caption_work/cache"),
                    help="downloads (SugarCrepe, DOCCI; COCO only if no local copy). Results go to --out_dir")
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--out_dir", default="out_grpo")
    ap.add_argument("--ckpt_dir", default=None, help="checkpoints, ~0.5 GB each (default <out_dir>/ckpt)")
    a = ap.parse_args()
    if a.mock:
        for k, v in dict(n_train_pool=16, n_eval_coco=4, n_eval_docci=3, n_eval_quick=2, max_new_tokens=12,
                         steps=min(a.steps, 3), batch_images=min(a.batch_images, 2), eval_every=2,
                         save_every=2, bootstrap=min(a.bootstrap, 500), eval_batch=2).items():
            if getattr(a, k) == ap.get_default(k):
                setattr(a, k, v)
    a.ckpt_dir = a.ckpt_dir or os.path.join(a.out_dir, "ckpt")
    a.device = gpu.resolve(a, mock=a.mock)
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    with open(os.path.join(a.out_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    emb, policy = build(a)
    processor = emb.processor
    layer, means, _ = load_reward_config(a, emb.L)
    self_score = Reward(emb, policy, a.reward_score, layer, means)
    extra = {}  # held-out scorers: never optimised, used only to check the reward gain is real
    if a.reward == "self":
        reward = self_score
    else:
        import rewards as R
        if a.reward == "external":
            reward = R.ExternalReward(a.external_model, a.device, mock=a.mock)
        elif a.reward == "length":
            reward = R.LengthReward(a.length_target_log, a.length_target_words)
        elif a.reward == "random":
            reward = R.RandomReward(a.seed)
        elif a.reward == "longclip":
            reward = R.LongCLIPReward(a.longclip_repo, a.longclip_ckpt, a.device, mock=a.mock)
        else:
            reward = SelfCertainty(a, policy, processor)
        extra["self"] = self_score  # the self score is what the paper's hypothesis is about: report it in every run
    if a.extra_external and a.reward != "external":
        import rewards as R
        extra["external"] = R.ExternalReward(a.external_model, a.device, mock=a.mock)
    reward_name = reward.name
    print(f"[reward] optimised: {reward_name}", flush=True)
    for sc in [x for x in a.extra_scores.split(",") if x]:
        mp = os.path.join(a.step1_dir or "", "means", f"{sc}.pt")
        m = torch.load(mp) if os.path.exists(mp) else None
        extra[sc] = Reward(emb, policy, sc, emb.L, m)
    print("[data] loading COCO / DOCCI lists (first run downloads COCO annotations 241 MB and DOCCI images 3.8 GB)",
          flush=True)
    data = Images(a)
    print(f"train pool {len(data.train)} | eval COCO {len(data.eval_coco)} | eval DOCCI {len(data.eval_docci)}")

    if not a.eval_only:
        train(a, emb, policy, processor, reward, data)
        plot_training(a)
    res = {"coco_val": evaluate_set(a, policy, processor, reward, data, data.eval_coco, "coco_val", True, extra)}
    if data.eval_docci:
        res["docci_ood"] = evaluate_set(a, policy, processor, reward, data, data.eval_docci, "docci_ood", True, extra)
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(dict(reward=reward_name, reward_kind=a.reward, results=res), f, indent=2)
    for sc in [reward, *extra.values()]:
        if hasattr(sc, "truncated_frac"):
            print(f"[reward] {sc.name}: {100 * sc.truncated_frac():.1f}% of the captions it scored were cut to its text limit")
    write_report(a, reward_name, res)
    for name, r in res.items():
        for sname, sc in r["scores"].items():
            print(f"[eval] {name:9s} {sname:6s} {sc['name']:16s} before {sc['before']:.4f} -> after {sc['after']:.4f} "
                  f"(after wins {100 * sc['win']['mean']:.1f}%)")
        print(f"[eval] {name:9s} words {r['words_before']:.1f} -> {r['words_after']:.1f}")
    print(f"wrote {a.out_dir}/results.md")


if __name__ == "__main__":
    main()
