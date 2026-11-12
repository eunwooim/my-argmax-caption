"""Turn a GRPO LoRA checkpoint into a plain Hugging Face model folder (LoRA merged in) for lmms-eval.

grpo.py saves PEFT adapters (<ckpt_dir>/step_N/, <out_dir>/adapter/). lmms-eval's qwen2_5_vl model loads
only `pretrained=<folder>`, so each checkpoint we evaluate is merged into the base weights here and saved
with its processor. ~16 GB per 7B folder (bf16): export only the checkpoints in the evaluation plan.

Checks, all on the same image and the caption prompt the adapter was trained with. Logits are compared
teacher-forced over the whole greedy caption of base + LoRA (every position, not one):
  1. merged model vs base + LoRA (PEFT). In bf16 the merge rounds W + BA to bf16, so the logits move a
     little; the check asks that this rounding noise is small next to what the training changed:
     top-1 agreement >= --min_top1 and mean KL(base+LoRA || merged) <= --max_kl_ratio x mean KL(base+LoRA || base)
     (or below --kl_floor, for adapters that changed almost nothing)
  2. the saved folder, reloaded the way lmms-eval loads it (from_pretrained + AutoProcessor), gives the same
     logits (top-1 >= --min_top1, mean KL <= --reload_max_kl) as the merged model in memory: nothing lost
     in saving. A different greedy caption is only a warning (a near tie)
  3. the reloaded processor's chat template gives the exact prompt string grpo.py trained with
The result goes into <out>/export.json; a failed check stops with an error. --check_only runs checks 1 and 3
without writing the ~16 GB folder (lmms_eval_lora.py does the same merge in memory at evaluation time).

  python export_ckpt.py --adapter out_grpo/ckpt/step_300 --out exports/self_step300 --grpo_dir out_grpo
  python export_ckpt.py --adapter out_grpo/adapter --out exports/self_final          # finds out_grpo/args.json
  python export_ckpt.py --mock --out exports_mock/base                                # tiny random model, no LoRA
  python export_ckpt.py --mock --adapter out_grpo_mock/ckpt/step_2 --out exports_mock/self_step2
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch GPU numbers == nvidia-smi numbers

import argparse
import hashlib
import json
import time

import torch

import embedder as E
import gpu
import grpo
import mock_data as M
import prompts as P


# Qwen2.5-VL's chat format for single images (the real processor ships its own template)
MOCK_CHAT_TEMPLATE = (
    "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for c in message['content'] %}{% if c['type'] == 'image' or 'image' in c %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def train_config(a):
    """What the adapter was trained with: the run's args.json, updated by <adapter>/grpo_train.json (final adapter).
    Older runs wrote grpo_train.json without max_pixels, so both are read."""
    cands = [os.path.join(a.adapter, "grpo_train.json")] if a.adapter else []
    if a.grpo_dir:
        cands.append(os.path.join(a.grpo_dir, "args.json"))
    if a.adapter:  # <out_dir>/adapter/ and <out_dir>/ckpt/step_N/ (ckpt_dir may live elsewhere: pass --grpo_dir)
        cands += [os.path.join(a.adapter, "..", "args.json"), os.path.join(a.adapter, "..", "..", "args.json")]
    found = [os.path.normpath(p) for p in cands if os.path.exists(p)]
    cfg = {}
    for p in reversed(found):  # the first candidate wins
        with open(p) as f:
            cfg.update(json.load(f))
    return (", ".join(found) or None), cfg


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


@torch.no_grad()
def greedy(model, processor, image, prompt, device, max_new_tokens):
    """Processor inputs, prompt + greedy caption ids, caption text (repetition penalty 1.0 as in grpo.py)."""
    inp = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    tok = processor.tokenizer
    eos, pad = grpo.special_ids(tok)
    seq = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                         pad_token_id=pad, repetition_penalty=1.0)
    return inp, seq, tok.decode(seq[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def forced_logits(model, inp, seq):
    """Teacher-forced logits [T, V] at every position that predicts a caption token of seq."""
    n = inp["input_ids"].shape[1]
    kw = {}
    for k, v in inp.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if torch.is_tensor(v) and v.dim() == 2 and v.shape == inp["input_ids"].shape:  # per-token, e.g. mm_token_type_ids
            v = torch.cat([v, v.new_zeros(1, seq.shape[1] - n)], 1)
        kw[k] = v
    out = model(input_ids=seq, attention_mask=torch.ones_like(seq), **kw).logits
    return out[0, n - 1:-1].float().cpu()


def compare(ref, other):
    """How far `other` is from `ref` (both [T, V])."""
    lr, lo = ref.log_softmax(-1), other.log_softmax(-1)
    kl = (lr.exp() * (lr - lo)).sum(-1)
    return dict(top1_agree=float((ref.argmax(-1) == other.argmax(-1)).float().mean()),
                mean_kl=float(kl.mean()), max_kl=float(kl.max()), max_abs_logit_diff=float((ref - other).abs().max()),
                positions=int(ref.shape[0]))


def show(d):
    return (f"top-1 agreement {d['top1_agree']:.3f}, mean KL {d['mean_kl']:.2e}, max KL {d['max_kl']:.2e}, "
            f"max |logit diff| {d['max_abs_logit_diff']:.2e} ({d['positions']} positions)")


def test_image(a):
    if a.probe_image:
        from PIL import Image
        return Image.open(a.probe_image).convert("RGB")
    return M.image("synth_5000.png")  # any image works for an equality check


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=None, help="PEFT adapter folder (step_N/ or adapter/); none = export the base")
    ap.add_argument("--out", required=True, help="output folder (a full HF model, ~16 GB for 7B)")
    ap.add_argument("--grpo_dir", default=None, help="the GRPO out_dir (for args.json when the ckpt is elsewhere)")
    ap.add_argument("--model", default=None, help="base model (default: from the training config)")
    ap.add_argument("--max_pixels", type=int, default=None, help="default: from the training config")
    ap.add_argument("--dtype", default=None, help="default bfloat16 (lmms-eval loads in bf16); --mock: float32")
    ap.add_argument("--probe_image", default=None, help="image for the equality checks (default: a synthetic one)")
    ap.add_argument("--probe_tokens", type=int, default=64, help="greedy caption length compared in check 2")
    ap.add_argument("--min_top1", type=float, default=0.95,
                    help="check 1: least top-1 agreement of merged with base + LoRA over the caption positions")
    ap.add_argument("--max_kl_ratio", type=float, default=0.2,
                    help="check 1: merge noise (mean KL) at most this fraction of what the LoRA changed")
    ap.add_argument("--kl_floor", type=float, default=None,
                    help="check 1: a mean KL below this always passes (default 1e-2 for bf16, 1e-6 for float32)")
    ap.add_argument("--reload_max_kl", type=float, default=None,
                    help="check 2: largest mean KL reloaded vs in memory (default 1e-3 for bf16, 1e-6 for float32)")
    ap.add_argument("--check_only", action="store_true",
                    help="checks 1 and 3 only, no model folder written (~0 GB): for lmms_eval_lora.py, which merges "
                         "the adapter in memory the same way; <out> gets export.json only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mock", action="store_true", help="tiny random Qwen2.5-VL (as grpo.py --mock), CPU")
    gpu.add_args(ap)
    a = ap.parse_args()
    a.device = gpu.resolve(a, mock=a.mock)
    if a.mock:
        a.dtype = a.dtype or "float32"  # the mock model is built in float32 (bfloat16: cast, as a bf16 test)
    a.dtype = a.dtype or "bfloat16"

    # never write into a training folder: the adapters and checkpoints stay exactly as they are
    out = os.path.realpath(a.out)
    for d in [a.adapter, a.grpo_dir]:
        if d and (out == os.path.realpath(d) or out.startswith(os.path.realpath(d) + os.sep)):
            raise SystemExit(f"[export] --out {a.out} is inside {d}; pick a folder outside the training run")
    if os.path.exists(os.path.join(a.out, "adapter_config.json")) or os.path.exists(os.path.join(a.out, "train_log.jsonl")):
        raise SystemExit(f"[export] --out {a.out} holds a LoRA adapter / GRPO run; pick an empty folder")
    if os.path.exists(os.path.join(a.out, "export.json")) and not a.overwrite:
        raise SystemExit(f"[export] {a.out} already exported (export.json); pass --overwrite to redo it")
    cfg_path, cfg = train_config(a)
    a.model = a.model or cfg.get("model", "Qwen/Qwen2.5-VL-7B-Instruct")
    a.max_pixels = a.max_pixels or cfg.get("max_pixels", 1024 * 28 * 28)
    caption_prompt = cfg.get("caption_prompt", "Describe this image.")
    if a.adapter and not cfg:
        print("[export] no training config found (grpo_train.json / args.json): using defaults; "
              "pass --grpo_dir to be sure", flush=True)
    if a.adapter:
        with open(os.path.join(a.adapter, "adapter_config.json")) as f:
            base_in_adapter = json.load(f).get("base_model_name_or_path")
        if base_in_adapter and not a.mock and base_in_adapter != a.model:
            raise SystemExit(f"[export] adapter was trained on {base_in_adapter}, not {a.model}")

    t0 = time.time()
    emb = E.load_mock(a.max_pixels, device=a.device) if a.mock else E.load_real(a.model, a.max_pixels, a.device, a.dtype)
    if a.mock:
        emb.model.to(getattr(torch, a.dtype))
    model, processor = emb.model, emb.processor
    prompt = P.qwen_chat(P.IMAGE_TOKEN + caption_prompt)
    image = test_image(a)
    n_tok = a.probe_tokens if not a.mock else 12
    fp32 = a.dtype == "float32"
    reload_kl = a.reload_max_kl if a.reload_max_kl is not None else (1e-6 if fp32 else 1e-3)
    kl_floor = a.kl_floor if a.kl_floor is not None else (1e-6 if fp32 else 1e-2)
    report = dict(base=a.model, adapter=os.path.abspath(a.adapter) if a.adapter else None, train_config=cfg_path,
                  caption_prompt=caption_prompt, max_pixels=a.max_pixels, min_pixels=56 * 56, dtype=a.dtype,
                  mock=a.mock, train=cfg)

    if a.adapter:
        from peft import PeftModel
        report["adapter_sha256_16"] = sha_of(os.path.join(a.adapter, "adapter_model.safetensors"))
        peft_model = PeftModel.from_pretrained(model, a.adapter).eval()
        inp, seq, ref_caption = greedy(peft_model, processor, image, prompt, a.device, n_tok)
        ref_logits = forced_logits(peft_model, inp, seq)
        with peft_model.disable_adapter():
            base_logits = forced_logits(peft_model, inp, seq)
        model = peft_model.merge_and_unload()
        mem_logits = forced_logits(model, inp, seq)
        _, _, mem_caption = greedy(model, processor, image, prompt, a.device, n_tok)
        lora, merge = compare(ref_logits, base_logits), compare(ref_logits, mem_logits)
        limit = max(a.max_kl_ratio * lora["mean_kl"], kl_floor)
        report.update(lora_effect=lora, merge_error=merge, merge_kl_limit=limit,
                      base_lora_caption=ref_caption, merged_caption=mem_caption,
                      merged_same_caption=mem_caption == ref_caption)
        print(f"[export] what the LoRA changed (base+LoRA vs base): {show(lora)}", flush=True)
        print(f"[export] check 1: merged vs base + LoRA: {show(merge)}", flush=True)
        print(f"[export]   limit: top-1 >= {a.min_top1}, mean KL <= {limit:.2e} "
              f"(= max({a.max_kl_ratio} x LoRA effect, {kl_floor:g})); same greedy caption: {mem_caption == ref_caption}",
              flush=True)
        if merge["top1_agree"] < a.min_top1 or merge["mean_kl"] > limit:
            raise SystemExit(f"[export] the merged model is too far from base + LoRA (nothing saved):\n"
                             f"  base+LoRA: {ref_caption!r}\n  merged   : {mem_caption!r}\n"
                             f"  lmms-eval always loads in bf16, so a float32 export would not help; if the numbers above\n"
                             f"  look like plain bf16 rounding, loosen --min_top1 / --max_kl_ratio")
    else:
        inp, seq, mem_caption = greedy(model, processor, image, prompt, a.device, n_tok)
        mem_logits = forced_logits(model, inp, seq)

    if a.mock and not getattr(processor, "chat_template", None):
        processor.chat_template = MOCK_CHAT_TEMPLATE  # lmms-eval formats prompts with the processor's template
    os.makedirs(a.out, exist_ok=True)
    report["check_only"] = a.check_only
    if a.check_only:  # lmms_eval_lora.py loads the base's own processor; check 3 on that
        check_template(report, processor, caption_prompt, prompt)
        return finish(a, report, t0)
    model.save_pretrained(a.out, safe_serialization=True)
    processor.save_pretrained(a.out)
    del model, emb
    if a.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # reload exactly as lmms-eval's qwen2_5_vl does (same dtype as saved here)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    attn = None if a.mock else E.attn_implementation()
    re_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.out, dtype=getattr(torch, a.dtype), device_map=a.device,
                                                                  **({"attn_implementation": attn} if attn else {})).eval()
    re_proc = AutoProcessor.from_pretrained(a.out, max_pixels=a.max_pixels, min_pixels=56 * 56)
    re_inp, _, re_caption = greedy(re_model, re_proc, image, prompt, a.device, n_tok)
    same_inputs = torch.equal(re_inp["input_ids"].cpu(), inp["input_ids"].cpu())
    reload = compare(mem_logits, forced_logits(re_model, re_inp, seq)) if same_inputs else None
    report.update(reload=reload, reload_same_inputs=same_inputs, reload_same_caption=re_caption == mem_caption,
                  probe_caption=re_caption)
    print(f"[export] check 2: reloaded folder vs merged model in memory: "
          f"{show(reload) if reload else 'processor gives different input ids'}, same greedy caption: "
          f"{re_caption == mem_caption}", flush=True)
    if not same_inputs or reload["mean_kl"] > reload_kl or reload["top1_agree"] < a.min_top1:
        raise SystemExit(f"[export] the saved folder does not reproduce the merged model (limit: top-1 >= "
                         f"{a.min_top1}, mean KL <= {reload_kl:g}):\n"
                         f"  in memory: {mem_caption!r}\n  reloaded : {re_caption!r}")
    if re_caption != mem_caption:
        print(f"[warn] check 2: greedy captions differ (logits agree within the limits, so a near tie):\n"
              f"  in memory: {mem_caption!r}\n  reloaded : {re_caption!r}", flush=True)

    check_template(report, re_proc, caption_prompt, prompt)
    finish(a, report, t0)


def check_template(report, re_proc, caption_prompt, prompt):
    """Check 3: lmms-eval builds the prompt with the processor's chat template and its default system prompt."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": caption_prompt}]}]
    try:
        templ = re_proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        report["chat_template_matches_training"] = templ == prompt
    except Exception as e:  # the mock processor has no chat template
        templ, report["chat_template_matches_training"] = None, None
        print(f"[export] check 3 skipped: {e}", flush=True)
    if report["chat_template_matches_training"] is False:
        print(f"[warn] check 3: lmms-eval's prompt differs from the training prompt:\n"
              f"  lmms-eval: {templ!r}\n  training : {prompt!r}", flush=True)
    elif report["chat_template_matches_training"]:
        print("[export] check 3: chat template gives the training prompt exactly", flush=True)


def finish(a, report, t0):
    report["sec"] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, "export.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[export] wrote {a.out}" + (" (export.json only: checks passed, no model saved)" if a.check_only else "")
          + f" ({report['sec']}s)")


if __name__ == "__main__":
    main()
