"""The test-time candidates with vLLM (README "Reward comparison"): the same items, prompts and pixels as the
lmms-eval cells, sampled ~5x faster than HF generate (continuous batching, the image prefix shared by the N
samples of an item). Writes prompting_trace.jsonl in the format of prompting_baselines.py's tts_sample cell, so
tts.py score / select / vqa read it unchanged.

  dump    lmms-eval's own task code builds every request of the tasks (prompt = the request's context, images =
          the task's doc_to_visual) on the CPU, no model; each image is saved as the JPEG (quality 85) lmms-eval's
          Qwen2.5-VL would pass, and each item's fields (CaptionQA: id, questions, shuffle_perms) are kept.
  sample  vLLM, N samples per item with Self-Validation's sampling (temperature 0.5, top-k 50, top-p 1) and
          Qwen's repetition penalty 1.05 (its generation_config.json; the HF path applies it too), seeded per item.
          Images: the saved JPEGs resized by qwen_vl_utils with the plan's min/max pixels, exactly as the HF path
          does before its processor, so the model reads the same pixels. Prompt: Qwen's chat template with
          lmms-eval's system prompt. Resumes: items already in the output are skipped.

  python tts_vllm.py dump   --tasks captionqa_x_natural,...,chair500 --include_path <custom tasks> --out_dir tts_out/items
  python tts_vllm.py sample --items tts_out/items --model <hf folder> --out_dir lmms_eval_tts/base-sample-vllm
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import base64
import json
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompting_baselines as PB  # noqa: E402

SYSTEM_PROMPT = "You are a helpful assistant."  # lmms-eval Qwen2_5_VL default


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        out = []
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a kill
        return out


def cmd_dump(a):
    if a.hf_dir:  # the same data folders as lmms_eval_run.py's caption_long cells (downloaded once)
        import lmms_eval_run as R
        os.environ.update(R.group_env(types.SimpleNamespace(hf_dir=os.path.abspath(a.hf_dir)), dict(os.environ), a.group))
    from lmms_eval.models.model_utils.media_encoder import encode_image_to_data_url
    from lmms_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager("WARNING", include_path=a.include_path)
    names = [t for t in a.tasks.split(",") if t]
    td = get_task_dict(names, tm)
    img_dir = os.path.join(a.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, "items.jsonl")
    done = {PB.item_key(r["task"], r["doc_id"]) for r in read_jsonl(out_path)}
    n_new = 0
    with open(out_path, "a") as fh:
        for name in names:
            task = td[name]
            if isinstance(task, tuple):
                task = task[1]
            task.build_all_requests(limit=a.limit, rank=0, world_size=1)
            raw = task.dataset
            for inst in task.instances:
                context, gen_kwargs, doc_to_visual, doc_id, tname, split = inst.args[:6]
                key = PB.item_key(tname, doc_id)
                if key in done:
                    continue
                doc = raw[split][doc_id]  # what lmms-eval's model reads (lm.task_dict = task.dataset)
                fields = PB.doc_fields(doc)
                try:  # the processed doc (CaptionQA's shuffle_perms come from process_docs)
                    fields.update(PB.doc_fields(task.eval_docs[doc_id]))
                except Exception:
                    pass
                os.makedirs(os.path.join(img_dir, tname), exist_ok=True)
                paths = []
                for k, im in enumerate([v for v in (doc_to_visual(doc) or []) if hasattr(v, "convert")]):
                    url = encode_image_to_data_url(im, image_format="JPEG", mime_type="image/jpeg", convert_rgb=True,
                                                   quality=85)
                    p = os.path.join(img_dir, tname, f"{doc_id}_{k}.jpg")
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(url.split(",", 1)[1]))
                    paths.append(p)
                fh.write(json.dumps(dict(task=tname, doc_id=doc_id, split=split, prompt=PB.strip_visual_placeholders(context),
                                         max_new_tokens=a.max_new_tokens, images=paths, doc=fields), ensure_ascii=False) + "\n")
                done.add(key)
                n_new += 1
            print(f"[dump] {name}: {len(task.instances)} items", flush=True)
    print(f"[dump] {n_new} new items -> {out_path} (images in {img_dir})", flush=True)


def generation_penalty(model):
    p = os.path.join(model, "generation_config.json")
    if os.path.exists(p):
        with open(p) as f:
            return float(json.load(f).get("repetition_penalty", 1.0))
    return 1.0


def cmd_sample(a):
    items = read_jsonl(os.path.join(a.items, "items.jsonl"))
    if a.limit:
        per = {}
        items = [r for r in items if per.setdefault(r["task"], []).append(1) or len(per[r["task"]]) <= a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, "prompting_trace.jsonl")
    done = {PB.item_key(r["task"], r["doc_id"]) for r in read_jsonl(out_path)}
    todo = [r for r in items if PB.item_key(r["task"], r["doc_id"]) not in done]
    print(f"[sample] {len(items)} items, {len(items) - len(todo)} sampled already", flush=True)
    if not todo:
        return
    import gpu  # only now: a finished run never waits for a GPU
    idx = gpu.pick(gpu.candidates(a.gpus), a.gpu_max_mem_mb, a.gpu_max_util, a.gpu_wait)  # never 0, 1, 2, 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)  # vLLM sees only that GPU (set before torch / vllm load)
    print(f"[sample] GPU {idx}", flush=True)
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    min_px, max_px = a.min_pixels, a.max_pixels
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=min_px, max_pixels=max_px)
    rp = a.repetition_penalty if a.repetition_penalty is not None else generation_penalty(a.model)
    tok = proc.tokenizer
    stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    n_img = max(len(r["images"]) for r in items)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_fraction, max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": max(n_img, 1)}, seed=a.seed, enable_prefix_caching=True,
              mm_processor_kwargs={"min_pixels": min_px, "max_pixels": max_px}, generation_config="vllm")
    settings = dict(n=a.n, temperature=a.temperature, top_k=a.top_k, top_p=1.0, repetition_penalty=rp)
    print(f"[sample] vLLM {a.model}: {json.dumps(settings)}, max_model_len {a.max_model_len}", flush=True)
    with open(os.path.join(a.out_dir, "sampling.json"), "w") as f:
        json.dump(dict(settings, model=a.model, min_pixels=min_px, max_pixels=max_px, seed=a.seed, engine="vllm"), f, indent=1)
    t0 = time.time()
    for c in range(0, len(todo), a.chunk):
        chunk = todo[c:c + a.chunk]
        reqs, params = [], []
        for r in chunk:
            content = [{"type": "image", "image": p, "min_pixels": min_px, "max_pixels": max_px} for p in r["images"]]
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content + [{"type": "text", "text": r["prompt"]}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            images = process_vision_info(msgs)[0]  # the HF path's resize (smart_resize with the plan's limits)
            reqs.append(dict(prompt=text, multi_modal_data={"image": images} if images else {}))
            params.append(SamplingParams(max_tokens=r["max_new_tokens"], stop_token_ids=stops,
                                         seed=PB.item_seed(a.seed, r["task"], r["doc_id"]), **settings))
        outs = llm.generate(reqs, params, use_tqdm=False)
        with open(out_path, "a") as fh:
            for r, o in zip(chunk, outs):
                cands = [x.text.strip() for x in o.outputs]
                rec = dict(task=r["task"], doc_id=r["doc_id"], split=r["split"], prompt=r["prompt"],
                           max_new_tokens=r["max_new_tokens"], images=r["images"], candidates=cands, doc=r["doc"],
                           finish=[x.finish_reason for x in o.outputs])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = c + len(chunk)
        el = time.time() - t0
        print(f"[sample] {n}/{len(todo)} items ({el:.0f}s, ~{el / n * (len(todo) - n) / 60:.0f} min left)", flush=True)
    print(f"[sample] done -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--tasks", required=True)
    p.add_argument("--include_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--hf_dir", default=None, help="lmms_eval_run.py's --hf_dir (benchmark data under datasets_<group>)")
    p.add_argument("--group", default="caption_long")
    p = sub.add_parser("sample")
    p.add_argument("--items", required=True, help="dump's out_dir")
    p.add_argument("--model", required=True, help="HF model folder (an export_ckpt.py folder for a GRPO run)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=None, help="default: the model's generation_config.json")
    p.add_argument("--min_pixels", type=int, default=3136)
    p.add_argument("--max_pixels", type=int, default=802816)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="first N items per task (quick run)")
    p.add_argument("--chunk", type=int, default=128, help="items per vLLM call (results written after each)")
    p.add_argument("--gpu_fraction", type=float, default=0.45)
    p.add_argument("--max_model_len", type=int, default=8192)
    import gpu
    gpu.add_args(p)
    a = ap.parse_args()
    {"dump": cmd_dump, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
