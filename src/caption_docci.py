"""DOCCI captions by the Step 2 captioner, before and after GRPO: Caption 2 for the P/N/H QA (generate.py).

Same model, processor, chat string and generation code as grpo.py's before/after evaluation
(embedder.load_real + PEFT adapter, grpo.generate, greedy, repetition penalty 1.0):
  before = BeforeRlModelCaption = Qwen2.5-VL, LoRA off
  after  = Argmax*              = Qwen2.5-VL + GRPO LoRA (--adapter out_grpo/adapter)
Only the caption prompt and the length are set here. grpo.py itself trains and evaluates with
"Describe this image." and up to 4096 new tokens (the defaults below); run_all.py passes its own prompts.

Output, one line per image, in the Caption 2 format generate.py reads (resumes from what is there):
  <out_dir>/caption2.jsonl  {"example_id", "image_file", "caption", "model", "captioner", "prompt_version",
                             "prompt", "n_tokens", "truncated"}
Images: --images <jsonl with example_id, image_file> (run_all.py writes it), else DOCCI test,
--n_images picked with --seed exactly like grpo.py (n=200, seed 0 = grpo.py's DOCCI eval images).
--shard k/N splits the images over processes (one per GPU; run_all.py sets it).

  python caption_docci.py --which before --n_images 200 --out_dir docci_cap_before
  python caption_docci.py --which after --adapter out_grpo/adapter --n_images 200 --out_dir docci_cap_after
  python caption_docci.py --mock --which after --out_dir docci_cap_mock      # tiny random model, CPU
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch GPU numbers == nvidia-smi numbers

import argparse
import glob
import json
import tarfile
import time
import urllib.request
import zlib

import data as D
import gpu
import grpo


def in_shard(shard, example_id):
    k, n = (int(x) for x in shard.split("/"))
    return n == 1 or zlib.crc32(example_id.encode()) % n == k


def shard_path(shard, path):
    k, n = (int(x) for x in shard.split("/"))
    return path if n == 1 else f"{path[:-len('.jsonl')]}.{k}-of-{n}.jsonl"


def read_rows(path):
    """Rows of x.jsonl and of every x.<k>-of-<N>.jsonl (a line cut off by a crash is skipped)."""
    rows = []
    for p in [path] + sorted(glob.glob(glob.escape(path[:-len(".jsonl")]) + ".*-of-*.jsonl")):
        if os.path.exists(p):
            with open(p) as f:
                for l in f:
                    try:
                        rows.append(json.loads(l))
                    except json.JSONDecodeError:
                        pass
    return rows


def records(a):
    """[(example_id, image_file)] and the image folder."""
    img_dir = a.image_dir or os.path.join(a.docci_dir, "images") if a.docci_dir else \
        a.image_dir or os.path.join(a.cache_dir, "docci", "images")
    if a.images:
        with open(a.images) as f:
            recs = [json.loads(l) for l in f if l.strip()]
    elif a.mock:  # mock_data's synthetic DOCCI stand-ins, nothing downloaded
        import mock_data as M
        recs = [dict(example_id=r["example_id"], image_file=r["example_id"] + ".png") for r in M.docci(a.n_images or 8)]
    else:
        jl = os.path.join(a.docci_dir, "docci_descriptions.jsonlines") if a.docci_dir else \
            D._download(D.DOCCI_JSONL, os.path.join(a.cache_dir, "docci", "docci_descriptions.jsonlines"))
        with open(jl) as f:
            recs = [json.loads(l) for l in f]
        n = a.n_images or sum(r["split"] == "test" for r in recs)
        recs = D._docci_sample(recs, n, a.seed)
    return [(r["example_id"], r["image_file"]) for r in recs], img_dir


def fetch_missing(files, img_dir):
    """Stream the images not on disk yet out of the DOCCI image tar (one pass), like data.load_docci."""
    os.makedirs(img_dir, exist_ok=True)
    missing = {f for f in files if not os.path.exists(os.path.join(img_dir, f))}
    if not missing:
        return
    print(f"[docci] streaming {len(missing)} images out of the 3.8 GB image tar (one pass) ...", flush=True)
    with urllib.request.urlopen(D.DOCCI_IMAGES_TAR) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if m.isfile() and name in missing:
                with open(os.path.join(img_dir, name), "wb") as g:
                    g.write(tf.extractfile(m).read())
                missing.discard(name)
                if not missing:
                    break


def load(a):
    """The captioner as grpo.build loads it for evaluation: base model, plus the adapter for 'after'."""
    import embedder as E
    from peft import LoraConfig, PeftModel, get_peft_model

    emb = E.load_mock(a.max_pixels, device=a.device) if a.mock else E.load_real(a.model, a.max_pixels, a.device, a.dtype)
    model = emb.model
    if a.which == "after":
        if a.adapter:
            model = PeftModel.from_pretrained(model, a.adapter)
            print(f"[model] loaded adapter {a.adapter}")
        elif a.mock:  # a fresh LoRA (B = 0): same captions as before, just to run the code path
            model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=grpo.LORA_TARGETS))
        else:
            raise SystemExit("[model] --which after needs --adapter <grpo.py out_dir>/adapter")
    return model, emb.processor


def mock_images(eids):
    import mock_data as M
    imgs = [r["image"] for r in M.docci(3)]
    return {e: imgs[i % len(imgs)] for i, e in enumerate(eids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", default="after", choices=["before", "after"],
                    help="before = LoRA off (BeforeRlModelCaption), after = GRPO LoRA on (Argmax*)")
    ap.add_argument("--adapter", default=None, help="GRPO LoRA (grpo.py writes <out_dir>/adapter)")
    ap.add_argument("--mock", action="store_true", help="tiny random Qwen2.5-VL + synthetic images, CPU")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    ap.add_argument("--caption_prompt", default="Describe this image.")
    ap.add_argument("--prompt_version", default="argmax", help="label stored with each caption (qa_prompts.CAPTION_PROMPTS name)")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8, help="images per generate call (grpo.py --eval_batch)")
    ap.add_argument("--images", default=None, help="JSONL with example_id and image_file (else DOCCI test)")
    ap.add_argument("--image_dir", default=None, help="folder with the DOCCI images (default: <cache_dir>/docci/images)")
    ap.add_argument("--n_images", type=int, default=200, help="without --images: DOCCI test images (0 = all 5,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=os.environ.get("ARGMAX_CACHE_DIR", "/mnt/data2/mchung45/argmax_caption_work/cache"))
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--out_dir", default="docci_captions")
    ap.add_argument("--shard", default="0/1", help="k/N: this process does part k of N")
    ap.add_argument("--no_gpu_pick", action="store_true", help="use cuda:0 of CUDA_VISIBLE_DEVICES as already set (run_all.py)")
    ap.add_argument("--n_gpus", type=int, default=1, help="accepted for run_all.py; one GPU per process")
    gpu.add_args(ap)
    a = ap.parse_args()
    a.temperature = 1.0  # read by grpo.generate only when sampling; captions here are greedy

    recs, img_dir = records(a)
    base = os.path.join(a.out_dir, "caption2.jsonl")
    done = {r["example_id"] for r in read_rows(base) if r.get("caption") is not None}
    todo = [(e, f) for e, f in recs if e not in done and in_shard(a.shard, e)]
    print(f"[caption] {a.which} RL, prompt {a.caption_prompt!r}, max {a.max_new_tokens} new tokens: "
          f"{len(todo)} images to do, {len(done)} done -> {os.path.abspath(base)}", flush=True)
    if not todo:
        return
    a.device = "cpu" if a.mock and a.device == "auto" else "cuda:0" if a.no_gpu_pick else gpu.resolve(a, mock=a.mock)
    if not a.mock:
        fetch_missing([f for _, f in todo], img_dir)
    model, processor = load(a)
    eos, _ = grpo.special_ids(processor.tokenizer)
    name = "mock tiny Qwen2.5-VL" if a.mock else a.model
    label = f"{name} (before RL, LoRA off)" if a.which == "before" else f"{name} + GRPO LoRA ({a.adapter or 'fresh'})"
    mock_imgs = mock_images([e for e, _ in todo]) if a.mock else None
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        images = [mock_imgs[e] if a.mock else D.load_image(os.path.join(img_dir, f)) for e, f in chunk]
        # LoRA off for 'before' is the plain base model, as grpo.generate(use_adapter=False) gives it
        _, plen, out, mask, texts = grpo.generate(a, model, processor, images, 1, sample=False, use_adapter=True)
        comp = out[:, plen:]
        rows = [dict(example_id=e, image_file=f, caption=t, model=label, captioner=a.which,
                     prompt_version=a.prompt_version, prompt=a.caption_prompt, n_tokens=int(m.sum()),
                     truncated=not bool((c == eos).any()))
                for (e, f), t, c, m in zip(chunk, texts, comp, mask)]
        with open(shard_path(a.shard, base), "a") as g:
            for r in rows:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[caption] {min(s + a.batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s  "
              f"(cut off at max_new_tokens: {sum(r['truncated'] for r in rows)}/{len(rows)})", flush=True)


if __name__ == "__main__":
    main()
