"""Caption 2: an AI caption for every DOCCI image, written by a VLM (default Qwen2.5-VL-7B-Instruct).

The QA generator compares it with Caption 1 (the DOCCI human description). Caption prompt
(qa_prompts.CAPTION_PROMPTS["simple"]): "Describe this image in detail."
Greedy decoding, up to 4096 new tokens, repetition penalty 1.05 (the model's own generation config;
it keeps long greedy captions from looping). Output, one line per image:
  <out_dir>/caption2.jsonl   {"example_id", "image_file", "caption", "model", "prompt_version", "prompt"}
To use captions you already have instead, skip this script and give generate.py --caption2 <your.jsonl>
(one JSON object per line with "example_id" and "caption").

  python caption.py --prompt_version simple --n_images 200 --out_dir cap2_simple
  python caption.py --mock --out_dir out_mock/caption2
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import os

import qa_common as C
import qa_prompts as P
from llm import LazyChat, add_llm_args, make_backend, resolve_defaults


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="fake captions with known mistakes, no GPU")
    ap.add_argument("--mock_captioner", default="before", choices=["before", "after"],
                    help="--mock only: imitate the captioner before or after GRPO (run_all.py --mock)")
    ap.add_argument("--splits", default="test")
    ap.add_argument("--n_images", type=int, default=0, help="random sample of this many images (0 = all)")
    ap.add_argument("--prompt_version", default="simple", choices=sorted(P.CAPTION_PROMPTS),
                    help="which caption prompt (qa_prompts.CAPTION_PROMPTS)")
    ap.add_argument("--prompt", default=None, help="your own caption prompt instead (prompt_version becomes custom)")
    ap.add_argument("--max_side", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"))
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--out_dir", default="out_caption2")
    add_llm_args(ap, "Qwen/Qwen2.5-VL-7B-Instruct", 4096)
    C.add_gpu_args(ap)
    C.add_shard_args(ap)
    a = ap.parse_args()
    if a.mock:
        a.backend, a.model = "mock", "mock"
    resolve_defaults(a, 4096)
    if a.temperature is None:
        a.temperature = 0.0
    if a.repetition_penalty is None:
        a.repetition_penalty = 1.05
    version = "custom" if a.prompt else a.prompt_version
    prompt = a.prompt or P.CAPTION_PROMPTS[a.prompt_version]
    if not a.mock:
        C.setup_caches()
    os.makedirs(a.out_dir, exist_ok=True)

    import mock_docci
    recs = mock_docci.records(a.n_images or 8) if a.mock else \
        C.load_docci(a.cache_dir, a.splits.split(","), a.n_images, a.seed, a.docci_dir)
    base = os.path.join(a.out_dir, "caption2.jsonl")
    done = {r["example_id"] for r in C.read_shards(base) if r.get("caption")}
    todo = [r for r in recs if r["example_id"] not in done and C.in_shard(a, r["example_id"])]
    print(f"[caption2] prompt {version}: {prompt!r}, max {a.max_tokens} tokens\n"
          f"[caption2] {len(todo)} images to caption, {len(done)} done  ->  {os.path.abspath(base)}", flush=True)
    if not todo:
        return

    def load():
        if a.backend == "vllm":
            C.pick_gpus(a)
        return make_backend(a, limit_images=1, mock_fn=lambda m: mock_docci.caption2(
            m[-1]["content"][-1]["example_id"], version, a.mock_captioner))
    chat = LazyChat(load)
    paths = {} if a.mock else C.docci_image_paths([r["image_file"] for r in todo], a.cache_dir, a.docci_dir)
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        convs = []
        for r in chunk:
            src = mock_docci.image(r["example_id"]) if a.mock else paths[r["image_file"]]
            m = P.caption_messages(C.image_data_url(src, a.max_side), prompt)
            if a.mock:
                m[-1]["content"][-1]["example_id"] = r["example_id"]
            convs.append(m)
        outs = chat(convs)
        C.append_jsonl(C.shard_path(a, base), [
            dict(example_id=r["example_id"], image_file=r["image_file"], caption=C.strip_think(o),
                 model=f"mock captioner ({a.mock_captioner} RL)" if a.mock else a.model,
                 captioner=a.mock_captioner if a.mock else "vllm", prompt_version=version, prompt=prompt)
            for r, o in zip(chunk, outs)])
        print(f"[caption2] {min(s + a.batch, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
