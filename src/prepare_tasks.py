"""lmms-eval task folder for the benchmarks lmms-eval does not ship. Now one: DOCCI captioning.

  docci_caption   DOCCI test images (google/docci; the GRPO runs never trained on them: training images are COCO
                  train2017, and DOCCI test is only the QA / before-after evaluation set), prompt "Describe this
                  image." (the training prompt), one caption per image, scored against the human description:
                    docci_CAPTURE   CAPTURE (DetailCaps' detailed-caption metric: objects, attributes, relations
                                    of the caption vs the reference; capture_metric package, GPU)
                    docci_CIDEr, docci_Bleu_4, docci_METEOR, docci_ROUGE_L   pycocoevalcap (Java PTB tokenizer)
                  lmms_eval_report.py adds the mean words per caption.
                  --n_images 1000 (seed 0) by default: a seeded sample of the 5,000 test images (~250 MB of
                  images under the DOCCI cache; 0 = all 5,000, ~1.2 GB, ~5x the generation time).

The folder (yaml + jsonl + docci_utils.py) is small; the images stay where the QA pipeline keeps them
(<DOCCI_QA_ROOT>/cache/docci/images, streamed out of the official tar once; or --docci_dir).

  python prepare_tasks.py --out_dir /mnt/data2/mchung45/lmms_eval_work/custom_tasks
  python prepare_tasks.py --out_dir tasks_mock --mock          # synthetic images, no downloads (smoke test)
  python prepare_tasks.py --out_dir <dir> --utils_only         # rewrite docci_utils.py only (scorer fixes)
"""
import argparse
import json
import os

DOCCI_YAML = """task: docci_caption
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function docci_utils.doc_to_visual
doc_to_text: !function docci_utils.doc_to_text
doc_to_target: description
generation_kwargs:
  max_new_tokens: {max_new_tokens}
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
  until: []
process_results: !function docci_utils.process_results
metric_list:
  - metric: docci_CAPTURE
    aggregation: !function docci_utils.capture
    higher_is_better: true
  - metric: docci_CIDEr
    aggregation: !function docci_utils.cider
    higher_is_better: true
  - metric: docci_Bleu_4
    aggregation: !function docci_utils.bleu4
    higher_is_better: true
  - metric: docci_METEOR
    aggregation: !function docci_utils.meteor
    higher_is_better: true
  - metric: docci_ROUGE_L
    aggregation: !function docci_utils.rougel
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    prompt: {prompt!r}
metadata:
  version: 0.0
  source: "google/docci test split, {n} images (seed {seed}); references = the human descriptions"
"""

UTILS = r'''"""Scorer of the docci_caption task (written by prepare_tasks.py).

Metrics are computed once per set of predictions (lmms-eval calls one aggregation function per metric, each with
its own list of the same records) and cached, like DetailCaps does them: CAPTURE on the raw texts, the pycocoevalcap metrics on PTB-tokenized
texts. DOCCI_MOCK_METRICS=1 (smoke test) replaces CAPTURE, which needs its GPU models, by a word-overlap F1.
"""
import hashlib
import json
import os
import sys
import tempfile
import types

from PIL import Image

METRICS = ["CAPTURE", "CIDEr", "Bleu_4", "METEOR", "ROUGE_L"]


def doc_to_visual(doc):
    return [Image.open(doc["image_path"]).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return (lmms_eval_specific_kwargs or {}).get("prompt", "Describe this image.")


def process_results(doc, results):
    rec = dict(image_id=doc["example_id"], pred=results[0], answer=[doc["description"]])
    return {f"docci_{m}": rec for m in METRICS}


# lmms-eval imports this file once per `!function` tag (importlib.util.module_from_spec), so a module-level dict
# would be a different dict for every metric and CAPTURE (15 + 12 min of parsing per 1000 captions) would run
# once per metric, as it did until 2026-11-07. The cache therefore lives in sys.modules, shared by every copy,
# and on disk (DOCCI_SCORE_CACHE_DIR, default $TMPDIR/docci_scores), keyed by the content of the records.
def _shared():
    m = sys.modules.setdefault("_docci_scores_shared", types.ModuleType("_docci_scores_shared"))
    return m.__dict__.setdefault("scores", {})


def _all_scores(results):
    recs = sorted((r["image_id"], r["pred"], list(r["answer"])) for r in results)
    key = hashlib.sha1(json.dumps(recs, ensure_ascii=False).encode()).hexdigest()
    mem = _shared()
    if key in mem:
        return mem[key]
    cache_dir = os.environ.get("DOCCI_SCORE_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "docci_scores")
    path = os.path.join(cache_dir, f"{key}{'_mock' if os.environ.get('DOCCI_MOCK_METRICS') else ''}.json")
    if os.path.exists(path):
        with open(path) as f:
            mem[key] = json.load(f)
        return mem[key]
    gts = {r["image_id"]: list(r["answer"]) for r in results}
    res = {r["image_id"]: [r["pred"]] for r in results}
    out = {}
    if os.environ.get("DOCCI_MOCK_METRICS"):
        out["CAPTURE"] = _overlap_f1(gts, res)
    else:
        from capture_metric.capture import CAPTURE
        score, _ = CAPTURE().compute_score(gts, res)
        out["CAPTURE"] = float(score)
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tok = PTBTokenizer()
    tgts = tok.tokenize({k: [dict(caption=c) for c in v] for k, v in gts.items()})
    tres = tok.tokenize({k: [dict(caption=c) for c in v] for k, v in res.items()})
    for scorer, name in ((Cider(), "CIDEr"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L")):
        score, _ = scorer.compute_score(tgts, tres)
        out[name] = float(score[3] if isinstance(score, list) else score)
    mem[key] = out
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)
    except OSError:
        pass  # the in-process cache still holds the scores
    return out


def _overlap_f1(gts, res):
    f1 = []
    for k, refs in gts.items():
        a, b = set(res[k][0].lower().split()), set(" ".join(refs).lower().split())
        tp = len(a & b)
        p, r = (tp / len(a) if a else 0.0), (tp / len(b) if b else 0.0)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1) / len(f1) if f1 else 0.0


def capture(results):
    return _all_scores(results)["CAPTURE"]


def cider(results):
    return _all_scores(results)["CIDEr"]


def bleu4(results):
    return _all_scores(results)["Bleu_4"]


def meteor(results):
    return _all_scores(results)["METEOR"]


def rougel(results):
    return _all_scores(results)["ROUGE_L"]
'''


def docci_rows(a):
    if a.mock:
        import mock_docci as M
        img_dir = os.path.join(a.out_dir, "mock_images")
        os.makedirs(img_dir, exist_ok=True)
        rows = []
        for r in M.records(a.mock_n):
            p = os.path.join(img_dir, r["example_id"] + ".png")
            M.image(r["example_id"]).save(p)
            rows.append(dict(example_id=r["example_id"], image_path=p, description=r["description"]))
        return rows
    import qa_common as C
    cache = a.docci_cache or os.path.join(C.WORK_ROOT, "cache")
    recs = C.load_docci(cache, splits=("test",), n=a.n_images, seed=a.seed, docci_dir=a.docci_dir)
    paths = C.docci_image_paths([r["image_file"] for r in recs], cache, a.docci_dir)
    return [dict(example_id=r["example_id"], image_path=paths[r["image_file"]], description=r["description"])
            for r in recs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="task folder for lmms_eval_run.py --include_path")
    ap.add_argument("--n_images", type=int, default=1000, help="DOCCI test images, seeded sample (0 = all 5,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="Describe this image.", help="caption prompt (the GRPO training prompt)")
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="task default; lmms_eval_run.py's caption group overrides it with the training limit")
    ap.add_argument("--docci_cache", default=None, help="DOCCI cache (default <DOCCI_QA_ROOT>/cache, as the QA pipeline)")
    ap.add_argument("--docci_dir", default=None, help="folder with docci_descriptions.jsonlines + images/ instead")
    ap.add_argument("--mock", action="store_true", help="synthetic images and descriptions, no downloads")
    ap.add_argument("--mock_n", type=int, default=4)
    ap.add_argument("--utils_only", action="store_true", help="rewrite only docci_utils.py in an existing folder")
    a = ap.parse_args()
    a.out_dir = os.path.abspath(a.out_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "docci_utils.py"), "w") as f:
        f.write(UTILS)
    if a.utils_only:
        print(f"[tasks] rewrote {a.out_dir}/docci_utils.py", flush=True)
        return
    rows = docci_rows(a)
    data = os.path.join(a.out_dir, "docci_caption.jsonl")
    with open(data, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(a.out_dir, "docci_caption.yaml"), "w") as f:
        f.write(DOCCI_YAML.format(data=data, max_new_tokens=a.max_new_tokens, prompt=a.prompt, n=len(rows), seed=a.seed))
    words = sum(len(r["description"].split()) for r in rows) / max(len(rows), 1)
    print(f"[tasks] docci_caption: {len(rows)} images, reference descriptions {words:.0f} words on average; "
          f"wrote {a.out_dir}: pass it as lmms_eval_run.py --include_path", flush=True)


if __name__ == "__main__":
    main()
