"""lmms-eval tasks for the two Self-Aug benchmarks lmms-eval does not ship: POPE on A-OKVQA and MMHal-Bench.

Self-Aug (arXiv 2510.13315) evaluates on POPE (MSCOCO and A-OKVQA), MME-Perception, MMVP, MMHal-Bench,
LLaVA-Bench (in-the-wild) and MM-Vet. lmms-eval has all of them except these two, so this script writes
them as an lmms-eval task folder (pass it to lmms_eval_run.py --include_path):

  pope_aokvqa_{random,popular,adversarial}  3 x 3000 yes/no questions, the A-OKVQA split of POPE as the VCD
                                            code (which Self-Aug builds on) ships it. Images: local COCO
                                            (the files are named COCO_val2014_<id>.jpg; looked up by id in
                                            val2017/ then train2017/, any depth under COCO_ROOT), else
                                            downloaded once into <data_dir>/coco_val2014 (500 images, ~80 MB).
                                            Prompt and metrics as lmms-eval's pope.
  mmhal_bench                               96 questions (Shengcao1006/MMHal-Bench: response_template.json +
                                            images/). Scored by the LLM judge with the official MMHal prompt
                                            (LLaVA-RLHF eval_gpt_mmhal.py): rating 0-6, hallucination = rating < 3.
                                            The judge is the OpenAI-compatible server at OPENAI_API_URL
                                            (lmms_eval_run.py sets it from --judge_url); the paper used GPT-4.

  python prepare_selfaug.py --out_dir /mnt/data2/minhochung/lmms_eval_work/selfaug_tasks
  python prepare_selfaug.py --out_dir selfaug_mock --mock         # synthetic images, no downloads (smoke test)
"""
import argparse
import json
import os
import urllib.request

VCD_COMMIT = "d6568ff81b8fd306a49e630df44f2db5c2300191"
POPE_URL = ("https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/" + VCD_COMMIT +
            "/experiments/data/POPE/aokvqa/aokvqa_pope_{split}.json")
SPLITS = ["random", "popular", "adversarial"]
COCO_ROOT = os.environ.get("COCO_ROOT", "/mnt/data2/shared_dataset/COCO-2017")

POPE_YAML = """task: pope_aokvqa_{split}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function selfaug_utils.doc_to_visual
doc_to_text: !function selfaug_utils.pope_doc_to_text
doc_to_target: answer
generation_kwargs:
  max_new_tokens: 128
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
process_results: !function selfaug_utils.pope_process_results
metric_list:
  - metric: pope_accuracy
    aggregation: !function selfaug_utils.pope_accuracy
    higher_is_better: true
  - metric: pope_precision
    aggregation: !function selfaug_utils.pope_precision
    higher_is_better: true
  - metric: pope_recall
    aggregation: !function selfaug_utils.pope_recall
    higher_is_better: true
  - metric: pope_f1_score
    aggregation: !function selfaug_utils.pope_f1
    higher_is_better: true
  - metric: pope_yes_ratio
    aggregation: !function selfaug_utils.pope_yes_ratio
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer the question using a single word or phrase."
metadata:
  version: 0.0
  source: "VCD {commit} experiments/data/POPE/aokvqa/aokvqa_pope_{split}.json"
"""

MMHAL_YAML = """task: mmhal_bench
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function selfaug_utils.doc_to_visual
doc_to_text: !function selfaug_utils.mmhal_doc_to_text
doc_to_target: gt_answer
generation_kwargs:
  max_new_tokens: 1024
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
  until: []
process_results: !function selfaug_utils.mmhal_process_results
metric_list:
  - metric: mmhal_score
    aggregation: !function selfaug_utils.mmhal_score
    higher_is_better: true
  - metric: mmhal_hallucination_rate
    aggregation: !function selfaug_utils.mmhal_hal_rate
    higher_is_better: false
  - metric: mmhal_judge_failures
    aggregation: !function selfaug_utils.mmhal_failures
    higher_is_better: false
metadata:
  version: 0.0
  source: "Shengcao1006/MMHal-Bench response_template.json + images/; judge prompt from LLaVA-RLHF Eval/eval_gpt_mmhal.py"
"""

UTILS = r'''"""Helpers for the tasks prepare_selfaug.py writes (POPE A-OKVQA, MMHal-Bench)."""
import os
import re
import time

import requests
from PIL import Image


def doc_to_visual(doc):
    return [Image.open(doc["image_path"]).convert("RGB")]


# ---------------------------------------------------------------- POPE (same prompt / scoring as lmms-eval pope)
def pope_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kw = lmms_eval_specific_kwargs or {}
    return f"{kw.get('pre_prompt', '')}{doc['question'].strip()}{kw.get('post_prompt', '')}"


def pope_process_results(doc, results):
    pred, gt = results[0].lower().strip(), doc["answer"].lower().strip()
    r = {"question_id": doc["question_id"], "score": float(pred == gt), "prediction": pred, "ground_truth": gt}
    return {k: r for k in ("pope_accuracy", "pope_precision", "pope_recall", "pope_f1_score", "pope_yes_ratio")}


def _counts(results):
    tp = sum(r["ground_truth"] == "yes" and r["prediction"] == "yes" for r in results)
    fp = sum(r["ground_truth"] == "no" and r["prediction"] == "yes" for r in results)
    fn = sum(r["ground_truth"] == "yes" and r["prediction"] != "yes" for r in results)
    return tp, fp, fn


def pope_accuracy(results):
    return sum(r["score"] for r in results) / len(results)


def pope_precision(results):
    tp, fp, _ = _counts(results)
    return tp / (tp + fp) if tp + fp else 0.0


def pope_recall(results):
    tp, _, fn = _counts(results)
    return tp / (tp + fn) if tp + fn else 0.0


def pope_f1(results):
    p, r = pope_precision(results), pope_recall(results)
    return 2 * p * r / (p + r) if p + r else 0.0


def pope_yes_ratio(results):
    return sum(r["prediction"] == "yes" for r in results) / len(results)


# ---------------------------------------------------------------- MMHal-Bench (official GPT-4 judge prompt)
MMHAL_TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmhal_judge_prompt.txt")).read()


def mmhal_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def _judge(prompt):
    base = os.environ.get("OPENAI_API_URL", "http://localhost:8000/v1").rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    model = os.environ.get("MODEL_VERSION", "Qwen/Qwen2.5-14B-Instruct")
    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, timeout=300, json=dict(
                model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1536))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[mmhal] judge error ({attempt + 1}/5): {e}", flush=True)
            time.sleep(10)
    return ""


def mmhal_process_results(doc, results):
    content = doc["image_content"]
    content = ", ".join(content) if isinstance(content, list) else str(content)
    review = _judge(MMHAL_TEMPLATE.format(content, doc["question"], doc["gt_answer"], results[0]))
    # the official script wants exactly one "Rating: N"; a local judge often restates the options or explains
    # first, so the last "Rating: N" in the review is its verdict
    found = re.findall(r"rating\s*[:\uff1a]\s*\**\s*([0-6])\b", review, flags=re.I)
    score = int(found[-1]) if found else None
    rec = {"id": doc["id"], "question_type": doc.get("question_type"), "score": 0 if score is None else score,
           "failed": score is None, "review": review}
    return {"mmhal_score": rec, "mmhal_hallucination_rate": rec, "mmhal_judge_failures": rec}


def mmhal_score(results):
    return sum(r["score"] for r in results) / len(results)


def mmhal_hal_rate(results):
    return sum(r["score"] < 3 for r in results) / len(results)


def mmhal_failures(results):
    return sum(r["failed"] for r in results) / len(results)
'''

# LLaVA-RLHF Eval/eval_gpt_mmhal.py `template`, verbatim (the four {} are image contents, question,
# standard answer, model answer)
MMHAL_PROMPT_URL = ("https://raw.githubusercontent.com/llava-rlhf/LLaVA-RLHF/main/Eval/eval_gpt_mmhal.py")


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode()


def mmhal_prompt():
    src = fetch(MMHAL_PROMPT_URL)
    start = src.index("template = '''") + len("template = '''")
    return src[start:src.index("'''", start)]


COCO_VAL2014_URL = "http://images.cocodataset.org/val2014/{}"


def coco_images(names, dl_dir):
    """{COCO_val2014_<id>.jpg: path}. A local copy under COCO_ROOT first (val2017 / train2017 / val2014, found a
    few levels deep as data.py does); the rest is downloaded once from cocodataset.org into dl_dir (~160 KB each)."""
    import data as D
    D.COCO_ROOT = COCO_ROOT
    dirs = [d for d in (D._find_local("img", s) for s in ("val2017", "train2017", "val2014")) if d]
    found, todo = {}, []
    for name in sorted(set(names)):
        stem = name.rsplit("_", 1)[-1]
        p = next((os.path.join(d, fn) for d in dirs for fn in (stem, name) if os.path.exists(os.path.join(d, fn))), None)
        if p:
            found[name] = p
        else:
            todo.append(name)
    if todo:
        print(f"[pope_aokvqa] {len(found)} images found under {COCO_ROOT}; downloading {len(todo)} into {dl_dir}",
              flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def get(name):
            try:
                return name, D._download(COCO_VAL2014_URL.format(name), os.path.join(dl_dir, name))
            except Exception as e:
                print(f"[pope_aokvqa] {name}: {e}", flush=True)
                return name, None
        with ThreadPoolExecutor(16) as ex:
            found.update({n: p for n, p in ex.map(get, todo) if p})
    return found


def write_pope(a, tasks):
    for split in SPLITS:
        if a.mock:
            import mock_data as M
            rows = []
            for i in range(a.mock_n):
                path = os.path.join(tasks, "mock_images", f"synth_{5000 + i}.png")
                s, c, _, _ = M._scene(5000 + i)
                rows.append(dict(question_id=i, image_path=path, question=f"Is there a {s} in the image?",
                                 answer="yes" if i % 2 == 0 else "no"))
        else:
            src = [json.loads(l) for l in fetch(POPE_URL.format(split=split)).splitlines() if l.strip()]
            paths = coco_images([r["image"] for r in src], os.path.join(a.data_dir, "coco_val2014"))
            missing = sorted({r["image"] for r in src} - set(paths))
            if missing:
                raise SystemExit(f"[pope_aokvqa] {len(missing)} COCO images neither under {COCO_ROOT} nor downloadable "
                                 f"(e.g. {missing[:3]}); set COCO_ROOT or check the network")
            rows = [dict(question_id=r["question_id"], image_path=paths[r["image"]], question=r["text"],
                         answer=r["label"]) for r in src]
            assert len(rows) == 3000, len(rows)
        data = os.path.join(tasks, f"pope_aokvqa_{split}.jsonl")
        with open(data, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(os.path.join(tasks, f"pope_aokvqa_{split}.yaml"), "w") as f:
            f.write(POPE_YAML.format(split=split, data=data, commit=VCD_COMMIT))
        print(f"[pope_aokvqa] {split}: {len(rows)} questions, {len({r['image_path'] for r in rows})} images")


def write_mmhal(a, tasks):
    if a.mock:
        rows = [dict(id=i, question_type="attribute", question_topic="mock", image_content=["shape"],
                     question="What shape is on the left?", gt_answer="A shape.",
                     image_path=os.path.join(tasks, "mock_images", f"synth_{5000 + i}.png")) for i in range(a.mock_n)]
    else:
        from huggingface_hub import snapshot_download
        root = snapshot_download("Shengcao1006/MMHal-Bench", repo_type="dataset",
                                 allow_patterns=["response_template.json", "images/*"],
                                 local_dir=os.path.join(a.data_dir, "MMHal-Bench"))
        with open(os.path.join(root, "response_template.json")) as f:
            src = json.load(f)
        rows = []
        for i, r in enumerate(src):  # the official scorer depends on this order (question type = index % 8)
            path = os.path.join(root, "images", os.path.basename(r["image_src"]))
            if not os.path.exists(path):
                raise SystemExit(f"[mmhal] image missing: {path}")
            rows.append(dict(id=i, question_type=r.get("question_type"), question_topic=r.get("question_topic"),
                             image_content=r["image_content"], question=r["question"], gt_answer=r["gt_answer"],
                             image_path=path))
        assert len(rows) == 96, len(rows)
    data = os.path.join(tasks, "mmhal_bench.jsonl")
    with open(data, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(tasks, "mmhal_bench.yaml"), "w") as f:
        f.write(MMHAL_YAML.format(data=data))
    print(f"[mmhal] {len(rows)} questions")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="task folder for lmms_eval_run.py --include_path")
    ap.add_argument("--data_dir", default=None, help="where MMHal-Bench is downloaded (default <out_dir>/data)")
    ap.add_argument("--mock", action="store_true", help="synthetic images and a stub prompt, no downloads")
    ap.add_argument("--mock_n", type=int, default=4)
    ap.add_argument("--utils_only", action="store_true", help="rewrite only selfaug_utils.py (scorer fix) in an existing folder")
    a = ap.parse_args()
    tasks = os.path.abspath(a.out_dir)
    a.data_dir = os.path.abspath(a.data_dir or os.path.join(tasks, "data"))
    os.makedirs(tasks, exist_ok=True)
    if a.mock:
        import mock_data as M
        os.makedirs(os.path.join(tasks, "mock_images"), exist_ok=True)
        for i in range(a.mock_n):
            M.image(f"synth_{5000 + i}.png").save(os.path.join(tasks, "mock_images", f"synth_{5000 + i}.png"))
        prompt = "Image contents: {}\nQuestion: {}\nAnswer: {}\nResponse: {}\nRate it (Rating: 0-6)."
    else:
        prompt = mmhal_prompt()
        assert prompt.count("{}") == 4 and "Rating: 6" in prompt, "MMHal judge prompt not found"
    with open(os.path.join(tasks, "mmhal_judge_prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join(tasks, "selfaug_utils.py"), "w") as f:
        f.write(UTILS)
    if a.utils_only:
        print(f"[selfaug] rewrote {tasks}/selfaug_utils.py", flush=True)
        return
    write_pope(a, tasks)
    write_mmhal(a, tasks)
    print(f"[selfaug] wrote {tasks}: pass it as lmms_eval_run.py --include_path")


if __name__ == "__main__":
    main()
