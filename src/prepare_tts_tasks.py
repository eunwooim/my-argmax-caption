"""lmms-eval tasks of the reward comparison (test-time scaling and training, README "Reward comparison"), written
into the same task folder as prepare_tasks.py (lmms_eval_run.py --include_path):

  captionqa_x_{natural,document,ecommerce,embodiedai}
      lmms-eval's CaptionQA (Borise/CaptionQA): the model writes one caption per image, a text-only judge answers
      the image's multiple-choice questions from the caption alone. Same data, prompt, option shuffling, judge
      prompt and metrics as lmms-eval's captionqa_* tasks; the only change is that every question's chosen option
      is logged too (captionqa_ext.py), so tts_report.py can compare it with the captioning model's own answer
      when it sees the image (agreement). The judge's questions of one image are sent in parallel
      (CAPTIONQA_JUDGE_WORKERS, default 8; greedy, so the answers are the same).
  chair500
      Self-Validation's CHAIR-500 (github.com/Liushiyu-0709/SelfVal, commit below): its 500 COCO val2014 images,
      its prompt ("Please describe this image in detail."), scored by its chair.py with its chair.pkl
      (CHAIR_s, CHAIR_i, object recall, precision and F1, the paper's protocol).
      Images: local COCO (val2017 has 73 of the 500; any folder of <id>.jpg / COCO_val2014_<id>.jpg via
      --coco_dirs), the rest downloaded once from images.cocodataset.org (~70 MB).
  amber_g
      ships with lmms-eval; it reads AMBER's metadata from $AMBER_BASE_DIR/metadata. This script downloads the
      four metadata files of the AMBER repo (commit below) into <out_dir>/amber/metadata; rl_paper_commands.sh
      exports AMBER_BASE_DIR=<out_dir>/amber.

  python prepare_tts_tasks.py --out_dir /mnt/data2/mchung45/lmms_eval_work/custom_tasks
  python prepare_tts_tasks.py --out_dir tasks_mock --mock --selfval_dir <SelfVal clone>   # smoke test, offline
"""
import argparse
import json
import os
import shutil
import urllib.request

SELFVAL_COMMIT = "4147ca2800888e03ee96043fb64364e554dce6fd"
SELFVAL_RAW = f"https://raw.githubusercontent.com/Liushiyu-0709/SelfVal/{SELFVAL_COMMIT}/chair500/"
AMBER_COMMIT = "534babf6bbfcce2e735c26289dedfb21cef3c939"
AMBER_RAW = f"https://raw.githubusercontent.com/junyangwang0410/AMBER/{AMBER_COMMIT}/data/"
AMBER_FILES = ["annotations.json", "relation.json", "safe_words.txt", "metrics.txt"]
COCO_URL = "http://images.cocodataset.org/val2014/COCO_val2014_{id:012d}.jpg"
CAPTIONQA_DOMAINS = ["natural", "document", "ecommerce", "embodiedai"]

CAPTIONQA_YAML = """task: captionqa_x_{domain}
dataset_path: Borise/CaptionQA
dataset_kwargs:
  trust_remote_code: true
test_split: {domain}
output_type: generate_until
dataset_name: null
process_docs: !function captionqa_ext.captionqa_process_docs
doc_to_visual: !function captionqa_ext.captionqa_doc_to_visual
doc_to_text: !function captionqa_ext.captionqa_doc_to_text
doc_to_target: "id"
generation_kwargs:
  max_new_tokens: 4096
  temperature: 0.7
  top_p: 1.0
  num_beams: 1
  do_sample: true
process_results: !function captionqa_ext.captionqa_process_results
metric_list:
  - metric: captionqa_score
    aggregation: !function captionqa_ext.captionqa_aggregate_score
    higher_is_better: true
  - metric: captionqa_accuracy
    aggregation: !function captionqa_ext.captionqa_aggregate_accuracy
    higher_is_better: true
  - metric: captionqa_cannot_answer_rate
    aggregation: !function captionqa_ext.captionqa_aggregate_cannot_answer
    higher_is_better: false
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: ""
metadata:
  version: 0.0
  source: "lmms-eval captionqa_{domain} (Borise/CaptionQA); scorer also logs each question's chosen option"
"""

CAPTIONQA_EXT = r'''"""CaptionQA scorer that also logs each question's chosen option (written by prepare_tts_tasks.py).

Data, prompt, option shuffling, judge prompt and metrics are lmms-eval's own (lmms_eval/tasks/captionqa/utils.py,
imported, not copied); captionqa_process_results below takes the same steps as lmms-eval's and adds, per
question, "chosen" (the original index of the option the judge picked, None when its answer had no letter) and
"n_choices". The judge's questions of one image are sent in parallel (CAPTIONQA_JUDGE_WORKERS).
"""
import concurrent.futures as cf
import os

from lmms_eval.tasks.captionqa import utils as U

captionqa_process_docs = U.captionqa_process_docs
captionqa_doc_to_visual = U.captionqa_doc_to_visual
captionqa_doc_to_text = U.captionqa_doc_to_text
captionqa_aggregate_score = U.captionqa_aggregate_score
captionqa_aggregate_accuracy = U.captionqa_aggregate_accuracy
captionqa_aggregate_cannot_answer = U.captionqa_aggregate_cannot_answer
WORKERS = max(1, int(os.environ.get("CAPTIONQA_JUDGE_WORKERS", "8")))


def questions_of(doc):
    questions = doc.get("questions", [])
    if not questions and "question" in doc:  # single question format (as lmms-eval)
        cat = doc.get("category", [])
        if isinstance(cat, list):
            cat = cat[0] if cat else ""
        questions = [{"question": doc["question"], "choices": doc.get("choices", []), "answer": doc.get("answer"),
                      "category": cat}]
    return questions


def prepared(doc):
    """[(q_idx, question, choices, gt_idx, options with 'cannot answer', permutation, category)] as lmms-eval builds
    them; questions lmms-eval skips (fewer than 2 choices, no ground-truth letter) are skipped here too."""
    out = []
    for q_idx, q in enumerate(questions_of(doc)):
        question_text, choices, answer = q.get("question", ""), q.get("choices", []), q.get("answer")
        category = q.get("category", [])
        if isinstance(category, list):
            category = category[0] if category else ""
        if not choices or len(choices) < 2:
            continue
        gt_letter = U.normalize_gt_letter(choices, answer)
        if gt_letter is None:
            continue
        opts = U.add_cannot_answer_option(question_text, choices)
        perm = U.get_shuffle_permutation(doc, q_idx, len(opts))
        out.append((q_idx, question_text, choices, U.LETTER_ALPH.index(gt_letter), opts, perm, category))
    return out


def captionqa_process_results(doc, results):
    caption = results[0] if results else ""
    image_id = doc.get("id", "unknown")
    items = prepared(doc)
    prompts = [U.build_caption_qa_prompt(caption, qt, [opts[i] for i in perm]) for _, qt, _, _, opts, perm, _ in items]
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        responses = list(ex.map(U.call_llm_judge, prompts))
    question_results = []
    for (q_idx, _, choices, gt_idx, opts, perm, category), response in zip(items, responses):
        n_opts = len(opts)
        letter = U.extract_letter(response, n_opts)
        is_correct = is_cannot_answer = False
        score, chosen = 0.0, None
        if letter is not None:
            shuf_idx = U.LETTER_ALPH.find(letter)
            if 0 <= shuf_idx < len(perm):
                orig_idx = perm[shuf_idx]
                chosen = orig_idx
                if orig_idx == n_opts - 1 and n_opts > len(choices):
                    is_cannot_answer = True
                    score = (1.0 / len(choices)) + 0.05
                elif orig_idx == gt_idx:
                    is_correct = True
                    score = 1.0
        question_results.append(dict(image_id=image_id, question_idx=q_idx, category=category, is_correct=is_correct,
                                     is_cannot_answer=is_cannot_answer, score=score, chosen=chosen,
                                     n_choices=len(choices)))
    return {"captionqa_score": question_results, "captionqa_accuracy": question_results,
            "captionqa_cannot_answer_rate": question_results}
'''

CHAIR_YAML = """task: chair500
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function chair500_utils.doc_to_visual
doc_to_text: !function chair500_utils.doc_to_text
doc_to_target: image
generation_kwargs:
  max_new_tokens: 4096
  temperature: 0
  top_p: 1.0
  num_beams: 1
  do_sample: false
  until: []
process_results: !function chair500_utils.process_results
metric_list:
  - metric: chair500_CHAIRs
    aggregation: !function chair500_utils.chair_s
    higher_is_better: false
  - metric: chair500_CHAIRi
    aggregation: !function chair500_utils.chair_i
    higher_is_better: false
  - metric: chair500_Recall
    aggregation: !function chair500_utils.recall
    higher_is_better: true
  - metric: chair500_Precision
    aggregation: !function chair500_utils.precision
    higher_is_better: true
  - metric: chair500_F1
    aggregation: !function chair500_utils.f1
    higher_is_better: true
metadata:
  version: 0.0
  source: "Self-Validation CHAIR-500 (SelfVal {commit}): {n} COCO val2014 images, scored by its chair.py + chair.pkl"
"""

CHAIR_UTILS = r'''"""Scorer of the chair500 task (written by prepare_tts_tasks.py): Self-Validation's chair.py (copied as
chair500/selfval_chair.py, unchanged) with its chair.pkl evaluator; metrics in percent.

Per caption, the evaluator's own caption_to_words and ground-truth objects, counted as its compute_chair loop
counts them (per_caption); the totals are its overall_metrics formulas. Only the division is ours: a set of
captions with no MSCOCO word at all gives NaN where compute_chair would raise ZeroDivisionError after hours of
generation. tts_smoke.py checks the numbers equal compute_chair's on real captions.

lmms-eval imports this file once per `!function` tag, so the evaluator and the scores are kept in sys.modules
(one copy for every metric) and computed once per set of captions.
"""
import hashlib
import json
import os
import pickle
import sys
import types

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["CHAIRs", "CHAIRi", "Recall", "Precision", "F1"]


def doc_to_visual(doc):
    return [Image.open(doc["image_file"]).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["text"]


def process_results(doc, results):
    rec = dict(image=doc["image"], caption=results[0])
    return {f"chair500_{m}": rec for m in METRICS}


def _shared():
    return sys.modules.setdefault("_chair500_shared", types.ModuleType("_chair500_shared")).__dict__


def _evaluator():
    sh = _shared()
    if "ev" not in sh:
        sys.path.insert(0, os.path.join(HERE, "chair500"))
        import selfval_chair as CH  # the official chair.py

        class Unpickler(pickle.Unpickler):  # chair.pkl was pickled from chair.py run as a script (__main__.CHAIR)
            def find_class(self, module, name):
                if name == "CHAIR":
                    return CH.CHAIR
                return super().find_class(module, name)
        with open(os.path.join(HERE, "chair500", "chair.pkl"), "rb") as f:
            sh["ev"] = Unpickler(f).load()
    return sh["ev"]


def image_id(name):
    return int(name.replace("COCO_val2014_", "").replace(".jpg", ""))  # as chair.py load_generated_captions


def per_caption(images, captions):
    """[hallucinated caption 0/1, hallucinated mentions, MSCOCO mentions, gt objects, gt objects mentioned,
    distinct objects mentioned, words], the counts of chair.py compute_chair's loop."""
    ev = _evaluator()
    rows = []
    for im, cap in zip(images, captions):
        words, node_words, idxs, raw_words = ev.caption_to_words(cap)
        gt = ev.imid_to_objects[image_id(im)]
        halluc = [nw for _, nw, _ in zip(words, node_words, idxs) if nw not in gt]
        recall = {nw for _, nw, _ in zip(words, node_words, idxs) if nw in gt}
        rows.append([int(bool(halluc)), len(halluc), len(node_words), len(gt), len(recall), len(set(node_words)),
                     len(raw_words)])
    return rows


def overall(rows):
    s = [sum(r[k] for r in rows) for k in range(7)]
    div = lambda a, b: a / b if b else float("nan")
    rec, prec = div(s[4], s[3]), div(s[4], s[5])
    return {"CHAIRs": 100 * div(s[0], len(rows)), "CHAIRi": 100 * div(s[1], s[2]), "Recall": 100 * rec,
            "Precision": 100 * prec, "F1": 100 * div(2 * rec * prec, rec + prec)}


def _scores(results):
    recs = sorted((r["image"], r["caption"]) for r in results)
    key = hashlib.sha1(json.dumps(recs, ensure_ascii=False).encode()).hexdigest()
    cache = _shared().setdefault("scores", {})
    if key not in cache:
        cache[key] = overall(per_caption([r[0] for r in recs], [r[1] for r in recs]))
    return cache[key]


def chair_s(results):
    return _scores(results)["CHAIRs"]


def chair_i(results):
    return _scores(results)["CHAIRi"]


def recall(results):
    return _scores(results)["Recall"]


def precision(results):
    return _scores(results)["Precision"]


def f1(results):
    return _scores(results)["F1"]
'''


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, path)
    return path


def selfval_file(a, name, dst):
    if a.selfval_dir:
        src = os.path.join(a.selfval_dir, "chair500", name)
        if not os.path.exists(src):
            raise SystemExit(f"[tasks] {src} not found (--selfval_dir is a clone of SelfVal)")
        shutil.copyfile(src, dst)
        return dst
    return fetch(SELFVAL_RAW + name, dst)


def find_coco(a, image_id):
    for d in a.coco_dirs:
        for name in (f"{image_id:012d}.jpg", f"COCO_val2014_{image_id:012d}.jpg"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None


def chair500(a):
    d = os.path.join(a.out_dir, "chair500")
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    qfile = selfval_file(a, "chair-500.jsonl", os.path.join(d, "chair-500.jsonl"))
    selfval_file(a, "chair.pkl", os.path.join(d, "chair.pkl"))
    selfval_file(a, "chair.py", os.path.join(d, "selfval_chair.py"))
    with open(qfile) as f:
        qs = [json.loads(l) for l in f if l.strip()]
    if a.mock:
        qs = qs[:a.mock_n]
    rows, local, downloaded = [], 0, 0
    for q in qs:
        image_id = int(q["image"].replace("COCO_val2014_", "").replace(".jpg", ""))
        if a.mock:
            import mock_data as M
            p = os.path.join(d, "images", f"{image_id:012d}.png")
            M.image(f"synth_{5000 + len(rows)}.png").save(p)
        else:
            p = find_coco(a, image_id)
            if p:
                local += 1
            else:
                p = fetch(COCO_URL.format(id=image_id), os.path.join(d, "images", f"COCO_val2014_{image_id:012d}.jpg"))
                downloaded += 1
        rows.append(dict(question_id=q["question_id"], image=q["image"], image_file=p, text=q["text"]))
    data = os.path.join(a.out_dir, "chair500.jsonl")
    with open(data, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(os.path.join(a.out_dir, "chair500.yaml"), "w") as f:
        f.write(CHAIR_YAML.format(data=data, commit=SELFVAL_COMMIT[:7], n=len(rows)))
    with open(os.path.join(a.out_dir, "chair500_utils.py"), "w") as f:
        f.write(CHAIR_UTILS)
    print(f"[tasks] chair500: {len(rows)} images ({local} local, {downloaded} downloaded); prompt {rows[0]['text']!r}",
          flush=True)


def captionqa(a):
    with open(os.path.join(a.out_dir, "captionqa_ext.py"), "w") as f:
        f.write(CAPTIONQA_EXT)
    for dom in CAPTIONQA_DOMAINS:
        with open(os.path.join(a.out_dir, f"captionqa_x_{dom}.yaml"), "w") as f:
            f.write(CAPTIONQA_YAML.format(domain=dom))
    print(f"[tasks] captionqa_x_{{{','.join(CAPTIONQA_DOMAINS)}}}: lmms-eval's CaptionQA + chosen option logged "
          f"(the data is downloaded by lmms-eval at the first run)", flush=True)


def amber(a):
    d = os.path.join(a.out_dir, "amber", "metadata")
    os.makedirs(d, exist_ok=True)
    for name in AMBER_FILES:
        fetch(AMBER_RAW + name, os.path.join(d, name))
    print(f"[tasks] AMBER metadata ({AMBER_COMMIT[:7]}): {d}; export AMBER_BASE_DIR={os.path.dirname(d)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True, help="the task folder of lmms_eval_run.py --include_path")
    ap.add_argument("--coco_dirs", default="/mnt/data2/shared_dataset/COCO-2017/val2017",
                    help="comma list of local COCO image folders (<id>.jpg or COCO_val2014_<id>.jpg)")
    ap.add_argument("--selfval_dir", default=None, help="a local SelfVal clone instead of downloading its 3 files")
    ap.add_argument("--only", default="captionqa,chair500,amber", help="which tasks to write")
    ap.add_argument("--mock", action="store_true", help="synthetic images for chair500, no downloads (smoke test)")
    ap.add_argument("--mock_n", type=int, default=4)
    a = ap.parse_args()
    a.out_dir = os.path.abspath(a.out_dir)
    a.coco_dirs = [x for x in a.coco_dirs.split(",") if x]
    os.makedirs(a.out_dir, exist_ok=True)
    only = set(a.only.split(","))
    if "captionqa" in only:
        captionqa(a)
    if "chair500" in only:
        chair500(a)
    if "amber" in only and not a.mock:
        amber(a)


if __name__ == "__main__":
    main()
