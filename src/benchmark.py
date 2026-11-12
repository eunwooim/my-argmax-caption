"""Benchmark a vision-language model on the DOCCI Positive / Negative / Hallucinated QA (generate.py).

The VLM sees the DOCCI image and the question (open-ended: "... Answer in a few words."; MCQ: the four
options) and never the captions. Every answer gets a label:
  correct      the true answer: the QA's answer for positive / negative, caption1_truth for hallucinated
  false_claim  the false claim Caption 2 made: the QA's answer for hallucinated, and for a contradiction
               negative the claim of the hallucinated QA linked to it
  wrong        anything else (invalid = no letter in an MCQ reply)
MCQ answers and exact matches are labeled here. Other open-ended answers are labeled by grade.py (an LLM
judge, one model load for every run), or by word matching with --grading match.

Scores, percent with 95% intervals from resampling images:
  positive accuracy; negative accuracy, also per neg_type (contradiction / omission);
  hallucinated: truth accuracy (said what Caption 1 says) and repeat rate (said Caption 2's false
  claim; lower is better); each also per coverage case (P+N+H, P+N, P+H, P) and per category.
--no_image asks the same questions without the image: how far language priors alone get.

Each run goes to <out_dir>/<tag>/ (predictions, grades, labels, results.md); leaderboard.md in
<out_dir> compares all runs there.
  python benchmark.py --qa out_qa/qa_final.jsonl --no_score          # answers (one shard per GPU)
  python grade.py --runs 'out_bench/*'                               # LLM judge for open-ended answers
  python benchmark.py --qa out_qa/qa_final.jsonl --score_only
Smoke test:  python generate.py --mock --out_dir out_mock && python benchmark.py --mock --qa out_mock/qa_final.jsonl --grading match
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import collections
import json
import os
import random
import re

import qa_common as C
import qa_prompts as P
from llm import LazyChat, add_llm_args, make_backend, resolve_defaults

CASES = ["P+N+H", "P+N", "P+H", "P"]


# ------------------------------------------------------------------ answers and labels
def targets(q):
    """(truth, false claim, truth letter, false-claim letter); the letters only for MCQ."""
    if q["set"] == "hallucinated":
        truth, false = q["caption1_truth"], q["answer_text"]
    else:
        truth, false = q["answer_text"], q.get("caption2_claim")
    if q["format"] != "mcq":
        return truth, false, None, None
    if q["set"] == "hallucinated":  # the truth is an option only with generate.py --hal_mcq_truth
        return truth, false, q.get("truth_letter"), q["answer"]
    fl = next((k for k, v in q["options"].items() if false and C.norm_text(v) == C.norm_text(false)), None)
    return truth, false, q["answer"], fl


def parse_letter(text):
    t = C.strip_think(text).strip()
    m = re.match(r"^\(?([A-D])\b", t) or re.search(r"\b(?:answer|option)\s*(?:is)?\s*:?\s*\(?([A-D])\b", t, re.I) \
        or re.search(r"\b([A-D])[.)]", t)
    return m.group(1) if m else None


def short_answer(text):
    t = C.strip_think(text).strip()
    t = next((l.strip() for l in t.splitlines() if l.strip()), "")
    return t.rstrip(".").strip()[:200]


_ART = re.compile(r"\b(a|an|the|some)\b")


def _norm(s):
    toks = _ART.sub(" ", C.norm_text(s)).split()
    return " ".join(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks)


def quick_label(pred, truth, false):
    """Label without a judge when the answer is exactly the truth or the claim (or empty), else None."""
    p = _norm(pred)
    if not p:
        return "wrong"
    if p == _norm(truth):
        return "correct"
    if false and p == _norm(false):
        return "false_claim"
    return None


def match_label(pred, truth, false):
    """Word matching (--grading match): the truth / the claim appears in the answer as a phrase."""
    p = f" {_norm(pred)} "
    t = bool(truth) and f" {_norm(truth)} " in p
    f = bool(false) and f" {_norm(false)} " in p
    return "correct" if t and not f else "false_claim" if f and not t else "wrong"


def label(p, grades, grading):
    """correct / false_claim / wrong / invalid, or None while an open-ended answer waits for grade.py."""
    if p["format"] == "mcq":
        if p["pred"] is None:
            return "invalid"
        if p["truth_letter"] and p["pred"] == p["truth_letter"]:
            return "correct"
        if p["false_letter"] and p["pred"] == p["false_letter"]:
            return "false_claim"
        return "wrong"
    lab = quick_label(p["pred"], p["truth"], p["false"])
    if lab:
        return lab
    if grading == "match":
        return match_label(p["pred"], p["truth"], p["false"])
    return grades.get(p["qa_id"])


def run_labels(tag_dir, grading):
    preds = C.read_shards(os.path.join(tag_dir, "predictions.jsonl"))
    grades = {r["qa_id"]: r["label"] for r in C.read_shards(os.path.join(tag_dir, "grades.jsonl")) if r.get("label")}
    return {p["qa_id"]: label(p, grades, grading) for p in preds}, {p["qa_id"]: p for p in preds}


# ------------------------------------------------------------------ predict
def run_tag(a):
    if a.tag:
        return a.tag
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(a.model.rstrip("/")))
    return base + ("__noimage" if a.no_image else "")


def predict(a, qa, chat, tag_dir):
    base = os.path.join(tag_dir, "predictions.jsonl")
    path = C.shard_path(a, base)
    done = {r["qa_id"] for r in C.read_shards(base)}
    todo = [q for q in qa if q["qa_id"] not in done and C.in_shard(a, q["example_id"])]
    print(f"[bench] {len(todo)} questions to answer, {len(done)} done", flush=True)
    if not todo:
        return
    paths = {} if (a.no_image or a.mock) else \
        C.docci_image_paths(sorted({q["image_file"] for q in todo}), a.cache_dir, a.docci_dir)
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        # the same question on the same image (a contradiction and its hallucinated twin) is asked once
        asks = collections.OrderedDict()
        for q in chunk:
            asks.setdefault((q["image_file"], P.bench_text(q)), []).append(q)
        urls = {}
        if not a.no_image:
            for q in chunk:
                if q["image_file"] not in urls:
                    src = mock_image(q) if a.mock else paths[q["image_file"]]
                    urls[q["image_file"]] = C.image_data_url(src, a.max_side)
        convs = []
        for (img, _), qs in asks.items():
            m = P.bench_messages(qs[0], None if a.no_image else urls[img])
            if a.mock:
                m[-1]["example_id"] = qs[0]["example_id"]
            convs.append(m)
        outs = chat(convs, temperature=0.0)
        rows = []
        for qs, o in zip(asks.values(), outs):
            for q in qs:
                truth, false, tl, fl = targets(q)
                rows.append(dict(qa_id=q["qa_id"], example_id=q["example_id"], set=q["set"], format=q["format"],
                                 question=P.bench_text(q), truth=truth, false=false, truth_letter=tl,
                                 false_letter=fl, reply=o,
                                 pred=parse_letter(o) if q["format"] == "mcq" else short_answer(o)))
        C.append_jsonl(path, rows)
        print(f"[bench] {min(s + a.batch, len(todo))}/{len(todo)} ({len(convs)} distinct questions)", flush=True)


# ------------------------------------------------------------------ scoring
def score(qa, labels, B, seed):
    have = [q for q in qa if q["qa_id"] in labels]
    rows = [(q, labels[q["qa_id"]]) for q in have if labels[q["qa_id"]] is not None]

    def ci(sel, what="correct"):
        sel = list(sel)
        return C.bootstrap([l == what for _, l in sel], [q["example_id"] for q, _ in sel], B, seed)

    def block(sel):
        pos = [x for x in sel if x[0]["set"] == "positive"]
        neg = [x for x in sel if x[0]["set"] == "negative"]
        hal = [x for x in sel if x[0]["set"] == "hallucinated"]
        return dict(
            positive=ci(pos), negative=ci(neg),
            contradiction=ci(x for x in neg if x[0].get("neg_type") == "contradiction"),
            omission=ci(x for x in neg if x[0].get("neg_type") == "omission"),
            hal_truth=ci(x for x in hal if x[0]["format"] != "mcq" or x[0].get("truth_letter")),
            hal_repeat=ci(hal, "false_claim"),
            contra_repeat=ci((x for x in neg if x[0].get("caption2_claim")), "false_claim"))

    res = dict(n=len(rows), n_images=len({q["example_id"] for q, _ in rows}),
               ungraded=len(have) - len(rows), missing=len(qa) - len(have),
               invalid=sum(l == "invalid" for _, l in rows) / max(len(rows), 1),
               labels=dict(collections.Counter(f"{q['set']}:{l}" for q, l in rows)),
               pn_accuracy=ci(x for x in rows if x[0]["set"] != "hallucinated"), **block(rows))
    res["case"] = {c: block([x for x in rows if x[0].get("coverage") == c]) for c in CASES}
    res["case_images"] = {c: len({q["example_id"] for q, _ in rows if q.get("coverage") == c}) for c in CASES}
    res["category"] = {c: block([x for x in rows if x[0]["category"] == c])
                       for c in sorted({q["category"] for q, _ in rows})}
    res["format"] = {f: block([x for x in rows if x[0]["format"] == f]) for f in sorted({q["format"] for q, _ in rows})}
    return res


def fmt(d):
    if d["n"] == 0:
        return "–"
    return f"{100 * d['mean']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}] (n={d['n']})"


def pct(d):
    return "–" if d["n"] == 0 else f"{100 * d['mean']:.1f}"


BLOCK_COLS = [("positive", "positive"), ("negative", "negative"), ("contradiction", "contradiction"),
              ("omission", "omission"), ("hal_truth", "hal: truth"), ("hal_repeat", "hal: repeats ↓")]


def write_results(tag_dir, tag, res):
    with open(os.path.join(tag_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    L = [f"# {tag}", "", f"{res['n']} questions on {res['n_images']} images. Percent, 95% interval over images."]
    if res["ungraded"]:
        L += ["", f"**{res['ungraded']} open-ended answers are not labeled yet** (run grade.py, or score with "
                  f"--grading match); they are left out below."]
    L += ["", "| metric | value |", "|---|---|",
          f"| positive accuracy | {fmt(res['positive'])} |",
          f"| negative accuracy | {fmt(res['negative'])} |",
          f"| negative: contradiction | {fmt(res['contradiction'])} |",
          f"| negative: omission | {fmt(res['omission'])} |",
          f"| hallucinated: truth accuracy | {fmt(res['hal_truth'])} |",
          f"| hallucinated: repeats Caption 2's claim ↓ | {fmt(res['hal_repeat'])} |",
          f"| contradiction: repeats Caption 2's claim ↓ | {fmt(res['contra_repeat'])} |",
          f"| positive + negative accuracy | {fmt(res['pn_accuracy'])} |",
          f"| invalid replies | {100 * res['invalid']:.1f} |"]
    head = "| {} | " + " | ".join(n for _, n in BLOCK_COLS) + " |"
    sep = "|---|" + "---|" * len(BLOCK_COLS)
    L += ["", "## Per coverage case", "", head.format("case (images)"), sep]
    L += [f"| {c} ({res['case_images'][c]}) | " + " | ".join(pct(b[k]) for k, _ in BLOCK_COLS) + " |"
          for c, b in res["case"].items()]
    L += ["", "## Per category", "", head.format("category"), sep]
    L += [f"| {c} | " + " | ".join(pct(b[k]) for k, _ in BLOCK_COLS) + " |" for c, b in res["category"].items()]
    with open(os.path.join(tag_dir, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def write_leaderboard(out_dir):
    L = ["# DOCCI P/N/H QA leaderboard", "",
         "| run | " + " | ".join(n for _, n in BLOCK_COLS) + " | contra: repeats ↓ | n |",
         "|---|" + "---|" * (len(BLOCK_COLS) + 2)]
    for t in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, t, "results.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            L.append(f"| {t} | " + " | ".join(pct(r[k]) for k, _ in BLOCK_COLS) +
                     f" | {pct(r['contra_repeat'])} | {r['n']} |")
    with open(os.path.join(out_dir, "leaderboard.md"), "w") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------------ mock VLM (smoke tests only)
def mock_image(q):
    import mock_docci
    return mock_docci.image(q["example_id"]) if q["example_id"].startswith("synth_") \
        else mock_docci.image("synth_0")


def make_mock_vlm(qa, seed, blind):
    """Right most of the time, sometimes repeats Caption 2's claim: just so the numbers move."""
    known = {(q["example_id"], P.bench_text(q)): q for q in qa}

    def fn(msgs):
        c = msgs[-1]["content"]
        key = (msgs[-1]["example_id"], c if isinstance(c, str) else c[-1]["text"])
        q = known[key]
        rng = random.Random(f"{seed}|{blind}|{key}")  # same answer whatever the sharding
        truth, false, tl, fl = targets(q)
        right = 0.35 if blind else {"positive": 0.9, "negative": 0.65, "hallucinated": 0.6}[q["set"]]
        r = rng.random()
        if q["format"] == "mcq":
            return tl if r < right and tl else fl if fl and r < right + 0.25 else rng.choice(P.LETTERS)
        if r < right:
            return rng.choice([truth.capitalize() + ".", f"It is {truth}."])
        if false and r < right + 0.25:
            return false.capitalize() + "."
        return "I am not sure."
    return fn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, help="qa_final.jsonl from generate.py")
    ap.add_argument("--mock", action="store_true", help="mock VLM, synthetic images, no GPU")
    ap.add_argument("--no_image", action="store_true", help="blind baseline: questions without the image")
    ap.add_argument("--sets", default="positive,negative,hallucinated", help="only these QA sets")
    ap.add_argument("--splits", default=None, help="only these DOCCI splits, comma separated")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick runs)")
    ap.add_argument("--max_side", type=int, default=1280, help="resize images so the longest side is at most this")
    ap.add_argument("--batch", type=int, default=512, help="questions per write to disk")
    ap.add_argument("--grading", default="judge", choices=["judge", "match"],
                    help="open-ended answers that are not an exact match: judge = labels from grade.py, "
                         "match = word matching (no model)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no_score", action="store_true", help="only write predictions (shards; score once at the end)")
    ap.add_argument("--score_only", action="store_true", help="only score the predictions already on disk")
    ap.add_argument("--tag", default=None, help="run name (default: model name, + __noimage)")
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"))
    ap.add_argument("--docci_dir", default=None, help="folder with images/ (else streamed from the DOCCI release)")
    ap.add_argument("--out_dir", default="out_bench")
    add_llm_args(ap, "Qwen/Qwen2.5-VL-7B-Instruct", 32)
    C.add_gpu_args(ap)
    C.add_shard_args(ap)
    a = ap.parse_args()
    if a.mock:
        a.backend, a.model = "mock", "mock"
    resolve_defaults(a, 32)

    qa = [q for q in C.read_jsonl(a.qa) if q["set"] in a.sets.split(",")]
    if a.splits:
        qa = [q for q in qa if q["split"] in a.splits.split(",")]
    if a.limit:
        keep = set(sorted({q["example_id"] for q in qa})[:a.limit])
        qa = [q for q in qa if q["example_id"] in keep]
    if not a.mock:
        C.setup_caches()
    tag = run_tag(a)
    tag_dir = os.path.join(a.out_dir, tag)
    print(f"[paths] results: {os.path.abspath(tag_dir)}   DOCCI: {os.path.abspath(a.cache_dir)}", flush=True)
    os.makedirs(tag_dir, exist_ok=True)

    if not a.score_only:
        def load():
            if a.backend == "vllm":
                C.pick_gpus(a)
            return make_backend(a, mock_fn=make_mock_vlm(qa, a.seed, a.no_image),
                                limit_images=0 if a.no_image else 1)
        predict(a, qa, LazyChat(load), tag_dir)
    if a.no_score:
        return
    labels, _ = run_labels(tag_dir, a.grading)
    C.write_jsonl(os.path.join(tag_dir, "labels.jsonl"),
                  [dict(qa_id=k, label=v) for k, v in sorted(labels.items())])
    write_results(tag_dir, tag, score(qa, labels, a.bootstrap, a.seed))
    write_leaderboard(a.out_dir)


if __name__ == "__main__":
    main()
