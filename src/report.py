"""summary.md + summary.html for a run folder, everything run_all.py made on one page (with a contents list):
the captioner results (Step 1, GRPO before / after), each Caption 2's length and cut-offs, how many mistakes
it has (before vs after RL) and the same photos captioned before and after, the Positive / Negative /
Hallucinated datasets and their checks (generated -> kept, why dropped), which sets each image has
(coverage case), every benchmark, and sample images for every coverage case in one identical layout (photo, Caption 1, Caption 2, coverage, then the Positive, Negative and Hallucinated
tables with each VLM's answer; an empty set still gets its table, marked none).
summary.html is one self-contained file (photos embedded): open it in any browser.

run_all.py writes both at the end. To rebuild them from results already on disk (no GPU, seconds):
  python report.py --out_root runs
  python report.py --out_root runs --samples_per_case 4
  python report.py --out_root runs --notes analysis_ko.md     # put your own notes / analysis on top
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import base64
import html
import json
import os
import random
import re
import traceback

import qa_common as C

BLIND = "__noimage"
CASES = ["P+N+H", "P+N", "P+H", "P"]
CASE_NAMES = {"P+N+H": "all three sets", "P+N": "positive + negative", "P+H": "positive + hallucinated",
              "P": "positive only", "dropped": "no positive QA: dropped, not in the benchmark"}
SET_NOTE = {"positive": "answerable from both captions",
            "negative": "answer from Caption 1; Caption 2 contradicts it or leaves it out",
            "hallucinated": "the answer is Caption 2's false claim; Caption 1 says otherwise"}
MARK = {"correct": "✅", "false_claim": "⚠️ repeats claim", "wrong": "❌", "invalid": "? no letter"}


def datasets(out_root, order=None):
    """Dataset tags (qa_<tag> folders), in run order when given."""
    def key(t):  # <generator>__<captioner>__<prompt>: by generator, prompt, then before / after
        p = t.split("__")
        return (p[0], p[2], {"before": 0, "after": 1}.get(p[1], 2), p[1]) if len(p) == 3 else (t, "", 0, "")
    found = sorted((d[3:] for d in os.listdir(out_root) if d.startswith("qa_") and os.path.isdir(os.path.join(out_root, d))),
                   key=key)
    return [t for t in (order or []) if t in found] + [t for t in found if t not in (order or [])]


def bench_runs(out_root, ds):
    d = os.path.join(out_root, f"bench_{ds}")
    return sorted(t for t in os.listdir(d) if os.path.isdir(os.path.join(d, t))) if os.path.isdir(d) else []


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pct(x):
    return f"{100 * x:.1f}" if x is not None and x == x else "–"


def m(d):
    return pct(d["mean"]) if d and d.get("n") else "–"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------ tables
def dataset_table(out_root, dss):
    L = ["## Datasets", "",
         "Kept = images with at least one positive QA. P+N+H … P = which sets a kept image has.", "",
         "| dataset | Caption 2 by | Caption 2 prompt | format | images | kept | P+N+H | P+N | P+H | P only | dropped | "
         "positive QA | negative QA | contradiction | omission | hallucinated QA | judges |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in dss:
        s = load_json(os.path.join(out_root, f"qa_{ds}", "stats.json"))
        if s is None:
            L.append(f"| {ds} | " + " | ".join(["–"] * 16) + " |")
            continue
        cov, q, nt = s["coverage"], s["qa_per_set"], s["neg_type"]
        drop = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in s["dropped_images"].items()) or "0"
        L.append(f"| {ds} | {s['caption2_model']} | {s.get('caption2_prompt', '?')} | {s['qa_format']} | "
                 f"{s['images_in']} | {s['images']} | "
                 + " | ".join(str(cov.get(c, 0)) for c in CASES)
                 + f" | {drop} | {q['positive']} | {q['negative']} | {nt['contradiction']} | {nt['omission']} | "
                   f"{q['hallucinated']} | {', '.join(s['judges']) or 'none'} |")
    return L


def coverage_tables(out_root, dss):
    L = ["## Coverage: which sets each image has", ""]
    for ds in dss:
        s = load_json(os.path.join(out_root, f"qa_{ds}", "stats.json"))
        if s is None:
            continue
        L += [f"**{ds}**", "", "| case | meaning | images | share | positive QA | negative QA | hallucinated QA |",
              "|---|---|---|---|---|---|---|"]
        for c in CASES:
            n = s["coverage"].get(c, 0)
            q = s["coverage_qa"].get(c, {"positive": 0, "negative": 0, "hallucinated": 0})
            L.append(f"| {c} | {CASE_NAMES[c]} | {n} | {pct(n / max(s['images'], 1))} | {q['positive']} | "
                     f"{q['negative']} | {q['hallucinated']} |")
        nd = s["dropped_images"].get("no_positive", 0)
        had = ", ".join(f"{k} {v}" for k, v in s["dropped_no_positive_cases"].items()) or "–"
        L += [f"| no positive | {CASE_NAMES['dropped']} (had: {had}) | {nd} | – | 0 | – | – |", ""]
    return L


COLS = [("positive", "positive"), ("negative", "negative"), ("contradiction", "contradiction"),
        ("omission", "omission"), ("hal_truth", "hal: truth"), ("hal_repeat", "hal: repeats ↓"),
        ("contra_repeat", "contra: repeats ↓")]


def bench_table(out_root, dss):
    L = ["## Benchmarks", "",
         "Percent right. positive / negative = said the QA's answer. hal: truth = on a hallucinated QA said what "
         "Caption 1 says; hal: repeats ↓ = said Caption 2's false claim (lower is better); contra: repeats ↓ = "
         f"the same on contradiction negatives. `{BLIND}` = same questions without the image.", "",
         "| dataset (generator) | VLM | " + " | ".join(n for _, n in COLS) + " | n | not graded |",
         "|---|---|" + "---|" * (len(COLS) + 2)]
    for ds in dss:
        for run in bench_runs(out_root, ds):
            r = load_json(os.path.join(out_root, f"bench_{ds}", run, "results.json"))
            if r:
                L.append(f"| {ds} | {run} | " + " | ".join(m(r[k]) for k, _ in COLS) + f" | {r['n']} | {r['ungraded']} |")
    return L


def per_vlm_tables(out_root, dss, key, title, note):
    vlms = sorted({run for ds in dss for run in bench_runs(out_root, ds) if not run.endswith(BLIND)})
    L = [f"## {title}", "", note]
    for v in vlms:
        for ds in dss:
            r = load_json(os.path.join(out_root, f"bench_{ds}", v, "results.json"))
            if not r:
                continue
            L += ["", f"**{v}** on {ds}", "", f"| {key} | " + " | ".join(n for _, n in COLS[:-1]) + " |",
                  "|---|" + "---|" * (len(COLS) - 1)]
            for c, b in r["case" if key == "case" else "category"].items():
                k = r["case_images"][c] if key == "case" else 0
                lab = f"{c} ({k} image{'' if k == 1 else 's'})" if key == "case" else c
                L.append(f"| {lab} | " + " | ".join(m(b[k]) for k, _ in COLS[:-1]) + " |")
    return L


# ------------------------------------------------------------------ samples
def thumbnail(ex_id, image_file, out_root, cache_dir, docci_dir, mock, max_side=640):
    """samples/<id>.jpg next to summary.md, from images already on disk (never downloads)."""
    from PIL import Image
    dst = os.path.join(out_root, "samples", f"{ex_id}.jpg")
    if os.path.exists(dst):
        return dst
    src = os.path.join(docci_dir or os.path.join(cache_dir, "docci"), "images", image_file)
    if os.path.exists(src):
        img = Image.open(src).convert("RGB")
    elif mock or ex_id.startswith("synth_"):
        import mock_docci
        img = mock_docci.image(ex_id)
    else:
        return None
    img.thumbnail((max_side, max_side))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, quality=85)
    return dst


def load_answers(out_root, ds):
    """{run: {qa_id: (answer text, label)}} from each run's predictions and labels."""
    out = {}
    for run in bench_runs(out_root, ds):
        d = os.path.join(out_root, f"bench_{ds}", run)
        labels = {r["qa_id"]: r["label"] for r in C.read_jsonl(os.path.join(d, "labels.jsonl"))}
        out[run] = {p["qa_id"]: (p["pred"], labels.get(p["qa_id"]))
                    for p in C.read_shards(os.path.join(d, "predictions.jsonl"))}
    return out


def cell(ans, q, full=False):
    """full=True keeps the whole answer (sample_html.py); otherwise cut to 40 characters."""
    if ans is None:
        return "–"
    pred, lab = ans
    text = esc((pred or "").strip()[:None if full else 40]) or "(empty)"
    if q["format"] == "mcq" and pred in (q.get("options") or {}):
        text = f"{pred} ({esc(q['options'][pred])[:None if full else 30]})"
    return f"{text} {MARK.get(lab, '… not graded')}"


def question_text(q):
    t = esc(q["question"])
    if q.get("options"):
        t += " — " + " · ".join(f"{k}. {esc(v)}" for k, v in q["options"].items())
    if q.get("question_raw"):
        t += f" *(generated as: “{esc(q['question_raw'])}”)*"
    return t


def answer_text(q, key="answer"):
    if key == "answer" and q.get("options"):
        return f"**{q['answer']}** ({esc(q['answer_text'])})"
    return f"**{esc(q.get(key, ''))}**" if q.get(key) else "–"


def set_table(kind, items, gen_n, runs, answers, full=False):
    """The same table for every image and case; an empty set gets one 'none' row."""
    extra = {"positive": [], "negative": ["neg_type"], "hallucinated": []}[kind]
    ans_head = {"positive": "answer", "negative": "answer (Caption 1)", "hallucinated": "answer (Caption 2's claim)"}[kind]
    after = {"positive": [], "negative": ["Caption 2 says"], "hallucinated": ["Caption 1 truth"]}[kind]
    heads = ["id"] + extra + ["category", "format", "question", ans_head] + after + runs
    L = [f"**{kind.capitalize()} ({len(items)})**: {SET_NOTE[kind]}", "",
         "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for q in items:
        row = [q.get("id") or q["qa_id"]] + [q.get(x, "–") for x in extra] + [q["category"], q["format"],
                                                                              question_text(q), answer_text(q)]
        if kind == "negative":
            row.append(esc(q["caption2_claim"]) if q.get("caption2_claim") else
                       ("(not mentioned)" if q.get("neg_type") == "omission" else "–"))
        if kind == "hallucinated":
            row.append(answer_text(q, "caption1_truth"))
        row += [cell(answers.get(r, {}).get(q["qa_id"]), q, full) for r in runs]
        L.append("| " + " | ".join(str(x) for x in row) + " |")
    if not items:
        why = f"generated {gen_n}, all dropped by the checks" if gen_n else "the generator found none"
        L.append("| " + " | ".join([""] * (len(extra) + 3) + [f"none ({why})"] + [""] * (1 + len(after) + len(runs))) + " |")
    return L + [""]


def image_block(im, runs, answers, out_root, cache_dir, docci_dir, mock, full=False):
    e, cov = im["example_id"], im.get("coverage") or {}
    L = [f"#### {e} ({im['split']})", ""]
    th = thumbnail(e, im["image_file"], out_root, cache_dir, docci_dir, mock)
    L += [f"![{e}](samples/{os.path.basename(th)})" if th else f"*(image {im['image_file']} not on disk)*", ""]
    L += ["| caption | text |", "|---|---|",
          f"| Caption 1 (human, ground truth) | {esc(im['caption1'])} |",
          f"| Caption 2 ({esc(im.get('caption2_model') or '?')}, {im.get('caption2_prompt', '?')} prompt) | "
          f"{esc(im.get('caption2') or '(none)')} |", ""]
    n, g, nt = cov.get("n", {}), cov.get("n_generated", {}), cov.get("neg_types", {})
    status = "in the benchmark" if im.get("reason") is None else f"dropped: {im['reason'].replace('_', ' ')}"
    L += [f"Coverage **{cov.get('case', 'none')}** ({status}): "
          f"positive {'✓' if n.get('positive') else '✗'} {n.get('positive', 0)} · "
          f"negative {'✓' if n.get('negative') else '✗'} {n.get('negative', 0)} "
          f"(contradiction {nt.get('contradiction', 0)}, omission {nt.get('omission', 0)}) · "
          f"hallucinated {'✓' if n.get('hallucinated') else '✗'} {n.get('hallucinated', 0)}. "
          f"Generated before the checks: {g.get('positive', 0)} / {g.get('negative', 0)} / {g.get('hallucinated', 0)}.", ""]
    for kind in ["positive", "negative", "hallucinated"]:
        L += set_table(kind, im.get(kind, []), g.get(kind, 0), runs, answers, full)
    return L


def sample_section(out_root, ds, cache_dir, docci_dir, per_case, seed, mock):
    data = load_json(os.path.join(out_root, f"qa_{ds}", "qa_dataset.json"))
    if not data:
        return [f"## Samples by coverage case: {ds}", "", "No qa_dataset.json yet."]
    answers = load_answers(out_root, ds)
    runs = list(answers)
    groups = {c: [im for im in data["images"] if im["coverage"]["case"] == c] for c in CASES}
    groups["dropped"] = [im for im in data["dropped_images"] if im["reason"] == "no_positive"]
    rng = random.Random(seed)
    L = [f"## Samples by coverage case: {ds}", "",
         f"Up to {per_case} images per case (random, seed {seed}), all in the same layout: photo, both captions, "
         "coverage, then the Positive, Negative and Hallucinated tables. A set the image does not have still "
         "gets its table with a *none* row. VLM cells: ✅ true answer · ⚠️ repeats Caption 2's false claim · "
         "❌ other wrong answer · … waiting for grade.py.", ""]
    for c in CASES + ["dropped"]:
        ims = groups[c]
        title = "No positive (dropped)" if c == "dropped" else c
        L += [f"### {title}: {CASE_NAMES[c]} ({len(ims)} image{'' if len(ims) == 1 else 's'})", ""]
        if not ims:
            L += ["No images in this case.", ""]
            continue
        for im in sorted(rng.sample(ims, min(per_case, len(ims))), key=lambda x: x["example_id"]):
            L += image_block(im, runs if c != "dropped" else [], answers, out_root, cache_dir, docci_dir, mock)
    return L


def header(out_root, dss):
    L = ["# Captioner (GRPO) + DOCCI Positive / Negative / Hallucinated QA: all runs", "",
         "Caption 1 = DOCCI human description (ground truth). Caption 2 = the captioner's caption of the same image: "
         "*before* RL (Qwen2.5-VL, LoRA off) or *after* RL (GRPO LoRA on), from the caption prompt *simple* "
         "(\"Describe this image in detail.\"). "
         "Each gives its own dataset, named `<generator>__<captioner>__<prompt>`.",
         "", "- **Positive**: answerable from both captions (same answer).",
         "- **Negative**: answer from Caption 1; Caption 2 contradicts it (*contradiction*) or leaves it out "
         "(*omission*).",
         "- **Hallucinated**: the answer is a false claim of Caption 2; *Caption 1 truth* is what Caption 1 says.",
         "- An image stays in the dataset when it has at least one positive QA. Its **coverage case** says which "
         "sets it has: P+N+H, P+N, P+H or P (positive only). qa_dataset.json records it per image."]
    caps = {(load_json(os.path.join(out_root, f"qa_{ds}", "stats.json")) or {}).get("caption2_model") for ds in dss}
    vlms = {run for ds in dss for run in bench_runs(out_root, ds) if not run.endswith(BLIND)}
    base = lambda c: re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(c.split()[0]))
    same = sorted({c.split()[0] for c in caps if c and not c.startswith("mock") and base(c) in vlms})
    if same:
        L += ["", f"Note: the benchmarked VLM {', '.join(same)} is also the captioner's base model, so the negatives and "
                  "hallucinated QA of the *before* datasets are its own mistakes (expect it to score low on them)."]
    return L + [""]


def captioner_section(out_root, dirs):
    """Step 1 and GRPO results.md (headings moved down a level) and their plots, as they wrote them."""
    L = []
    for title, d in dirs:
        p = os.path.join(d, "results.md")
        if not os.path.exists(p):
            L += [f"## {title}", "", f"No results yet in `{d}`.", ""]
            continue
        with open(p) as f:
            lines = f.read().splitlines()
        L += [f"## {title}", "", f"From `{p}`.", ""]
        for l in lines:
            m = re.match(r"^(#{1,3}) (.*)", l)
            L.append(f"{'#' * min(len(m.group(1)) + 2, 4)} {m.group(2)}" if m else l)
        for png in ["docci_curve.png", "train_curves.png"]:
            if os.path.exists(os.path.join(d, png)):
                L += ["", f"![{png}]({os.path.relpath(os.path.join(d, png), out_root)})"]
        L.append("")
    return L


def caption2_table(out_root, dss):
    """Mistakes per image in each Caption 2, read off its QA: before vs after RL side by side."""
    L = ["## Caption 2 mistakes per image (before vs after RL)", "",
         "Per kept image. contradiction / hallucinated = Caption 2 said something false, omission = it left a "
         "Caption 1 fact out. Fewer after RL = the GRPO captioner hallucinates / omits less.", "",
         "| dataset | captioner | prompt | images | Caption 2 words | positive | contradiction | omission | "
         "hallucinated | images with a hallucination ↓ |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for ds in dss:
        s = load_json(os.path.join(out_root, f"qa_{ds}", "stats.json"))
        data = load_json(os.path.join(out_root, f"qa_{ds}", "qa_dataset.json"))
        if not s or not data:
            continue
        parts = ds.split("__")
        cap = parts[1] if len(parts) == 3 else "–"
        n = max(s["images"], 1)
        words = sum(len((im.get("caption2") or "").split()) for im in data["images"]) / n
        withh = s["coverage"].get("P+N+H", 0) + s["coverage"].get("P+H", 0)
        rows.append(((parts[0], s.get("caption2_prompt", "?"), {"before": 0, "after": 1}.get(cap, 2), cap),
                     ds, cap, s, words, n, withh))
    for _, ds, cap, s, words, n, withh in sorted(rows, key=lambda r: r[0]):
        q, nt = s["qa_per_set"], s["neg_type"]
        L.append(f"| {ds} | {cap} | {s.get('caption2_prompt', '?')} | {s['images']} | {words:.0f} | "
                 f"{q['positive'] / n:.2f} | {nt['contradiction'] / n:.2f} | {nt['omission'] / n:.2f} | "
                 f"{q['hallucinated'] / n:.2f} | {pct(withh / n)} |")
    return L


CAP_ORDER = {"before": 0, "after": 1}


def caption2_files(out_root):
    """{(captioner, prompt): rows} from <out_root>/caption2_<captioner>__<prompt>/caption2*.jsonl."""
    out = {}
    for d in os.listdir(out_root):
        m_ = re.fullmatch(r"caption2_(.+?)__(.+)", d)
        if m_ and os.path.isdir(os.path.join(out_root, d)):
            rows = [r for r in C.read_shards(os.path.join(out_root, d, "caption2.jsonl")) if r.get("caption") is not None]
            if rows:
                out[(m_.group(1), m_.group(2))] = rows
    return dict(sorted(out.items(), key=lambda kv: (kv[0][1], CAP_ORDER.get(kv[0][0], 2), kv[0][0])))


def caption2_stats(caps):
    L = ["## Caption 2: length and cut-offs", "",
         "One row per Caption 2 file. *cut off* = the caption hit the max-new-tokens limit before it ended "
         "(only caption_docci.py records it).", "",
         "| captioner | prompt | model | images | words per caption | tokens per caption | cut off at the limit ↓ |",
         "|---|---|---|---|---|---|---|"]
    for (cap, v), rows in caps.items():
        n = len(rows)
        words = sum(len(r["caption"].split()) for r in rows) / n
        toks = [r["n_tokens"] for r in rows if "n_tokens" in r]
        cut = [r["truncated"] for r in rows if "truncated" in r]
        L.append(f"| {cap} | {v} | {esc(rows[0].get('model', '?'))} | {n} | {words:.0f} | "
                 f"{sum(toks) / len(toks):.0f} | {pct(sum(cut) / len(cut))} |" if toks and cut else
                 f"| {cap} | {v} | {esc(rows[0].get('model', '?'))} | {n} | {words:.0f} | – | – |")
    return L + [""]


def image_mistakes(out_root, dss):
    """{(captioner, prompt): (generator, {example_id: (contradiction, omission, hallucinated) or None}, images)}
    from the first generator's dataset of each Caption 2."""
    out = {}
    for ds in dss:
        parts = ds.split("__")
        if len(parts) != 3 or (parts[1], parts[2]) in out:
            continue
        data = load_json(os.path.join(out_root, f"qa_{ds}", "qa_dataset.json"))
        if not data:
            continue
        mm, ims = {}, {}
        for im in data["images"] + data["dropped_images"]:
            cov = im.get("coverage") or {}
            nt, n = cov.get("neg_types") or {}, cov.get("n") or {}
            mm[im["example_id"]] = (nt.get("contradiction", 0), nt.get("omission", 0), n.get("hallucinated", 0)) \
                if cov else None
            ims[im["example_id"]] = im
        out[(parts[1], parts[2])] = (parts[0], mm, ims)
    return out


def caption2_examples(out_root, dss, caps, n_show, seed, cache_dir, docci_dir, mock):
    """The same photos captioned by every captioner and prompt, next to Caption 1, with the mistakes the QA found."""
    if len(caps) < 2:
        return []
    by = {k: {r["example_id"]: r for r in rows} for k, rows in caps.items()}
    mist = image_mistakes(out_root, dss)
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    pick = sorted(random.Random(seed).sample(common, min(n_show, len(common))))
    gen = next((g for g, _, _ in mist.values()), None)
    L = ["## Caption 2 examples: the same photo before vs after RL", "",
         f"{len(pick)} random images (seed {seed}), each with every Caption 2 next to Caption 1. Mistake counts are the "
         f"negative (contradiction, omission) and hallucinated QA that {gen or 'the generator'} kept for that caption.", ""]
    for e in pick:
        im = next((x[2][e] for x in mist.values() if e in x[2]), None)
        f = next(iter(by.values()))[e].get("image_file", e + ".jpg")
        th = thumbnail(e, f, out_root, cache_dir, docci_dir, mock)
        L += [f"#### {e}", "", f"![{e}](samples/{os.path.basename(th)})" if th else f"*(image {f} not on disk)*", ""]
        if im:
            L += ["| caption | text |", "|---|---|", f"| Caption 1 (human, ground truth) | {esc(im['caption1'])} |", ""]
        L += ["| captioner | prompt | Caption 2 | contradiction | omission | hallucinated |", "|---|---|---|---|---|---|"]
        for (cap, v), rows in by.items():
            c = (mist.get((cap, v)) or (None, {}, {}))[1].get(e)
            L.append(f"| {cap} | {v} | {esc(rows[e]['caption'])} | "
                     + (" | ".join(str(x) for x in c) if c else "– | – | –") + " |")
        L.append("")
    return L


def qa_checks_table(out_root, dss):
    L = ["## QA checks: generated → kept", "",
         "Each generator checks its own QA (right set and answer, one fact per question) and the rule checks run on "
         "top. kept / generated per set (% kept), and the most common reasons a QA was dropped (one QA can break "
         "several rules). Per-dataset details: `qa_<dataset>/stats.md`.", "",
         "| dataset | self-check by | positive kept | negative kept | hallucinated kept | meta phrases removed | "
         "top reasons for dropping |", "|---|---|---|---|---|---|---|"]
    for ds in dss:
        s = load_json(os.path.join(out_root, f"qa_{ds}", "stats.json"))
        data = load_json(os.path.join(out_root, f"qa_{ds}", "qa_dataset.json"))
        if not s or not data:
            continue
        gen = {k: 0 for k in ("positive", "negative", "hallucinated")}
        for im in data["images"] + data["dropped_images"]:
            for k, v in ((im.get("coverage") or {}).get("n_generated") or {}).items():
                gen[k] = gen.get(k, 0) + v
        kept = s["qa_per_set"]
        cells = [f"{kept[k]} / {gen[k]} ({pct(kept[k] / gen[k]) if gen[k] else '–'}%)" for k in gen]
        top = "; ".join(f"{k} {v}" for k, v in list(s.get("dropped_qa", {}).items())[:4]) or "none"
        L.append(f"| {ds} | {', '.join(s['judges']) or 'rules only'} | " + " | ".join(cells)
                 + f" | {s.get('meta_language_removed', 0)} | {esc(top)} |")
    return L + [""]


CSS = """
:root { --bg: #f7f8f8; --surface: #ffffff; --fg: #1b2124; --muted: #5d686d; --line: #dfe4e6;
  --accent: #0f6f6a; --good: #1f8a4c; --good-bg: #e3f3e9; --bad: #c0392b; --bad-bg: #fbe6e3;
  --mid-bg: #fff4d6; --sans: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Noto Sans KR", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #111618; --surface: #182024;
  --fg: #e4eaec; --muted: #98a6ab; --line: #2a3438; --accent: #4cc3b8; --good: #5fd08f; --good-bg: #183424;
  --bad: #ff8a7a; --bad-bg: #3a1f1c; --mid-bg: #3a3218; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #111618; --surface: #182024; --fg: #e4eaec; --muted: #98a6ab; --line: #2a3438;
  --accent: #4cc3b8; --good: #5fd08f; --good-bg: #183424; --bad: #ff8a7a; --bad-bg: #3a1f1c; --mid-bg: #3a3218;
  color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px; line-height: 1.6;
  margin: 0; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1120px; margin: 0 auto; display: grid; gap: 14px; }
main > * { min-width: 0; margin: 0; }
h1 { font-size: 1.9rem; line-height: 1.25; text-wrap: balance; }
h2 { font-size: 1.35rem; margin-top: 28px; padding-top: 18px; border-top: 1px solid var(--line); text-wrap: balance; }
h3 { font-size: 1.15rem; margin-top: 22px; color: var(--accent); text-wrap: balance; }
h4 { font-size: 1rem; margin-top: 16px; padding-top: 12px; border-top: 1px dashed var(--line);
  font-family: var(--mono); color: var(--fg); }
p, li { max-width: 75ch; }
code { font-family: var(--mono); font-size: .88em; background: var(--surface); border: 1px solid var(--line);
  padding: 0 4px; border-radius: 4px; overflow-wrap: anywhere; }
.tbl { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12px; letter-spacing: .02em; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
td.num { font-family: var(--mono); text-align: right; white-space: nowrap; }
td.hi { background: var(--good-bg); } td.lo { background: var(--mid-bg); } td.vlo { background: var(--bad-bg); }
td.ok { color: var(--good); } td.ng { color: var(--bad); font-weight: 600; }
td.warn { color: var(--bad); font-weight: 600; background: var(--bad-bg); } td.none { color: var(--muted); font-style: italic; }
img.sample { max-width: min(100%, 560px); border-radius: 8px; border: 1px solid var(--line); display: block; }
.noimg { max-width: 560px; padding: 28px 16px; border: 1px dashed var(--line); border-radius: 8px;
  color: var(--muted); text-align: center; }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; }
summary { cursor: pointer; color: var(--muted); }
.notes { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 4px 20px 16px; }
.notes h2 { border-top: 0; margin-top: 12px; padding-top: 0; }
.notes h3 { font-family: var(--sans); color: var(--fg); }
.toc { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 20px; }
.toc ol { margin: 6px 0 0; padding-left: 22px; columns: 2 320px; column-gap: 32px; }
.toc li { max-width: none; break-inside: avoid; }
a { color: var(--accent); }
"""


def _inline(s):
    s = html.escape(s, quote=False).replace("\\|", "|")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def _cell(c, tag, head=""):
    c = c.strip()
    cls = []
    if tag == "td":
        if re.fullmatch(r"\d{1,3}\.\d", c) and ("ratio" in head or head in ("share", "%")):  # a share, not a score
            cls = ["num"]
        elif re.fullmatch(r"\d{1,3}\.\d", c) and "↓" in head:  # lower is better
            v = float(c)
            cls = ["num", "hi" if v <= 10 else "" if v <= 25 else "lo" if v <= 40 else "vlo"]
        elif re.fullmatch(r"\d{1,3}\.\d", c):  # a percentage: shade by value
            v = float(c)
            cls = ["num", "hi" if v >= 90 else "lo" if v < 75 and v >= 60 else "vlo" if v < 60 else ""]
        elif re.fullmatch(r"[\d,]+|–", c):
            cls = ["num"]
        elif "⚠️" in c:
            cls = ["warn"]
        elif "✅" in c:
            cls = ["ok"]
        elif "❌" in c:
            cls = ["ng"]
        elif c.startswith("none ("):
            cls = ["none"]
    attr = f' class="{" ".join(x for x in cls if x)}"' if any(cls) else ""
    return f"<{tag}{attr}>{_inline(c)}</{tag}>"


def _row(line):
    return [p for p in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def md_to_html(md, base_dir):
    """The small Markdown subset summary.md uses: headings, tables, images, <details>, paragraphs, lists."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
            heads = [h.strip().lower() for h in _row(l)]
            head = "".join(_cell(c, "th") for c in _row(l))
            i += 2
            body = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = _row(lines[i])
                body.append("<tr>" + "".join(_cell(c, "td", heads[j] if j < len(heads) else "")
                                             for j, c in enumerate(cells)) + "</tr>")
                i += 1
            out.append(f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
            continue
        m = re.match(r"^(#{1,4}) (.*)", l)
        img = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", l)
        if m:
            n = len(m.group(1))
            hid = f' id="s{sum(o.startswith("<h2") for o in out) + 1}"' if n == 2 else ""
            out.append(f"<h{n}{hid}>{_inline(m.group(2))}</h{n}>")
        elif img:
            p = os.path.join(base_dir, img.group(2))
            if os.path.exists(p):
                data = base64.b64encode(open(p, "rb").read()).decode()
                mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
                out.append(f'<img class="sample" alt="{html.escape(img.group(1))}" src="data:{mime};base64,{data}">')
            else:
                out.append(f'<div class="noimg">사진 없음: {html.escape(img.group(2))}</div>')
        elif l.startswith("<"):
            out.append(l)
        elif l.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif l.strip():
            para = [l]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|<|- )", lines[i + 1]):
                i += 1
                para.append(lines[i])
            out.append(f"<p>{_inline(' '.join(para))}</p>")
        i += 1
    return "\n".join(out)


FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
         '&family=IBM+Plex+Sans+KR:wght@400;600&display=swap">')


def write_html(md, out_path, base_dir, title="DOCCI P/N/H QA results", notes_md=None, fragment=False):
    """fragment=True leaves out <html>/<head>/<body> (for pages that get wrapped elsewhere)."""
    notes = f'<section class="notes">{md_to_html(notes_md, base_dir)}</section>' if notes_md else ""
    body = md_to_html(md, base_dir)
    toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in re.findall(r'<h2 id="(s\d+)">(.*?)</h2>', body))
    toc = f'<nav class="toc"><strong>Contents</strong><ol>{toc}</ol></nav>' if toc else ""
    h1_end = body.find("</h1>") + len("</h1>")
    body = body[:h1_end] + "\n" + notes + toc + body[h1_end:]  # notes, then contents, right under the title
    if fragment:
        page = (f'<title>{html.escape(title)}</title><meta name="claude:viewport-fit-sides" content="cover">'
                f'{FONTS}<style>{CSS}</style><main>{body}</main>')
    else:
        page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
                f'{FONTS}<style>{CSS}</style></head><body><main>{body}</main></body></html>')
    with open(out_path, "w") as f:
        f.write(page)
    return out_path




# ------------------------------------------------------------------ main
def write_report(out_root, order=None, cache_dir=None, docci_dir=None, failed=(), samples_per_case=2,
                 seed=0, mock=False, notes=None, captioner_dirs=()):
    cache_dir = cache_dir or os.path.join(C.WORK_ROOT, "cache")
    dss = datasets(out_root, order)
    caps = caption2_files(out_root)
    L = header(out_root, dss) + captioner_section(out_root, captioner_dirs)
    L += caption2_stats(caps) if caps else []
    L += caption2_table(out_root, dss) + [""]
    L += caption2_examples(out_root, dss, caps, max(samples_per_case, 3), seed, cache_dir, docci_dir, mock)
    L += dataset_table(out_root, dss) + [""] + qa_checks_table(out_root, dss) + coverage_tables(out_root, dss)
    L += bench_table(out_root, dss) + [""]
    L += per_vlm_tables(out_root, dss, "case", "Benchmark per coverage case (with the image)",
                        "Same questions, split by which sets their image has.") + [""]
    L += per_vlm_tables(out_root, dss, "category", "Benchmark per category (with the image)",
                        "Low negative / high repeats in a category = the VLM shares the captioner's mistakes there.") + [""]
    for ds in dss:
        try:
            L += sample_section(out_root, ds, cache_dir, docci_dir, samples_per_case, seed, mock)
        except Exception:  # the numbers above matter more than the samples: never lose them
            traceback.print_exc()
            L += [f"## Samples by coverage case: {ds}", "", "Could not build the samples (see run_all.log)."]
    if failed:
        L += ["", "## Failed steps", ""] + [f"- {n}: see `{log}`" for n, log in failed]
    path = os.path.join(out_root, "summary.md")
    md = "\n".join(L) + "\n"
    with open(path, "w") as f:
        f.write(md)
    notes_md = None
    if notes:
        with open(notes) as f:
            notes_md = f.read()
    write_html(md, os.path.join(out_root, "summary.html"), out_root, notes_md=notes_md)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"), help="where DOCCI images are")
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--samples_per_case", type=int, default=2, help="sample images per coverage case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default=None, help="a Markdown file (e.g. your analysis) shown at the top of summary.html")
    ap.add_argument("--step1_dir", default=None, help="Step 1 folder (default <out_root>/step1)")
    ap.add_argument("--grpo_dir", default=None, help="GRPO folder (default <out_root>/grpo)")
    a = ap.parse_args()
    dirs = [("Step 1: score checks", a.step1_dir or os.path.join(a.out_root, "step1")),
            ("Step 2: GRPO, captions before / after RL", a.grpo_dir or os.path.join(a.out_root, "grpo"))]
    path = write_report(a.out_root, cache_dir=a.cache_dir, docci_dir=a.docci_dir,
                        samples_per_case=a.samples_per_case, seed=a.seed, notes=a.notes, captioner_dirs=dirs)
    print(f"[report] {path}\n[report] {os.path.join(a.out_root, 'summary.html')} (open in a browser; photos inside)")


if __name__ == "__main__":
    main()
