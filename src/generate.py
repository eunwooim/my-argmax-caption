"""Positive / Negative / Hallucinated QA from two captions of each DOCCI image (default Qwen3-32B).

Caption 1 = the DOCCI human description (ground truth). Caption 2 = an AI caption of the same image
(caption.py, or your own file). The prompt is pnh_prompt.md (edit it there):
  positive      answerable from both captions, same answer
  negative      answer from Caption 1; Caption 2 contradicts it (neg_type contradiction) or leaves it out (omission)
  hallucinated  answer = a false claim of Caption 2 that conflicts with Caption 1, plus caption1_truth

Stages (run all, or one at a time; every stage resumes from what is already in --out_dir):
  gen     the LLM writes the three sets for each image -> gen_raw.jsonl
  verify  an independent judge checks every item against both captions (right set? atomic?)
          -> verify/<tag>.jsonl. Run it again with another --model to add a second judge.
  build   rule checks + judges + coverage per image ->
            qa_dataset.json  one entry per image in the prompt's schema (positive / negative / hallucinated
                             lists) with "coverage": which of the three sets the image has
            qa_final.jsonl   one row per QA (input of benchmark.py), each with its image's coverage
            stats.md / stats.json

An image is kept when it has at least one positive QA. Its coverage case is P+N+H (all three sets),
P+N, P+H or P (positive only). Images without a positive are listed in qa_dataset.json under
"dropped_images" with the reason and whatever QA they had.

Smoke test without a model or downloads:  python generate.py --mock --out_dir out_mock
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ next to the code: every write goes under WORK_ROOT

import argparse
import collections
import json
import os
import re

import qa_common as C
import qa_prompts as P
from llm import LazyChat, add_llm_args, make_backend, reasons, resolve_defaults

SHORT = {"positive": "pos", "negative": "neg", "hallucinated": "hal"}
CASES = ["P+N+H", "P+N", "P+H", "P"]  # kept; anything without P is dropped
CASE_NAMES = {"P+N+H": "all three sets", "P+N": "positive + negative", "P+H": "positive + hallucinated",
              "P": "positive only"}

# "..., according to the candidate claim?" (as in the prompt's own few-shot) -> context-free question
_SRC = r"(the )?(candidate( claim| caption)?|ai( model)?( caption)?|caption ?[12]|captions|ground truth|human caption|description|claim)"
META_TAIL = re.compile(r",?\s+(according to|as stated in|as described in|as per|per|based on|in) " + _SRC + r"(?=\s*\?)", re.I)
META_HEAD = re.compile(r"^(according to|as stated in|as described in|based on|in) " + _SRC + r"\s*,\s*", re.I)
LEAKY = re.compile(r"\b(caption|captions|candidate claim|ground truth|the description|described|mentioned|according to)\b", re.I)


# ------------------------------------------------------------------ mock model (smoke tests only)
_MOCK_COLORS = ["red", "green", "blue", "yellow", "black", "white", "orange", "purple"]


def _mock_captions(msgs):
    m = re.search(r"- Caption 1: (.*)\n- Caption 2: (.*)", msgs[-1]["content"], re.S)
    return m.group(1).strip(), m.group(2).strip()


def _sides(text):
    return {s: (c, sh) for s, c, sh in re.findall(r"On the (left|right) is an? (\w+) (\w+)\.", text)}


def mock_gen(msgs):
    c1, c2 = _mock_captions(msgs)
    mcq = "Format for this run: MCQ" in msgs[0]["content"]
    truth_opt = "DO include it" in msgs[0]["content"]
    s1, s2 = _sides(c1), _sides(c2)
    out = {"positive": [], "negative": [], "hallucinated": []}

    def add(kind, q, ans, cat, **kw):
        it = dict(id=f"{SHORT[kind]}_{len(out[kind]) + 1}", question=q, answer=ans, category=cat,
                  format="mcq" if mcq else "open-ended", **kw)
        if mcq:
            avoid = {ans, kw.get("caption1_truth")}
            pool = [c for c in _MOCK_COLORS if c not in avoid] if cat in ("attribute", "scene") else \
                [s for s in ["circle", "square", "triangle", "star", "hexagon", "dog", "bird"] if s not in avoid]
            opts = [ans] + pool[:3]
            if truth_opt and kind == "hallucinated":
                opts = [ans, kw["caption1_truth"]] + pool[:2]
            it["options"] = dict(zip(P.LETTERS, opts))
            it["answer"] = "A"
        out[kind].append(it)

    if "gray background" in c2:
        add("positive", "What color is the background?", "gray", "scene")
    for side, (col, shape) in s1.items():
        if side not in s2:
            add("negative", f"What shape is on the {side}?", shape, "object", neg_type="omission")
            add("negative", f"What color is the {shape} on the {side}?", col, "attribute", neg_type="omission")
            continue
        col2, shape2 = s2[side]
        if shape2 == shape:
            add("positive", f"What shape is on the {side}?", shape, "object")
        if col2 == col:
            add("positive", f"What color is the {shape} on the {side}?", col, "attribute")
        else:
            add("negative", f"What color is the {shape} on the {side}?", col, "attribute", neg_type="contradiction")
            add("hallucinated", f"What color is the {shape} on the {side} according to the candidate claim?",
                col2, "attribute", caption1_truth=col)
    if "cat" in c2 and "Nothing else" in c1:
        add("hallucinated", "What animal sits between the shapes?", "a cat", "object", caption1_truth="nothing")
    # three items the checks must drop: a rule check (answer too long) and the judge (wrong set, not atomic)
    add("positive", "According to Caption 1, how many shapes are there?", "two", "count")
    add("positive", "What is in the picture?", "two colored shapes on a plain gray background", "scene")
    add("positive", "What color and shape is the object on the left?", "a colored shape", "attribute")
    return json.dumps(out)


def mock_verify(msgs):
    item = json.loads(msgs[-1]["content"].split("Item:\n", 1)[1].split("\n\nReturn", 1)[0])
    atomic = " and " not in item["question"].lower()
    valid = not (item["set"] == "positive" and "how many" in item["question"].lower())  # Caption 2 says "some"
    return json.dumps(dict(reason="mock", valid=valid, atomic=atomic))


def mock_records(n):
    import mock_docci
    return mock_docci.records(n)


# ------------------------------------------------------------------ Caption 2
def load_caption2(path):
    """{example_id: (caption, model, prompt version)} from a JSONL with "example_id" and "caption"
    (and optionally "model" and "prompt_version", as caption.py writes them)."""
    rows = C.read_shards(path)
    if not rows:
        raise SystemExit(f"[gen] no captions in {path} (or its shards): run caption.py first")
    return {r["example_id"]: (r["caption"], r.get("model", "unknown"), r.get("prompt_version", "unknown"))
            for r in rows if r.get("caption")}


def attach_caption2(recs, path, mock):
    if not path and mock:
        import mock_docci
        return [dict(r, caption2=mock_docci.caption2(r["example_id"]), caption2_model="mock",
                     caption2_prompt="simple") for r in recs], []
    if not path:
        raise SystemExit("[gen] --caption2 <file.jsonl> is required (Caption 2 for every image, e.g. from caption.py)")
    caps = load_caption2(path)
    have = [dict(r, caption2=caps[r["example_id"]][0], caption2_model=caps[r["example_id"]][1],
                 caption2_prompt=caps[r["example_id"]][2]) for r in recs if r["example_id"] in caps]
    missing = [r for r in recs if r["example_id"] not in caps]
    if missing:
        print(f"[gen] {len(missing)} images have no Caption 2 and are skipped, e.g. {missing[0]['example_id']}",
              flush=True)
    return have, missing


# ------------------------------------------------------------------ stage: gen
def stage_gen(a, recs, chat):
    base = os.path.join(a.out_dir, "gen_raw.jsonl")
    path = C.shard_path(a, base)
    recs = [r for r in recs if C.in_shard(a, r["example_id"])]
    done = {r["example_id"] for r in C.read_shards(base) if r["parse_ok"]}
    for attempt in range(1 + a.gen_retries):
        todo = [r for r in recs if r["example_id"] not in done]
        if not todo:
            break
        print(f"[gen] {len(todo)} images to do (pass {attempt + 1}), {len(done)} done", flush=True)
        temp = None if attempt == 0 else 0.6  # a retry samples a bit more freely
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]
            convs = [P.gen_messages(r["description"], r["caption2"], a.qa_format, a.prompt, a.hal_mcq_truth)
                     for r in chunk]
            outs = chat(convs, schema=P.GEN_SCHEMA, temperature=temp)
            rows = []
            for r, o in zip(chunk, outs):
                obj = C.parse_json(o)
                ok = isinstance(obj, dict) and any(isinstance(obj.get(k), list) for k in P.SETS)
                if ok:  # a set the model left out entirely counts as empty
                    obj = {k: [x for x in obj.get(k) or [] if isinstance(x, dict)] for k in P.SETS}
                rows.append(dict(example_id=r["example_id"], split=r["split"], image_file=r["image_file"],
                                 caption1=r["description"], caption2=r["caption2"],
                                 caption2_model=r["caption2_model"], caption2_prompt=r["caption2_prompt"],
                                 gen_model=a.model, qa_format=a.qa_format,
                                 parse_ok=ok, output=obj if ok else None, raw=None if ok else o))
                if ok:
                    done.add(r["example_id"])
            C.append_jsonl(path, rows)
            print(f"[gen] {min(s + a.batch, len(todo))}/{len(todo)} "
                  f"(parse ok {sum(x['parse_ok'] for x in rows)}/{len(rows)})", flush=True)


def gen_records(out_dir):
    """Latest gen row per example_id (a retry overrides an earlier failure)."""
    last = {}
    for r in C.read_shards(os.path.join(out_dir, "gen_raw.jsonl")):
        if r["parse_ok"] or r["example_id"] not in last:
            last[r["example_id"]] = r
    return [last[k] for k in sorted(last)]


def _str(x):
    return x.strip() if isinstance(x, str) else ""


def context_free(q):
    """Remove 'according to the candidate claim' and the like: the benchmark asks about the image."""
    q2 = META_TAIL.sub("", META_HEAD.sub("", q)).strip()
    q2 = re.sub(r"\s+\?", "?", q2)
    return q2[:1].upper() + q2[1:] if q2 else q2


def _letter(ans, opts):
    a = ans.strip().rstrip(".").strip()
    if a.upper() in P.LETTERS:
        return a.upper()
    m = re.match(r"^\(?([A-D])[).:]\s", ans.strip())
    if m:
        return m.group(1)
    hit = [k for k, v in opts.items() if C.norm_text(v) == C.norm_text(ans)]
    return hit[0] if len(hit) == 1 else None


def _same(x, y):
    return bool(x) and bool(y) and C.norm_text(x) == C.norm_text(y)


def check_item(kind, it, a):
    """Normalized item + the rules it breaks."""
    q_raw = _str(it.get("question"))
    q = q_raw if a.keep_meta else context_free(q_raw)
    ans = _str(it.get("answer"))
    fmt = _str(it.get("format")).lower() or "open-ended"
    cat = _str(it.get("category")).lower()
    row = dict(set=kind, id=_str(it.get("id")), question=q, answer=ans, answer_text=ans, category=cat, format=fmt)
    if q != q_raw:
        row["question_raw"] = q_raw
    problems = []
    if not q or not ans:
        problems.append("empty")
    elif not q.endswith("?"):
        problems.append("not_a_question")
    if LEAKY.search(q):
        problems.append("meta_language")
    if cat not in P.CATEGORIES:
        problems.append("bad_category")
    if fmt not in P.FORMATS:
        problems.append("bad_format")
    if fmt == "mcq":
        opts = it.get("options") if isinstance(it.get("options"), dict) else {}
        opts = {k: _str(opts.get(k)) for k in P.LETTERS}
        letter = _letter(ans, opts) if all(opts.values()) else None
        if not all(opts.values()) or len({C.norm_text(v) for v in opts.values()}) < 4 or letter is None:
            problems.append("bad_options")
        else:
            row.update(options=opts, answer=letter, answer_text=opts[letter])
    if len(row["answer_text"].split()) > a.max_answer_words:
        problems.append("answer_too_long")
    if kind == "negative":
        row["neg_type"] = _str(it.get("neg_type")).lower()
        if row["neg_type"] not in P.NEG_TYPES:
            problems.append("bad_neg_type")
    if kind == "hallucinated":
        truth = _str(it.get("caption1_truth"))
        row["caption1_truth"] = truth
        if not truth:
            problems.append("no_caption1_truth")
        elif _same(truth, row["answer_text"]):
            problems.append("claim_equals_truth")
        elif len(truth.split()) > a.max_answer_words:
            problems.append("truth_too_long")
        if row.get("options"):
            hit = [k for k, v in row["options"].items() if _same(v, truth)]
            if hit:
                row["truth_letter"] = hit[0]
            if hit and not a.hal_mcq_truth:
                problems.append("truth_in_options")
            if not hit and a.hal_mcq_truth:
                problems.append("truth_not_in_options")
    return row, problems


def candidates(a):
    """Flatten gen output into one row per QA, with rule-check flags."""
    rows = []
    for g in gen_records(a.out_dir):
        if not g["parse_ok"]:
            continue
        for kind in P.SETS:
            seen = set()
            for i, it in enumerate(g["output"][kind]):
                row, problems = check_item(kind, it, a)
                key = C.norm_text(row["question"])
                if key in seen:  # within one set; the same question may be in negative and hallucinated
                    problems.append("duplicate")
                seen.add(key)
                rows.append(dict(qa_id=f"{g['example_id']}_{SHORT[kind]}{i}", example_id=g["example_id"],
                                 split=g["split"], image_file=g["image_file"], **row,
                                 gen_model=g["gen_model"], caption2_model=g["caption2_model"],
                                 rule_problems=problems))
    return rows


# ------------------------------------------------------------------ stage: verify
def verifier_tag(a):
    return a.verify_tag or re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(a.model.rstrip("/")))


def stage_verify(a, chat):
    caps = {g["example_id"]: g for g in gen_records(a.out_dir)}
    cands = [c for c in candidates(a) if not c["rule_problems"] and C.in_shard(a, c["example_id"])]
    base = os.path.join(a.out_dir, "verify", verifier_tag(a) + ".jsonl")
    path = C.shard_path(a, base)
    done = {r["qa_id"] for r in C.read_shards(base) if r["valid"] is not None}
    todo = [c for c in cands if c["qa_id"] not in done]
    print(f"[verify:{verifier_tag(a)}] {len(todo)} QA to check, {len(done)} done", flush=True)
    for s in range(0, len(todo), a.batch * 8):
        chunk = todo[s:s + a.batch * 8]
        convs = [P.verify_messages(caps[c["example_id"]]["caption1"], caps[c["example_id"]]["caption2"], c)
                 for c in chunk]
        outs = chat(convs, schema=P.VERIFY_SCHEMA, temperature=None if reasons(a) else 0.0)
        rows = []
        for c, o in zip(chunk, outs):
            obj = C.parse_json(o) or {}
            ok = isinstance(obj.get("valid"), bool)
            rows.append(dict(qa_id=c["qa_id"], valid=obj["valid"] if ok else None,
                             atomic=bool(obj.get("atomic", True)), reason=_str(obj.get("reason")), model=a.model))
        C.append_jsonl(path, rows)
        print(f"[verify] {min(s + len(chunk), len(todo))}/{len(todo)}", flush=True)


# ------------------------------------------------------------------ stage: build
def accepts(v):
    return v is not None and v["valid"] is True and v["atomic"]


def link_pairs(items):
    """A contradiction negative and the hallucinated QA about the same fact point at each other; the negative
    also gets the claim Caption 2 made (caption2_claim), so the benchmark can see if a VLM repeats it."""
    hals = [q for q in items if q["set"] == "hallucinated"]
    for n in items:
        if n["set"] != "negative" or n["neg_type"] != "contradiction":
            continue
        free = [h for h in hals if "pair_qa_id" not in h]
        h = next((h for h in free if _same(h["question"], n["question"])), None) or \
            next((h for h in free if _same(h["caption1_truth"], n["answer_text"])), None)
        if h:
            n["pair_qa_id"], h["pair_qa_id"] = h["qa_id"], n["qa_id"]
            n["caption2_claim"] = h["answer_text"]


def coverage(items, generated):
    n = {k: sum(q["set"] == k for q in items) for k in P.SETS}
    case = "+".join(L for L, k in zip("PNH", P.SETS) if n[k]) or "none"
    return dict(case=case, complete=case == "P+N+H", positive=n["positive"] > 0, negative=n["negative"] > 0,
                hallucinated=n["hallucinated"] > 0, n=n, n_generated=generated,
                neg_types={t: sum(q.get("neg_type") == t for q in items) for t in P.NEG_TYPES})


SCHEMA_KEYS = ["id", "qa_id", "question", "question_raw", "answer", "answer_text", "options", "neg_type",
               "caption1_truth", "truth_letter", "caption2_claim", "pair_qa_id", "category", "format"]


def stage_build(a):
    cands = candidates(a)
    vdir = os.path.join(a.out_dir, "verify")
    tags = [] if a.verifiers == "none" else a.verifiers.split(",") if a.verifiers else sorted(
        {C.unshard_name(f[:-6]) for f in os.listdir(vdir) if f.endswith(".jsonl")}) if os.path.isdir(vdir) else []
    verdicts = {}
    for t in tags:
        verdicts[t] = {}
        for r in C.read_shards(os.path.join(vdir, t + ".jsonl")):
            if r["valid"] is not None or r["qa_id"] not in verdicts[t]:
                verdicts[t][r["qa_id"]] = r
    if not tags:
        print("[build] no verify/ results: keeping every QA that passes the rule checks", flush=True)

    drop = collections.Counter()
    kept = collections.defaultdict(list)
    for c in cands:
        if c["rule_problems"]:
            drop.update(f"{c['set']}: rule {p}" for p in c["rule_problems"])
            continue
        votes = [accepts(verdicts[t].get(c["qa_id"])) for t in tags]
        need = {"all": len(tags), "any": min(1, len(tags)), "majority": len(tags) // 2 + 1}[a.require]
        if sum(votes) < need:
            v = [verdicts[t].get(c["qa_id"]) for t in tags]
            why = "not atomic" if any(x and x["valid"] and not x["atomic"] for x in v) else "wrong set / answer"
            drop[f"{c['set']}: judge ({why})"] += 1
            continue
        c["verifiers"] = {t: (verdicts[t].get(c["qa_id"]) or {}).get("valid") for t in tags}
        kept[c["example_id"]].append(c)

    images, dropped, final = [], [], []
    for g in gen_records(a.out_dir):
        e = g["example_id"]
        entry = dict(example_id=e, split=g["split"], image_file=g["image_file"], caption1=g["caption1"],
                     caption2=g["caption2"], caption2_model=g["caption2_model"],
                     caption2_prompt=g.get("caption2_prompt", "unknown"), gen_model=g["gen_model"])
        if not g["parse_ok"]:
            dropped.append(dict(entry, reason="json_parse_failure", coverage=None, **{k: [] for k in P.SETS}))
            continue
        items = sorted(kept.get(e, []), key=lambda q: q["qa_id"])
        link_pairs(items)
        cov = coverage(items, {k: len(g["output"][k]) for k in P.SETS})
        entry["coverage"] = cov
        for k in P.SETS:
            entry[k] = [{x: q[x] for x in SCHEMA_KEYS if q.get(x) not in (None, "")} for q in items if q["set"] == k]
        if cov["positive"] or a.keep_no_positive:
            images.append(entry)
            for q in items:
                q.pop("rule_problems")
                final.append(dict(q, coverage=cov["case"]))
        else:
            dropped.append(dict(entry, reason="no_positive"))
    for r in a.no_caption2:
        dropped.append(dict(example_id=r["example_id"], split=r["split"], image_file=r["image_file"],
                            caption1=r["description"], caption2=None, reason="no_caption2", coverage=None,
                            **{k: [] for k in P.SETS}))

    meta = summary(a, images, dropped, final, drop, tags)
    C.write_jsonl(os.path.join(a.out_dir, "qa_final.jsonl"), final)
    tmp = os.path.join(a.out_dir, "qa_dataset.json.tmp")
    with open(tmp, "w") as f:
        json.dump(dict(meta=meta, images=images, dropped_images=dropped), f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(a.out_dir, "qa_dataset.json"))
    write_stats(a, meta)


def summary(a, images, dropped, final, drop, tags):
    cases = collections.Counter(im["coverage"]["case"] for im in images)
    n_set = collections.Counter(q["set"] for q in final)
    gens = sorted({im["gen_model"] for im in images + dropped if im.get("gen_model")})
    capm = sorted({im["caption2_model"] for im in images + dropped if im.get("caption2_model")})
    capp = sorted({im["caption2_prompt"] for im in images + dropped if im.get("caption2_prompt")})
    by_case_set = {c: {k: sum(im["coverage"]["n"][k] for im in images if im["coverage"]["case"] == c)
                       for k in P.SETS} for c in cases}
    return dict(
        generator=", ".join(gens), caption2_model=", ".join(capm), caption2_prompt=", ".join(capp),
        judges=tags, require=a.require,
        qa_format=a.qa_format, hal_mcq_truth=a.hal_mcq_truth, max_answer_words=a.max_answer_words,
        images_in=len(images) + len(dropped), images=len(images),
        coverage={c: cases[c] for c in CASES + sorted(set(cases) - set(CASES))},
        coverage_qa=by_case_set,
        dropped_images=dict(collections.Counter(d["reason"] for d in dropped)),
        dropped_no_positive_cases=dict(collections.Counter(d["coverage"]["case"] for d in dropped
                                                           if d["reason"] == "no_positive")),
        qa=len(final), qa_per_set={k: n_set[k] for k in P.SETS},
        neg_type={t: sum(q.get("neg_type") == t for q in final) for t in P.NEG_TYPES},
        paired_contradictions=sum(1 for q in final if q["set"] == "negative" and q.get("pair_qa_id")),
        category={k: {c: sum(q["set"] == k and q["category"] == c for q in final) for c in P.CATEGORIES}
                  for k in P.SETS},
        format={k: {f: sum(q["set"] == k and q["format"] == f for q in final) for f in P.FORMATS} for k in P.SETS},
        meta_language_removed=sum(1 for q in final if q.get("question_raw")),
        dropped_qa=dict(drop.most_common()))


def write_stats(a, m):
    with open(os.path.join(a.out_dir, "stats.json"), "w") as f:
        json.dump(m, f, indent=2)
    nt, qs = m["neg_type"], m["qa_per_set"]
    L = ["# DOCCI Positive / Negative / Hallucinated QA", "",
         f"- generator: {m['generator']}; Caption 2: {m['caption2_model']} ({m['caption2_prompt']} prompt); "
         f"format: {m['qa_format']}",
         f"- images: {m['images_in']} in, **{m['images']} kept** (at least one positive QA); "
         f"dropped: {m['dropped_images'] or 'none'}",
         f"- QA: **{m['qa']}** = positive {qs['positive']} / negative {qs['negative']} "
         f"(contradiction {nt['contradiction']}, omission {nt['omission']}) / hallucinated {qs['hallucinated']}",
         f"- contradictions linked to their hallucinated QA: {m['paired_contradictions']}; "
         f"questions whose meta-language was removed: {m['meta_language_removed']}",
         f"- judges: {', '.join(m['judges']) or 'none'} (require {m['require']})", "",
         "## Coverage (which sets each kept image has)", "",
         "| case | meaning | images | % | positive QA | negative QA | hallucinated QA |", "|---|---|---|---|---|---|---|"]
    for c, n in m["coverage"].items():
        q = m["coverage_qa"].get(c, {k: 0 for k in P.SETS})
        L.append(f"| {c} | {CASE_NAMES.get(c, '')} | {n} | {100 * n / max(m['images'], 1):.1f} | "
                 f"{q['positive']} | {q['negative']} | {q['hallucinated']} |")
    if m["dropped_no_positive_cases"]:
        L += ["", f"Dropped for having no positive QA, by what they had: {m['dropped_no_positive_cases']}"]
    L += ["", "## Dropped QA", "", "| reason | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in m["dropped_qa"].items()] or ["| – | 0 |"]
    L += ["", "## Category", "", "| category | positive | negative | hallucinated |", "|---|---|---|---|"]
    for c in P.CATEGORIES:
        L.append(f"| {c} | " + " | ".join(str(m["category"][k][c]) for k in P.SETS) + " |")
    with open(os.path.join(a.out_dir, "stats.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    help="all, or any of gen,verify,build comma separated (e.g. gen,verify per shard, then build)")
    ap.add_argument("--mock", action="store_true", help="mock model + synthetic images and captions, no GPU")
    ap.add_argument("--caption2", default=None,
                    help="Caption 2 file: JSONL with example_id and caption (caption.py writes it; shards are read too)")
    ap.add_argument("--qa_format", default="open-ended", choices=["open-ended", "mcq", "mixed"])
    ap.add_argument("--hal_mcq_truth", action="store_true",
                    help="hallucinated MCQ: put the Caption 1 truth among the options (default: leave it out, "
                         "as the prompt says)")
    ap.add_argument("--prompt", default=P.PNH_PROMPT, help="generation prompt (Markdown, see pnh_prompt.md)")
    ap.add_argument("--max_answer_words", type=int, default=5, help="rule check: longer answers are dropped")
    ap.add_argument("--keep_meta", action="store_true",
                    help="keep 'according to the candidate claim' in questions (default: removed, so the "
                         "benchmark asks about the image)")
    ap.add_argument("--keep_no_positive", action="store_true",
                    help="build: keep images that have no positive QA (default: drop them)")
    ap.add_argument("--splits", default="test", help="DOCCI splits, comma separated: train,test,qual_dev,qual_test")
    ap.add_argument("--n_images", type=int, default=0, help="random sample of this many images (0 = all)")
    ap.add_argument("--batch", type=int, default=256, help="images per write to disk")
    ap.add_argument("--gen_retries", type=int, default=1, help="extra passes over images whose JSON failed")
    ap.add_argument("--verify_tag", default=None, help="name for this judge's results (default: model name)")
    ap.add_argument("--verifiers", default=None, help="build: judges to use, comma separated (default: all in verify/; none = rule checks only)")
    ap.add_argument("--require", default="all", choices=["all", "majority", "any"],
                    help="build: how many judges must accept a QA")
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"))
    ap.add_argument("--docci_dir", default=None, help="folder with docci_descriptions.jsonlines (and images/)")
    ap.add_argument("--out_dir", default="out_qa")
    add_llm_args(ap, "Qwen/Qwen3-32B", 3072)
    C.add_gpu_args(ap, n_gpus=1)  # one H200 (141 GB) holds Qwen3-32B bf16 or gpt-oss-120b
    C.add_shard_args(ap)
    ap.set_defaults(max_model_len=16384)  # prompt ~1.6k + Caption 2 up to 4k tokens + output up to 6k (gpt-oss)
    a = ap.parse_args()
    if a.mock:
        a.backend = "mock"
        a.model = "mock"
    resolve_defaults(a, 3072)
    if a.temperature is None:
        a.temperature = 0.2  # the prompt's recommendation (0.1-0.2) for consistent JSON
    stages = ["gen", "verify", "build"] if a.stage == "all" else a.stage.split(",")
    bad = set(stages) - {"gen", "verify", "build"}
    if bad:
        ap.error(f"unknown --stage {bad}")
    if "build" in stages and C.shard_kn(a)[1] > 1:
        ap.error("build reads every shard: run it once, without --shard, after all shards finish")
    if not a.mock:
        C.setup_caches()
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[paths] results: {os.path.abspath(a.out_dir)}   DOCCI: {os.path.abspath(a.cache_dir)}", flush=True)

    loaded = {}

    def load():  # one model load for gen and verify, and only if one of them has work left
        if "chat" not in loaded:
            if a.backend == "vllm":
                C.pick_gpus(a)
            loaded["chat"] = make_backend(a, mock_fn=mock_gen)
        return loaded["chat"]

    def for_stage(mock_fn):
        def get():
            chat = load()
            if a.mock:
                chat.fn = mock_fn
            return chat
        return LazyChat(get)

    a.no_caption2 = []
    if "gen" in stages or ("build" in stages and (a.caption2 or a.mock)):  # build: to list images without one
        recs = mock_records(a.n_images or 8) if a.mock else \
            C.load_docci(a.cache_dir, a.splits.split(","), a.n_images, a.seed, a.docci_dir)
        recs, a.no_caption2 = attach_caption2(recs, a.caption2, a.mock)
        if "gen" in stages:
            stage_gen(a, recs, for_stage(mock_gen))
    if "verify" in stages:
        stage_verify(a, for_stage(mock_verify))
    if "build" in stages:
        stage_build(a)


if __name__ == "__main__":
    main()
