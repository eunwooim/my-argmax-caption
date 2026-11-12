"""Sample page: real results for random images of chosen datasets (default: before RL, simple prompt).

For every dataset whose tag matches --datasets (qa_<generator>__<captioner>__<prompt> folders; the default
'*__before__simple' = each generator's QA from the captioner before RL with the simple prompt), N random images,
each with: the photo, Caption 1 (human), Caption 2 (the generated caption), coverage, then every Positive /
Negative / Hallucinated question with its answer and each VLM's full answer and grade, with the image and
without it (<vlm>__noimage). One self-contained HTML file (photos embedded). No GPU, seconds.

  python sample_html.py --out_root runs                          # P+N+H 20, P+N 10, P+H 10, P 10 images per dataset
  python sample_html.py --out_root runs --per_case "P+N+H=5,P=5"
  python sample_html.py --out_root runs --per_case "" --n 50     # 50 random images, any case
  python sample_html.py --out_root runs --n 100 --seed 1 --out runs/sample_before_simple_100.html
  python sample_html.py --out_root runs --datasets 'Qwen3-32B__before__simple' --case P+N+H
  python sample_html.py --out_root runs --datasets '*__after__simple' --out runs/sample_after_simple.html
"""
import sys

sys.dont_write_bytecode = True

import argparse
import fnmatch
import os
import random

import qa_common as C
import report as R


def section(a, ds):
    data = R.load_json(os.path.join(a.out_root, f"qa_{ds}", "qa_dataset.json"))
    if not data:
        return [f"## {ds}", "", "No qa_dataset.json yet."]
    ims = data["images"] + (data.get("dropped_images", []) if a.include_dropped else [])
    if a.case:
        ims = [im for im in ims if (im.get("coverage") or {}).get("case") == a.case]
    answers = R.load_answers(a.out_root, ds)
    runs = list(answers)
    rng = random.Random(a.seed)
    if a.per_case and not a.case:  # N images per coverage case, shown case by case
        groups = []
        for c, k in a.per_case.items():
            g = [im for im in ims if (im.get("coverage") or {}).get("case") == c]
            groups.append((c, len(g), sorted(rng.sample(g, min(k, len(g))), key=lambda x: x["example_id"])))
    else:
        groups = [(None, len(ims), sorted(rng.sample(ims, min(a.n, len(ims))), key=lambda x: x["example_id"]))]
    pick = [im for _, _, p in groups for im in p]
    gen, cap, prompt = (ds.split("__") + ["?", "?"])[:3]
    L = [f"## {ds}", "",
         f"QA generator **{R.esc(gen)}**, captioner **{cap}** RL, caption prompt **{prompt}**. "
         f"{len(pick)} of {len(ims)} images (random, seed {a.seed})"
         + (" by coverage case: " + ", ".join(f"{c} {len(p)}/{n}" for c, n, p in groups) if groups[0][0] else "")
         + (f", coverage case {a.case}" if a.case else "") + ". "
         + (f"VLM columns: {', '.join(f'`{r}`' for r in runs)}." if runs else "No benchmark answers yet."), ""]
    for c, n, p in groups:
        if c:
            L += [f"### {c}: {R.CASE_NAMES[c]} ({len(p)} of {n} images)", ""] + ([] if p else ["No images in this case.", ""])
        for im in p:
            L += R.image_block(im, runs if im.get("reason") is None else [], answers, a.out_root, a.cache_dir,
                               a.docci_dir, a.mock, full=True)
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--datasets", default="*__before__simple",
                    help="comma-separated tags or patterns (fnmatch) of qa_<tag> folders")
    ap.add_argument("--n", type=int, default=50, help="random images per dataset when --per_case is empty")
    ap.add_argument("--seed", type=int, default=0, help="same seed = same images for every dataset")
    ap.add_argument("--per_case", default="P+N+H=20,P+N=10,P+H=10,P=10",
                    help="images per coverage case, e.g. 'P+N+H=20,P+N=10,P+H=10,P=10'; empty = --n random images")
    ap.add_argument("--case", default=None, choices=R.CASES, help="only images of this coverage case")
    ap.add_argument("--include_dropped", action="store_true", help="also sample images dropped for no positive")
    ap.add_argument("--cache_dir", default=os.path.join(C.WORK_ROOT, "cache"), help="where DOCCI images are")
    ap.add_argument("--docci_dir", default=None)
    ap.add_argument("--mock", action="store_true", help="draw mock images (runs made with run_all.py --mock)")
    ap.add_argument("--out", default=None, help="default <out_root>/sample_<datasets>.html")
    a = ap.parse_args()
    a.per_case = {c.strip(): int(k) for c, k in (x.split("=") for x in a.per_case.split(",") if x.strip())}
    bad = [c for c in a.per_case if c not in R.CASES]
    if bad:
        ap.error(f"--per_case: unknown case {bad}; choose from {R.CASES}")
    pats = [p.strip() for p in a.datasets.split(",") if p.strip()]
    dss = [d for d in R.datasets(a.out_root) if any(fnmatch.fnmatch(d, p) for p in pats)]
    if not dss:
        raise SystemExit(f"no qa_<tag> folder in {a.out_root} matches {a.datasets}; found: {R.datasets(a.out_root)}")
    a.mock = a.mock or os.path.exists(os.path.join(a.out_root, "mock"))
    name = "_".join(p.replace("*", "").replace("__", "_").strip("_") for p in pats) or "all"
    out = a.out or os.path.join(a.out_root, f"sample_{name}.html")
    L = ["# Sample results: " + ", ".join(dss), "",
         "Each image: photo, Caption 1 (DOCCI human description, ground truth), Caption 2 (the generated caption the "
         "QA was made from), coverage, then the Positive, Negative and Hallucinated questions with the expected answer "
         "and each VLM's full answer. `__noimage` columns are the same VLM answering without the photo. "
         "✅ true answer · ⚠️ repeats Caption 2's false claim · ❌ other wrong answer · … not graded yet.", ""]
    for ds in dss:
        L += section(a, ds)
    R.write_html("\n".join(L) + "\n", out, a.out_root, title="Sample results")
    print(f"[sample] {out} ({len(dss)} dataset(s): {', '.join(dss)})")


if __name__ == "__main__":
    main()
