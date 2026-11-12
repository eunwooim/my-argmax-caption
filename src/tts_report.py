"""Reward comparison report: test-time scaling (Use 1), GRPO with each reward (Use 2), and the combination
(README "Reward comparison"). Reads lmms-eval's per-item sample files, so every number comes with a 95%
bootstrap interval over images, and every "ours - other" difference is paired (same images, same resamples).

  Use 1        lmms_eval_tts/base-sample (N = 1) and base-<selector>-n<N> (N = 2, 4, 8, 16)
  Combination  lmms_eval_tts/selfgrpo-sample (N = 1) and selfgrpo-self-n<N>: the self-reward GRPO model's samples,
               reranked by the self reward
  Use 2        lmms_eval_out/<model>/caption_long: greedy captions of base and the six GRPO runs
  Agreement    CaptionQA questions where the judge's answer from the caption equals the captioning model's own
               answer seeing the image (tts.py vqa: tts_out/vqa/<model>.jsonl); "cannot answer" counts as a
               disagreement, questions without a parsed image answer are left out

  python tts_report.py --tts lmms_eval_tts --checkpoints lmms_eval_out --vqa_dir tts_out/vqa --out_stem tts_out/reward_comparison
"""
import argparse
import glob
import html
import importlib.util
import json
import math
import os

import numpy as np

SELECTORS = [  # fixed order = fixed color slot
    ("self", "ours (S2)"), ("random", "random"), ("shortest", "shortest"), ("selfcert", "self-certainty"),
    ("longclip", "Long-CLIPScore"), ("lpfv", "Self-Validation (LPFV)"), ("jina", "jina-clip-v2 (embedder)")]
REWARDS = [  # Use 2: model name in lmms_eval_out -> label
    ("self", "ours (S2)"), ("random", "random"), ("length", "length only"), ("selfcert", "self-certainty (Intuitor)"),
    ("longclip", "Long-CLIPScore"), ("external", "jina-clip-v2 (embedder)")]
COLORS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
NS = [1, 2, 4, 8, 16]
CAPTIONQA = ["captionqa_x_natural", "captionqa_x_document", "captionqa_x_ecommerce", "captionqa_x_embodiedai"]
# (key, label, higher is better, benchmark)
METRICS = [
    ("cqa_score", "CaptionQA score", True, "captionqa"),
    ("cqa_agree", "CaptionQA agreement with own image answers", True, "captionqa"),
    ("chair_s", "CHAIR-500 CHAIR_s", False, "chair500"),
    ("chair_i", "CHAIR-500 CHAIR_i", False, "chair500"),
    ("chair_recall", "CHAIR-500 object recall", True, "chair500"),
    ("chair_f1", "CHAIR-500 object F1", True, "chair500"),
    ("amber_chair", "AMBER CHAIR", False, "amber"),
    ("amber_cover", "AMBER Cover", True, "amber"),
    ("amber_hal", "AMBER Hal", False, "amber"),
    ("amber_cog", "AMBER Cog", False, "amber"),
    ("words_cqa", "words per caption (CaptionQA)", None, "captionqa"),
    ("words_chair", "words per caption (CHAIR-500)", None, "chair500"),
    ("words_amber", "words per caption (AMBER)", None, "amber"),
]
PLOTTED = ["cqa_score", "cqa_agree", "chair_i", "chair_f1", "amber_hal", "amber_cover", "words_chair"]


def first(x):
    while isinstance(x, list):
        x = x[0] if x else ""
    return x


def sample_file(cell, task):
    fs = glob.glob(os.path.join(cell, "**", f"*_samples_{task}.jsonl"), recursive=True)
    return max(fs, key=os.path.getmtime) if fs else None


def read_rows(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class ChairScorer:
    """Self-Validation's chair.py through the task's own chair500_utils (same evaluator, same pickle), per caption."""

    def __init__(self, tasks_dir):
        spec = importlib.util.spec_from_file_location("_report_chair500", os.path.join(tasks_dir, "chair500_utils.py"))
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def per_caption(self, images, captions):
        """The task's own per-caption counts (chair.py's loop) and totals, so the report equals the task's results."""
        rows = self.mod.per_caption(images, captions)
        return np.array([r[:6] for r in rows], dtype=float), self.mod.overall(rows)


# ------------------------------------------------------------------ per-cell items
def cell_items(cell, vqa, chair):
    """{benchmark: (keys, array of per-image numbers)}; a benchmark whose sample files are missing is absent."""
    if not os.path.exists(os.path.join(cell, "done.json")):
        return None
    out = {}
    # CaptionQA: per image [sum score, questions, agreements, questions with an image answer, words]
    keys, rows = [], []
    for task in CAPTIONQA:
        p = sample_file(cell, task)
        if not p:
            continue
        for r in read_rows(p):
            qs = r.get("captionqa_score") or []
            agree = n_img = 0
            for q in qs:
                img = vqa.get((str(q["image_id"]), q["question_idx"])) if vqa is not None else None
                if img is not None:
                    n_img += 1
                    agree += int(q.get("chosen") == img)
            keys.append(f"{task}/{r['doc_id']}")
            rows.append([sum(q["score"] for q in qs), len(qs), agree, n_img, len(str(first(r["filtered_resps"])).split())])
    if rows:
        out["captionqa"] = (keys, np.array(rows, dtype=float))
    p = sample_file(cell, "chair500")
    if p:
        rs = sorted(read_rows(p), key=lambda r: r["doc_id"])
        caps = [str(first(r["filtered_resps"])) for r in rs]
        arr, overall = chair.per_caption([r["target"] for r in rs], caps)
        words = np.array([[len(c.split())] for c in caps], dtype=float)
        out["chair500"] = ([f"chair500/{r['doc_id']}" for r in rs], np.hstack([arr, words]))
    p = sample_file(cell, "amber_g")
    if p:
        rs = sorted(read_rows(p), key=lambda r: r["doc_id"])
        names = ["chair_score", "chair_num", "safe_cover_score", "safe_cover_num", "non_hallu_score", "non_hallu_num",
                 "hallu_cover_score", "hallu_cover_num"]
        arr = [[float(r["amber_chair"].get(n, 0) or 0) for n in names] + [len(str(first(r["filtered_resps"])).split())]
               for r in rs]
        out["amber"] = ([f"amber/{r['doc_id']}" for r in rs], np.array(arr, dtype=float))
    return out


def stats(bench, arr):
    """Metric values from the per-image numbers (sums of the resampled rows), the benchmarks' own formulas."""
    s = arr.sum(0)
    n = len(arr)
    div = lambda a, b: float("nan") if b == 0 else a / b
    if bench == "captionqa":
        return dict(cqa_score=div(s[0], s[1]), cqa_agree=100 * div(s[2], s[3]), words_cqa=s[4] / n)
    if bench == "chair500":
        rec, prec = div(s[4], s[3]), div(s[4], s[5])
        return dict(chair_s=100 * s[0] / n, chair_i=100 * div(s[1], s[2]), chair_recall=100 * rec,
                    chair_f1=100 * div(2 * rec * prec, rec + prec), words_chair=s[6] / n)
    return dict(amber_chair=100 * div(s[0], s[1]), amber_cover=100 * div(s[2], s[3]),
                amber_hal=100 - 100 * div(s[4], s[5]), amber_cog=100 * div(s[6], s[7]), words_amber=s[8] / n)


class Boot:
    """One set of resample indices per benchmark (over its sorted image keys), shared by every cell: intervals
    of a cell and of a paired difference both come from it."""

    def __init__(self, B, seed):
        self.B, self.rng, self.idx = B, np.random.default_rng(seed), {}

    def indices(self, bench, n):
        if (bench, n) not in self.idx:
            self.idx[(bench, n)] = self.rng.integers(0, n, size=(self.B, n))
        return self.idx[(bench, n)]

    def cell(self, items):
        """{metric: (value, lo, hi, samples)}"""
        res = {}
        for bench, (keys, arr) in items.items():
            order = np.argsort(keys)
            arr = arr[order]
            point = stats(bench, arr)
            draws = [stats(bench, arr[ix]) for ix in self.indices(bench, len(arr))]
            for m, v in point.items():
                d = np.array([x[m] for x in draws])
                res[m] = (v, float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), d, tuple(sorted(keys)))
        return res


def paired(a, b):
    """ours - other with its 95% interval (same resamples); None when the two cells hold different images."""
    if a is None or b is None or a[4] != b[4]:
        return None
    d = a[3] - b[3]
    return a[0] - b[0], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


# ------------------------------------------------------------------ text
def fmt(x, digits):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def digits_of(m):
    return 3 if m == "cqa_score" else 1


def cell_text(r, m):
    if r is None or m not in r:
        return "not run"
    v, lo, hi = r[m][:3]
    d = digits_of(m)
    return f"{fmt(v, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def diff_text(p, m, better):
    if p is None:
        return "–"
    v, lo, hi = p
    d = digits_of(m)
    tag = ""
    if better is not None and (lo > 0 or hi < 0):
        tag = " ✓" if (lo > 0) == better else " ✗"
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{tag}"


def rank_of(results, names, m, better):
    """(rank of names[0], rows ranked, rows tied with it): rank = 1 + rows strictly better, so a tie never counts
    in our favour."""
    vals = {n: results[n][m][0] for n in names if results.get(n) and m in results[n] and not math.isnan(results[n][m][0])}
    if better is None or names[0] not in vals:
        return None
    ours = vals[names[0]]
    others = [v for n, v in vals.items() if n != names[0]]
    ahead = sum((v > ours) if better else (v < ours) for v in others)
    return 1 + ahead, len(vals), sum(v == ours for v in others)


def rank_text(rk):
    if rk is None:
        return "–"
    return f"{rk[0]} / {rk[1]}" + (f" (tied with {rk[2]})" if rk[2] else "")


# ------------------------------------------------------------------ charts (inline SVG, one per metric)
def svg_chart(title, better, series, ns, dark_ids):
    """series: [(label, slot, [(n, v, lo, hi) or None], dashed)]; log2 x axis; hover = <title> per point."""
    W, H, L, R, T, Bm = 340, 220, 48, 108, 28, 30
    pts = [p for _, _, ps, _ in series for p in ps if p]
    if not pts:
        return f"<figure class='chart'><figcaption>{html.escape(title)}</figcaption><p class='muted'>not run yet</p></figure>"
    lo = min(p[2] for p in pts)
    hi = max(p[3] for p in pts)
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    xs = {n: L + (W - L - R) * (math.log2(n) / math.log2(max(ns))) for n in ns}
    y = lambda v: T + (H - T - Bm) * (1 - (v - lo) / (hi - lo))
    g = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='{html.escape(title)}'>"]
    for k in range(5):  # recessive grid + y labels
        v = lo + (hi - lo) * k / 4
        g.append(f"<line class='grid' x1='{L}' x2='{W - R}' y1='{y(v):.1f}' y2='{y(v):.1f}'/>"
                 f"<text class='tick' x='{L - 6}' y='{y(v) + 4:.1f}' text-anchor='end'>{v:.3g}</text>")
    for n in ns:
        g.append(f"<text class='tick' x='{xs[n]:.1f}' y='{H - 10}' text-anchor='middle'>{n}</text>")
    g.append(f"<text class='tick' x='{(L + W - R) / 2:.1f}' y='{H}' text-anchor='middle'>N candidates</text>")
    ends = []
    for label, slot, ps, dashed in series:
        ok = [p for p in ps if p]
        if not ok:
            continue
        cls = f"s{slot}" + (" dash" if dashed else "")
        path = " ".join(f"{'M' if i == 0 else 'L'}{xs[p[0]]:.1f},{y(p[1]):.1f}" for i, p in enumerate(ok))
        g.append(f"<path class='line {cls}' d='{path}'/>")
        for p in ok:
            g.append(f"<circle class='dot s{slot}' cx='{xs[p[0]]:.1f}' cy='{y(p[1]):.1f}' r='4'><title>{html.escape(label)}, "
                     f"N={p[0]}: {p[1]:.4g} [{p[2]:.4g}, {p[3]:.4g}]</title></circle>")
        ends.append((y(ok[-1][1]), label))
    ends.sort()
    last = -1e9
    for yy, label in ends:  # direct labels at the line ends, nudged apart
        yy = max(yy, last + 11)
        last = yy
        g.append(f"<text class='lab' x='{W - R + 6}' y='{yy + 4:.1f}'>{html.escape(label)}</text>")
    g.append("</svg>")
    arrow = "" if better is None else (" ↑" if better else " ↓")
    return f"<figure class='chart'><figcaption>{html.escape(title)}{arrow}</figcaption>{''.join(g)}</figure>"


CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--grid:#e6e9ec;--head:#f6f8fa;
%s}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--fg:#f0efec;
--muted:#c3c2b7;--line:#3d444d;--grid:#2c2c2a;--head:#232322;%s}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3d444d;--grid:#2c2c2a;
--head:#232322;%s}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
margin:0 auto;max-width:1180px;padding:24px 16px}
h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:32px 0 10px}.muted{color:var(--muted)}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%%}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}th{background:var(--head)}
td:first-child,th:first-child{text-align:left}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
figure.chart{margin:0}figcaption{font-weight:600;font-size:13px;margin-bottom:4px}
svg{width:100%%;height:auto;overflow:visible}.grid{stroke:var(--grid);stroke-width:1}
.tick,.lab{fill:var(--muted);font-size:10px}.lab{fill:var(--fg)}
.line{fill:none;stroke-width:2}.dash{stroke-dasharray:5 4}.dot{stroke:var(--bg);stroke-width:2}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:8px 0 12px;font-size:13px}
.legend span::before{content:"";display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle;
background:var(--c)}
"""


def css():
    def vars_(cols):
        return "".join(f"--s{i}:{c};" for i, c in enumerate(cols))
    rules = "".join(f".s{i}{{stroke:var(--s{i})}}circle.s{i}{{fill:var(--s{i})}}" for i in range(len(COLORS_LIGHT)))
    return CSS % (vars_(COLORS_LIGHT), vars_(COLORS_DARK), vars_(COLORS_DARK)) + rules


# ------------------------------------------------------------------ report
def load_vqa(path):
    if not path or not os.path.exists(path):
        return None
    return {(str(r["image_id"]), r["question_idx"]): r["chosen"] for r in read_rows(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tts", default="lmms_eval_tts")
    ap.add_argument("--checkpoints", default="lmms_eval_out")
    ap.add_argument("--vqa_dir", default="tts_out/vqa")
    ap.add_argument("--tasks_dir", default="/mnt/data2/mchung45/lmms_eval_work/custom_tasks")
    ap.add_argument("--group", default="caption_long")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_stem", default="tts_out/reward_comparison")
    a = ap.parse_args()
    chair = ChairScorer(a.tasks_dir)
    boot = Boot(a.bootstrap, 0)
    vqa = {n: load_vqa(os.path.join(a.vqa_dir, f"{n}.jsonl"))
           for n in ["base"] + [m for m, _ in REWARDS]}
    missing_vqa = [n for n, v in vqa.items() if v is None]

    def res(cell, vqa_name):
        items = cell_items(cell, vqa[vqa_name], chair)
        return boot.cell(items) if items else None

    # Use 1 and the combination
    def n1(prefix, vqa_name):  # candidate 0: its replay cell (vLLM candidates), else the HF sampling cell's answer
        c = os.path.join(a.tts, f"{prefix}-first-n1", a.group)
        return res(c if os.path.exists(c) else os.path.join(a.tts, f"{prefix}-sample", a.group), vqa_name)

    use1 = {}
    base_n1 = n1("base", "base")
    for s, _ in SELECTORS:
        use1[s] = {1: base_n1}
        for n in NS[1:]:
            use1[s][n] = res(os.path.join(a.tts, f"base-{s}-n{n}", a.group), "base")
    comb = {1: n1("selfgrpo", "self")}
    for n in NS[1:]:
        comb[n] = res(os.path.join(a.tts, f"selfgrpo-self-n{n}", a.group), "self")
    # Use 2
    use2 = {"base": res(os.path.join(a.checkpoints, "base", a.group), "base")}
    for m, _ in REWARDS:
        use2[m] = res(os.path.join(a.checkpoints, m, a.group), m)

    nmax = max(NS)
    md, h = [], []
    title = "Reward comparison: test-time scaling and GRPO"
    md += [f"# {title}", "",
           "Same benchmarks for both uses: CaptionQA (score, and agreement of the caption-only answers with the "
           "captioning model's own image answers), Self-Validation's CHAIR-500 (CHAIR and object F1), AMBER, caption "
           "length. Captions have no length cap (4096 tokens). Values: point estimate [95% bootstrap interval over "
           "images]. ✓ / ✗: the paired interval of ours − other excludes 0 in our favour / against us.", ""]
    h += [f"<title>{title}</title><style>{css()}</style><h1>{title}</h1>",
          "<p class='muted'>Same benchmarks for both uses. Captions without a length cap. Point estimate [95% bootstrap "
          "interval over images]; ✓ / ✗ = the paired interval of ours − other excludes 0 for / against us.</p>"]
    if missing_vqa:
        md += [f"> Agreement not available for: {', '.join(missing_vqa)} (tts.py vqa not run).", ""]
        h.append(f"<p class='muted'>Agreement not available for: {html.escape(', '.join(missing_vqa))}.</p>")

    def table(rows, header, md_out, h_out):
        md_out += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        md_out += ["| " + " | ".join(r) + " |" for r in rows]
        md_out.append("")
        h_out.append("<div class='scroll'><table><tr>" + "".join(f"<th>{html.escape(x)}</th>" for x in header) + "</tr>"
                     + "".join("<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in r) + "</tr>" for r in rows)
                     + "</table></div>")

    metric_cols = [(k, lab + ("" if b is None else (" ↑" if b else " ↓")), b) for k, lab, b, _ in METRICS]

    # ---- Use 1
    md += ["## Use 1: test-time scaling (base model, no training)", "",
           f"The base model samples 16 captions per image (Self-Validation's sampling: temperature 0.5, top-k 50, "
           f"top-p 1; Qwen's repetition penalty 1.05; vLLM, same prompts and pixels as lmms-eval); each selector keeps "
           f"one of the first N. N = 1 is the same candidate for every selector.", ""]
    h += ["<h2>Use 1: test-time scaling (base model, no training)</h2>",
          "<p>16 sampled captions per image (temperature 0.5, top-k 50); each selector keeps one of the first N. Dashed: "
          "the self-reward GRPO model's samples reranked by ours (combination).</p>"]
    h.append("<div class='legend'>" + "".join(f"<span style='--c:var(--s{i})'>{html.escape(lab)}</span>"
                                               for i, (_, lab) in enumerate(SELECTORS))
             + "<span style='--c:var(--s0)'>GRPO(ours) + ours selector (dashed)</span></div><div class='charts'>")
    for key, lab, better, _ in METRICS:
        if key not in PLOTTED:
            continue
        series = []
        for i, (s, slab) in enumerate(SELECTORS):
            series.append((slab, i, [(n, *use1[s][n][key][:3]) if use1[s][n] and key in use1[s][n] else None for n in NS], False))
        series.append(("GRPO(ours)+ours", 0, [(n, *comb[n][key][:3]) if comb[n] and key in comb[n] else None
                                              for n in NS], True))
        h.append(svg_chart(lab, better, series, NS, None))
    h.append("</div>")
    rows = []
    for s, slab in SELECTORS:
        rows.append([slab] + [cell_text(use1[s][nmax], k) for k, _, _ in metric_cols])
    md += [f"### All selectors at N = {nmax}", ""]
    h.append(f"<h3>All selectors at N = {nmax}</h3>")
    table(rows, ["selector"] + [lab for _, lab, _ in metric_cols], md, h)
    rows = []
    for s, slab in SELECTORS[1:]:
        rows.append([f"ours − {slab}"] + [diff_text(paired((use1['self'][nmax] or {}).get(k), (use1[s][nmax] or {}).get(k)), k, b)
                                         for k, _, b in metric_cols if b is not None])
    md += [f"### Ours minus each selector at N = {nmax} (paired)", ""]
    h.append(f"<h3>Ours minus each selector at N = {nmax} (paired)</h3>")
    table(rows, ["comparison"] + [lab for _, lab, b in metric_cols if b is not None], md, h)
    rows = []
    for k, lab, b in metric_cols:
        r = [lab]
        for n in NS[1:]:  # N = 1 is the same candidate for every selector: no ranking
            r.append(rank_text(rank_of({s: use1[s][n] for s, _ in SELECTORS}, [s for s, _ in SELECTORS], k, b)))
        rows.append(r)
    md += ["### Rank of ours among the selectors (1 = best; N = 1 is the same caption for all)", ""]
    h.append("<h3>Rank of ours among the selectors (1 = best; N = 1 is the same caption for all)</h3>")
    table([r for r in rows if any(x != "–" for x in r[1:])], ["metric"] + [f"N={n}" for n in NS[1:]], md, h)

    # ---- Use 2
    md += ["## Use 2: GRPO with each reward (same config, greedy captions)", ""]
    h.append("<h2>Use 2: GRPO with each reward (same config, greedy captions)</h2>")
    rows = [["base (no training)"] + [cell_text(use2["base"], k) for k, _, _ in metric_cols]]
    rows += [[lab] + [cell_text(use2[m], k) for k, _, _ in metric_cols] for m, lab in REWARDS]
    table(rows, ["reward"] + [lab for _, lab, _ in metric_cols], md, h)
    rows = []
    for m, lab in [("base", "base")] + REWARDS[1:]:
        rows.append([f"ours − {lab}"] + [diff_text(paired((use2['self'] or {}).get(k), (use2[m] or {}).get(k)), k, b)
                                        for k, _, b in metric_cols if b is not None])
    md += ["### Ours minus each reward (paired)", ""]
    h.append("<h3>Ours minus each reward (paired)</h3>")
    table(rows, ["comparison"] + [lab for _, lab, b in metric_cols if b is not None], md, h)
    rows = []
    for k, lab, b in metric_cols:
        rk = rank_of({m: use2[m] for m, _ in REWARDS}, [m for m, _ in REWARDS], k, b)
        if rk:
            rows.append([lab, rank_text(rk)])
    md += ["### Rank of ours among the rewards (1 = best)", ""]
    h.append("<h3>Rank of ours among the rewards (1 = best)</h3>")
    table(rows, ["metric", "rank"], md, h)

    # ---- combination
    md += ["## Combination: rerank the GRPO model's samples with our reward", ""]
    h.append("<h2>Combination: rerank the GRPO model's samples with our reward</h2>")
    rows = [["GRPO(ours), greedy"] + [cell_text(use2["self"], k) for k, _, _ in metric_cols]]
    rows += [[f"GRPO(ours) + ours selector, N={n}"] + [cell_text(comb[n], k) for k, _, _ in metric_cols] for n in NS]
    rows.append([f"base + ours selector, N={nmax}"] + [cell_text(use1["self"][nmax], k) for k, _, _ in metric_cols])
    table(rows, ["setting"] + [lab for _, lab, _ in metric_cols], md, h)
    rows = [[f"GRPO(ours)+ours N={nmax} − GRPO(ours) greedy"] +
            [diff_text(paired((comb[nmax] or {}).get(k), (use2['self'] or {}).get(k)), k, b) for k, _, b in metric_cols if b is not None],
            [f"GRPO(ours)+ours N={nmax} − base+ours N={nmax}"] +
            [diff_text(paired((comb[nmax] or {}).get(k), (use1['self'][nmax] or {}).get(k)), k, b) for k, _, b in metric_cols if b is not None]]
    table(rows, ["comparison"] + [lab for _, lab, b in metric_cols if b is not None], md, h)

    notes = [
        "The comparison that carries the claim is within each use (selectors in Use 1, rewards in Use 2); best-of-N "
        "and GRPO are shown together but not compared with each other.",
        "Long-CLIP reads at most 248 tokens of a caption (longer captions are cut, as its tokenizer does); "
        "Self-Validation's LPFV checks MSCOCO objects only, so captions without such objects score 1.0 (its rule).",
        "CaptionQA is judged by the local text LLM (not lmms-eval's default 72B judge): absolute values are not "
        "comparable with other papers, differences between our rows are.",
        "AMBER Cog: lower is better (AMBER's definition; lmms-eval's task file marks it otherwise).",
        "Intervals: 95% percentile bootstrap over images (CaptionQA: an image's questions stay together), "
        f"{a.bootstrap} resamples shared by all rows, so ours − other intervals are paired.",
    ]
    md += ["## Notes", ""] + [f"- {n}" for n in notes]
    h.append("<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
    os.makedirs(os.path.dirname(os.path.abspath(a.out_stem)), exist_ok=True)
    with open(a.out_stem + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(a.out_stem + ".html", "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                + "".join(h))
    print(f"wrote {a.out_stem}.md and {a.out_stem}.html")


if __name__ == "__main__":
    main()
