"""Model x benchmark table from lmms_eval_run.py's output: <out_dir>/summary.md, summary.html, summary.json.

Every metric lmms-eval reported, for every model in the frozen plan, with the difference to --baseline
(default: the first model, normally base). Arrows say which direction is better (lmms-eval's
higher_is_better). For open-ended tasks the mean caption length in words is added, because a
longer caption moves most caption metrics on its own. summary.html has the same tables (differences
colored by better / worse) and, for each open-ended task, the same items captioned by every model.

  python lmms_eval_report.py --out_dir lmms_out
  python lmms_eval_report.py --out_dir lmms_out --with lmms_baselines   # + the prompting baselines' columns
"""
import argparse
import glob
import html
import json
import os

import numpy as np


def latest(paths):
    return max(paths, key=os.path.getmtime) if paths else None


def first_text(x):
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x if isinstance(x, str) else None


def load_group(out):
    """Aggregated metrics {task: {metric: value}}, higher_is_better, mean words {task: (words, n)}."""
    res_path = latest(glob.glob(os.path.join(out, "**", "*_results.json"), recursive=True))
    if not res_path:
        return None
    with open(res_path) as f:
        r = json.load(f)
    metrics = {}
    for task, vals in r.get("results", {}).items():
        for k, v in vals.items():
            if k == "alias" or "_stderr" in k or isinstance(v, bool):
                continue
            try:  # a numpy float (e.g. DetailCaps CAPTURE) may be stored as a string
                metrics.setdefault(task, {})[k.split(",")[0]] = float(v)
            except (TypeError, ValueError):
                continue
    words, samples = {}, {}
    stamp = os.path.basename(res_path)[:-len("_results.json")]  # samples of the same run share its date id
    for p in glob.glob(os.path.join(os.path.dirname(res_path), f"{stamp}_samples_*.jsonl")):
        task = os.path.basename(p)[len(stamp) + len("_samples_"):-len(".jsonl")]
        with open(p) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        samples[task] = {r["doc_id"]: (r.get("input", ""), first_text(r.get("filtered_resps"))) for r in rows}
        texts = [t for t in (first_text(r.get("filtered_resps")) for r in rows) if t is not None]
        if texts:
            words[task] = (float(np.mean([len(t.split()) for t in texts])), len(texts))
    return dict(path=res_path, metrics=metrics, hib=r.get("higher_is_better", {}), words=words,
                n=r.get("n-samples", {}), samples=samples)


def version(plan):
    v = plan.get("lmms_eval_version") or {}
    return f"{v.get('version')}" + (f" ({v['commit']})" if v.get("commit") else "")


def fmt(v):
    return "" if v is None else f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"


def write_html(a, plan, models, base, summary, data, n_examples):
    import report as RP  # same look as run_all's summary.html
    e = html.escape
    H = [f"<h1>lmms-eval: {e(', '.join(models))}</h1>",
         f"<p>Plan frozen {e(str(plan.get('created')))}; lmms-eval {e(version(plan))}; "
         f"model args <code>{e(plan['model_args'])}</code>; seed {plan['seed']}"
         + (f"; <strong>limit {plan['limit']} per task (smoke test)</strong>" if plan.get("limit") else "") + ".</p>",
         f"<p>Cells: value and difference to <strong>{e(base)}</strong>; green = better, red = worse "
         f"(lmms-eval's higher_is_better). Caption length is not colored: longer is not better by itself.</p>",
         "<ul>" + "".join(f"<li><strong>{e(m)}</strong>: <code>{e(p)}</code></li>" for m, p in plan["models"].items()) + "</ul>"]
    for g, spec in plan["groups"].items():
        H.append(f"<h2>{e(g)}: {e(', '.join(spec['tasks']))}</h2>")
        if spec["gen_kwargs"]:
            H.append(f"<p>gen_kwargs <code>{e(spec['gen_kwargs'])}</code></p>")
        H.append('<div class="tbl"><table><tr><th>task</th><th>metric</th>'
                 + "".join(f"<th>{e(m)}</th>" for m in models) + "</tr>")
        for task, ms in sorted(summary.get(g, {}).items()):
            for k, d in sorted(ms.items()):
                vals, hib = d["values"], d["higher_is_better"]
                arrow = "" if hib is None else " ↑" if hib else " ↓"
                row = f"<tr><td>{e(task)}</td><td>{e(k)}{arrow}</td>"
                for m in models:
                    v, b = vals.get(m), vals.get(base)
                    if v is None:
                        row += '<td class="none">–</td>'
                    elif m == base or b is None:
                        row += f'<td class="num">{fmt(v)}</td>'
                    else:
                        cls = "" if hib is None or v == b else " hi" if (v > b) == bool(hib) else " vlo"
                        row += f'<td class="num{cls}">{fmt(v)} ({v - b:+.4f})</td>'
                H.append(row + "</tr>")
        H.append("</table></div>")
        if g not in ("caption", "oeq") or not n_examples:  # example captions: the caption group only
            continue
        for task in spec["tasks"]:
            per = {m: (data[m][g] or {}).get("samples", {}).get(task) for m in models}
            if not all(per.values()):
                continue
            ids = sorted(set.intersection(*(set(v) for v in per.values())))[:n_examples]
            if not ids:
                continue
            H.append(f"<h3>{e(task)}: same items, every model</h3>")
            for i in ids:
                prompt = per[models[0]][i][0]
                H.append(f"<details><summary>item {e(str(i))}: {e(prompt[:120])}</summary>"
                         '<div class="tbl"><table><tr><th>model</th><th>words</th><th>output</th></tr>'
                         + "".join(f"<tr><td>{e(m)}</td><td class=\"num\">{len((per[m][i][1] or '').split())}</td>"
                                   f"<td>{e(per[m][i][1] or '')}</td></tr>" for m in models)
                         + "</table></div></details>")
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>lmms-eval results</title>'
            f'{RP.FONTS}<style>{RP.CSS}</style></head><body><main>{"".join(H)}</main></body></html>')
    with open(os.path.join(a.out_dir, "summary.html"), "w") as f:
        f.write(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--baseline", default=None, help="model the differences are taken against (default: first)")
    ap.add_argument("--with", dest="with_dirs", default="", help="other out_dirs (comma list) whose models are added "
                    "as columns, e.g. the prompting baselines' (lmms_eval_run.py --settings_from this plan)")
    ap.add_argument("--examples", type=int, default=5, help="items per open-ended task shown in summary.html")
    a = ap.parse_args()
    with open(os.path.join(a.out_dir, "plan.json")) as f:
        plan = json.load(f)
    models = list(plan["models"])
    base = a.baseline or models[0]
    data = {m: {g: load_group(os.path.join(a.out_dir, m, g)) for g in plan["groups"]} for m in models}
    na = set()  # cells a prompting baseline skips: its method changes none of the group's tasks
    for d in [x for x in a.with_dirs.split(",") if x.strip()]:
        with open(os.path.join(d, "plan.json")) as f:
            other = json.load(f)
        diff = [g for g in other["groups"] if g in plan["groups"] and other["groups"][g]["tasks"] != plan["groups"][g]["tasks"]]
        if other["model_args"] != plan["model_args"] or other["seed"] != plan["seed"] or diff:
            print(f"[warn] {d}: settings differ from {a.out_dir} (model_args / seed / tasks of {diff}); compare with care")
        for m in other["models"]:
            if m in models:
                continue
            models.append(m)
            plan["models"][m] = other["models"][m]
            data[m] = {g: load_group(os.path.join(d, m, g)) if g in other["groups"] else None for g in plan["groups"]}
            p = (other.get("prompting") or {}).get(m)
            for g in plan["groups"]:
                if p and p["applies"] is not None and not any(t.startswith(tuple(p["applies"])) for t in plan["groups"][g]["tasks"]):
                    na.add(f"{m}/{g}")
        plan.setdefault("prompting", {}).update(other.get("prompting") or {})

    lines = [f"# lmms-eval: {', '.join(models)}", "",
             f"Plan frozen {plan.get('created')}; lmms-eval {version(plan)}; model args "
             f"`{plan['model_args']}`; seed {plan['seed']}" + (f"; **limit {plan['limit']} per task (smoke test)**"
                                                                  if plan.get("limit") else "") + ".",
             f"Cells: value (difference to {base}). ↑ / ↓ = higher / lower is better.", ""]
    missing = [f"{m}/{g}" for m in models for g in plan["groups"] if data[m][g] is None and f"{m}/{g}" not in na]
    if missing:
        lines += [f"Not run yet: {', '.join(missing)}", ""]
    if plan.get("prompting"):
        lines += ["Prompting baselines (base model + another paper's method; settings and deviations in BASELINES.md): "
                  + ", ".join(f"{m} = {p['method']}" for m, p in plan["prompting"].items() if m in models)
                  + (f". Not applicable (the method changes none of the tasks; empty cells): {', '.join(sorted(na))}"
                     if na else ""), ""]
    summary = {}
    for g, spec in plan["groups"].items():
        lines += [f"## {g} ({', '.join(spec['tasks'])})" + (f", gen_kwargs `{spec['gen_kwargs']}`" if spec["gen_kwargs"] else ""),
                  "", "| task | metric | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        rows = {}
        for m in models:
            d = data[m][g]
            for task, ms in (d["metrics"] if d else {}).items():
                for k, v in ms.items():
                    rows.setdefault((task, k), {})[m] = v
            for task, (w, n) in (d["words"] if d and g in ("caption", "oeq") else {}).items():
                rows.setdefault((task, "words per caption"), {})[m] = w
        for (task, k), vals in sorted(rows.items()):
            hib = next((data[m][g]["hib"].get(task, {}).get(k) for m in models
                        if data[m][g] and k in data[m][g]["hib"].get(task, {})), None)
            arrow = "" if hib is None else " ↑" if hib else " ↓"
            cells = []
            for m in models:
                v = vals.get(m)
                if v is None:
                    cells.append("")
                elif m == base or vals.get(base) is None:
                    cells.append(fmt(v))
                else:
                    cells.append(f"{fmt(v)} ({v - vals[base]:+.4f})")
            lines.append(f"| {task} | {k}{arrow} | " + " | ".join(cells) + " |")
            summary.setdefault(g, {}).setdefault(task, {})[k] = dict(values=vals, higher_is_better=hib)
        lines.append("")
    lines += ["Result files:", ""] + [f"- {m}/{g}: `{data[m][g]['path']}`" for m in models for g in plan["groups"]
                                      if data[m][g]]
    with open(os.path.join(a.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(dict(baseline=base, models=plan["models"], groups=summary), f, indent=2, ensure_ascii=False)
    write_html(a, plan, models, base, summary, data, a.examples)
    print("\n".join(lines))
    print(f"\nwrote {a.out_dir}/summary.md and summary.html")


if __name__ == "__main__":
    main()
