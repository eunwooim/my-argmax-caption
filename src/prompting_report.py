"""Compare the GRPO checkpoints with the prompting baselines on the benchmarks where self showed an edge.

Reads the frozen checkpoint results (lmms_eval_out) and the baselines' results (lmms_eval_baselines) and writes one
Markdown and one self-contained HTML file. lmms_eval_out/summary.* are not touched.

The three lists were chosen after seeing the checkpoint results, so this is a focused view and not the evaluation
itself: report it together with the full table (lmms_eval_report.py). Every value here is a point estimate; no
significance test is run.

  python prompting_report.py --checkpoints lmms_eval_out --baselines lmms_eval_baselines
  python prompting_report.py --checkpoints lmms_eval_out --baselines lmms_eval_baselines_quick   # marked as a smoke test
"""
import argparse
import html
import json
import os

from lmms_eval_report import fmt, load_group

CHECKPOINTS = ("base", "self", "external", "length")
BASELINES = ("selfval_bon", "selfval_fta", "logiccheck")
EPS = 1e-9

# (label, group, task, metric); lower_is_better comes from the results files
CATEGORIES = [
    ("base와 두 대조군을 모두 이김", "beats_all", [
        ("MMStar average", "mcq", "mmstar", "average"),
        ("MMStar fine-grained perception", "mcq", "mmstar", "fine-grained perception"),
        ("MMBench", "general", "mmbench_en_dev_static", "gpt_eval_score"),
        ("MME Perception", "general", "mme", "mme_perception_score"),
    ]),
    ("base는 이기지만 대조군까지 다 이기지는 못함", "beats_base", [
        ("COCO Recall", "caption", "coco_cap_chair", "coco_cap_recall"),
        ("MME Cognition", "general", "mme", "mme_cognition_score"),
        ("MMStar instance reasoning", "mcq", "mmstar", "instance reasoning"),
        ("MMStar logical reasoning", "mcq", "mmstar", "logical reasoning"),
        ("COCO CHAIR_i", "caption", "coco_cap_chair", "coco_cap_chair_i"),
    ]),
    ("base에는 지지만 대조군은 이김", "beats_controls", [
        ("MM-Vet", "general", "mmvet", "gpt_eval_score"),
        ("DOCCI CAPTURE", "caption", "docci_caption", "docci_CAPTURE"),
        ("DOCCI BLEU-4", "caption", "docci_caption", "docci_Bleu_4"),
        ("DOCCI METEOR", "caption", "docci_caption", "docci_METEOR"),
        ("DOCCI ROUGE-L", "caption", "docci_caption", "docci_ROUGE_L"),
    ]),
]
WORD_ROWS = [("COCO", "coco_cap_chair"), ("DOCCI", "docci_caption")]


def read_plan(d):
    path = os.path.join(d, "plan.json")
    if not os.path.exists(path):
        raise SystemExit(f"[report] {path} is missing: run the baselines first")
    with open(path) as f:
        return json.load(f)


def compare(a, b, higher):
    """'win' when a is better than b."""
    if abs(a - b) <= EPS:
        return "tie"
    return "win" if (a > b) == higher else "loss"


def collect(ckpt_dir, base_dir):
    ck_plan, bl_plan = read_plan(ckpt_dir), read_plan(base_dir)
    groups = {g for _, _, rows in CATEGORIES for _, g, _, _ in rows}
    data = {m: {g: load_group(os.path.join(ckpt_dir, m, g)) for g in groups} for m in CHECKPOINTS}
    missing_method = [m for m in BASELINES if m not in bl_plan["models"]]
    if missing_method:
        raise SystemExit(f"[report] {base_dir}/plan.json has no {missing_method}")
    for m in BASELINES:
        data[m] = {g: load_group(os.path.join(base_dir, m, g)) if g in bl_plan["groups"] else None for g in groups}

    def task_applies(method, task):
        applies = bl_plan["prompting"][method]["applies"]
        return applies is None or task.startswith(tuple(applies))

    warnings, cats = [], []
    for k in ("model_args", "seed", "batch_size"):
        if ck_plan.get(k) != bl_plan.get(k):
            warnings.append(f"설정이 다름: {k} (체크포인트 {ck_plan.get(k)!r}, 베이스라인 {bl_plan.get(k)!r})")
    for g in groups:
        ck, bl = ck_plan["groups"].get(g, {}), bl_plan["groups"].get(g, {})
        if ck.get("gen_kwargs") != bl.get("gen_kwargs"):
            warnings.append(f"생성 설정이 다름: {g} (체크포인트 {ck.get('gen_kwargs')!r}, 베이스라인 {bl.get('gen_kwargs')!r})")

    not_run = set()
    for title, key, rows in CATEGORIES:
        out = []
        for label, g, task, metric in rows:
            values, status, higher = {}, {}, None
            for m in CHECKPOINTS + BASELINES:
                if m in BASELINES and not task_applies(m, task):
                    status[m] = "na"
                    continue
                d = data[m][g]
                v = d["metrics"].get(task, {}).get(metric) if d else None
                if v is None:
                    status[m] = "missing"
                    not_run.add(f"{m}/{g}")
                    continue
                status[m], values[m] = "ok", v
                if higher is None:
                    higher = d["hib"].get(task, {}).get(metric)
            higher = True if higher is None else bool(higher)
            verdict = {}
            if "self" in values:
                verdict = {m: compare(values["self"], values[m], higher) for m in BASELINES if m in values}
            best = None
            if values:
                best = (max if higher else min)(values.values())
            out.append(dict(label=label, group=g, task=task, metric=metric, values=values, status=status,
                            higher=higher, verdict=verdict, best=best))
        # the lists were written from the frozen results; say so if those results have changed since
        for r in out:
            if not all(m in r["values"] for m in CHECKPOINTS):
                continue
            v, h = r["values"], r["higher"]
            vs_base = compare(v["self"], v["base"], h)
            vs_ctrl = [compare(v["self"], v[c], h) for c in ("external", "length")]
            ok = {"beats_all": vs_base == "win" and all(x == "win" for x in vs_ctrl),
                  "beats_base": vs_base == "win" and any(x != "win" for x in vs_ctrl),
                  "beats_controls": vs_base == "loss" and "win" in vs_ctrl}[key]
            if not ok:
                warnings.append(f"{r['label']}: 체크포인트 결과가 이 분류({title})와 맞지 않음")
        cats.append(dict(title=title, rows=out))

    words = []
    for label, task in WORD_ROWS:
        vals = {}
        for m in CHECKPOINTS + BASELINES:
            d = data[m].get("caption")
            if d and task in d["words"]:
                vals[m] = d["words"][task][0]
        words.append((label, vals))
    tally = {m: {"win": 0, "tie": 0, "loss": 0} for m in BASELINES}
    for c in cats:
        for r in c["rows"]:
            for m, res in r["verdict"].items():
                tally[m][res] += 1
    ns = {}
    for m in BASELINES:
        for g in groups:
            d = data[m][g]
            if d:
                for t, n in (d.get("n") or {}).items():
                    ns.setdefault(t, {})[m] = n.get("effective") if isinstance(n, dict) else n
    return dict(cats=cats, words=words, tally=tally, warnings=warnings, not_run=sorted(not_run),
                limit=bl_plan.get("limit"), skipped=bl_plan.get("skip_tasks") or [],
                created=bl_plan.get("created"), ckpt_dir=ckpt_dir, base_dir=base_dir, ns=ns)


COLS = CHECKPOINTS + BASELINES
VERDICT_KO = {"win": "이김", "tie": "동점", "loss": "짐"}
NOTES = [
    "이 세 목록은 체크포인트 결과를 본 뒤 고른 것입니다. 논문에는 전체 표(`lmms_eval_out/summary.md`)와 함께 보고해야 합니다.",
    "모든 값은 점 추정치입니다. 통계 검정을 하지 않았으므로 작은 차이는 우연일 수 있습니다.",
    "`selfval_bon`, `selfval_fta`는 캡션 방법이라 캡션 태스크(COCO, DOCCI)에만 적용됩니다.",
    "`logiccheck`는 MMStar, MMBench에서 답이 보기 글자뿐이라 사실상 base와 같게 나오는 것이 정상입니다.",
    "`logiccheck`의 MME 값은 원 논문에 없는 확장(짧은 Yes 답을 질문의 객체로 검증)에서 나온 것입니다. 질문에서 뽑힌 "
    "단어(예: python code의 python)를 객체로 보고 Yes를 No로 바꾸는 경우가 있어, 논문 방법 자체의 성능으로 읽으면 안 됩니다.",
    "`logiccheck`의 MM-Vet 값은 추론 답변을 객체 환각 교정 규칙으로 다시 쓴 결과입니다. 방법이 설계된 범위(이미지 설명) 밖의 적용입니다.",
    "MM-Vet은 로컬 judge로 채점해서 절대값은 다른 논문과 비교할 수 없습니다.",
    "공식 코드와 다른 점은 `BASELINES.md`에 있습니다. 캡션 길이 한도는 모든 모델이 128토큰입니다.",
]


def cell_text(r, m):
    if r["status"][m] == "na":
        return "해당 없음"
    if r["status"][m] == "missing":
        return "미실행"
    return fmt(r["values"][m])


def verdict_text(r):
    if not r["verdict"]:
        return ""
    return ", ".join(f"{m} {VERDICT_KO[v]}" for m, v in r["verdict"].items())


def to_markdown(s):
    lines = ["# self 우위 벤치마크: 체크포인트 vs 프롬프팅 베이스라인", ""]
    if s["limit"]:
        lines += [f"> **스모크 테스트 결과입니다. 태스크당 {s['limit']}개만 돌렸으니 숫자를 해석하지 마세요.**", ""]
    lines += [f"체크포인트 `{s['ckpt_dir']}`, 베이스라인 `{s['base_dir']}` (plan {s['created']}).",
              "굵은 글씨는 그 행에서 가장 좋은 값입니다. 화살표는 좋은 방향입니다.", ""]
    for w in s["warnings"]:
        lines += [f"> 경고: {w}"]
    if s["warnings"]:
        lines += [""]
    if s["not_run"]:
        lines += [f"> 미실행 칸: {', '.join(s['not_run'])}", ""]
    lines += ["## self와 베이스라인 비교 요약", "", "| 베이스라인 | self 이김 | 동점 | self 짐 |", "|---|---|---|---|"]
    for m, t in s["tally"].items():
        lines.append(f"| {m} | {t['win']} | {t['tie']} | {t['loss']} |")
    lines.append("")
    for i, c in enumerate(s["cats"], 1):
        lines += [f"## {i}. {c['title']}", "", "| 벤치마크 | " + " | ".join(COLS) + " | self 기준 |",
                  "|---|" + "---|" * (len(COLS) + 1)]
        for r in c["rows"]:
            cells = []
            for m in COLS:
                t = cell_text(r, m)
                if m in r["values"] and r["best"] is not None and abs(r["values"][m] - r["best"]) <= EPS:
                    t = f"**{t}**"
                cells.append(t)
            lines.append(f"| {r['label']} {'↑' if r['higher'] else '↓'} | " + " | ".join(cells) + f" | {verdict_text(r)} |")
        lines.append("")
    lines += ["## 캡션 평균 단어 수", "", "| 태스크 | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for label, vals in s["words"]:
        lines.append(f"| {label} | " + " | ".join(f"{vals[m]:.1f}" if m in vals else "" for m in COLS) + " |")
    lines += ["", "## 읽을 때 주의할 점", ""] + [f"- {n}" for n in NOTES]
    if s["skipped"]:
        lines.append(f"- 이 분류에 없는 {', '.join(s['skipped'])}는 베이스라인을 돌리지 않았습니다.")
    return "\n".join(lines) + "\n"


CSS = """
:root{color-scheme:light;--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--head:#f6f8fa;--best:#dafbe1;--warn:#fff8c5}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--head:#151b23;--best:#12361f;--warn:#3a2f05}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;margin:0 auto;max-width:1180px;padding:24px 16px}
h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:32px 0 10px}
.meta,.muted{color:var(--muted)}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right;white-space:nowrap}
th{background:var(--head)}td:first-child,th:first-child,td.note{text-align:left}td.note{white-space:normal;min-width:190px}
td.best{background:var(--best);font-weight:600}td.empty{color:var(--muted)}
.warn{background:var(--warn);border:1px solid var(--line);border-radius:6px;margin:10px 0;padding:8px 12px}
"""


def table(headers, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def to_html(s):
    e = html.escape
    parts = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>self 우위 벤치마크 비교</title><style>{CSS}</style></head><body>",
             "<h1>self 우위 벤치마크: 체크포인트 vs 프롬프팅 베이스라인</h1>"]
    if s["limit"]:
        parts.append(f"<div class='warn'><b>스모크 테스트 결과입니다.</b> 태스크당 {s['limit']}개만 돌렸으니 숫자를 해석하지 마세요.</div>")
    parts.append(f"<p class='meta'>체크포인트 <code>{e(s['ckpt_dir'])}</code>, 베이스라인 <code>{e(s['base_dir'])}</code> "
                 f"(plan {e(str(s['created']))}). 초록 칸은 그 행에서 가장 좋은 값입니다. 화살표는 좋은 방향입니다.</p>")
    for w in s["warnings"]:
        parts.append(f"<div class='warn'>경고: {e(w)}</div>")
    if s["not_run"]:
        parts.append(f"<div class='warn'>미실행 칸: {e(', '.join(s['not_run']))}</div>")
    parts.append("<h2>self와 베이스라인 비교 요약</h2>")
    parts.append(table(["베이스라인", "self 이김", "동점", "self 짐"],
                       [f"<tr><td>{e(m)}</td><td>{t['win']}</td><td>{t['tie']}</td><td>{t['loss']}</td></tr>"
                        for m, t in s["tally"].items()]))
    for i, c in enumerate(s["cats"], 1):
        parts.append(f"<h2>{i}. {e(c['title'])}</h2>")
        rows = []
        for r in c["rows"]:
            tds = [f"<td>{e(r['label'])} {'↑' if r['higher'] else '↓'}</td>"]
            for m in COLS:
                cls = ""
                if m in r["values"]:
                    if r["best"] is not None and abs(r["values"][m] - r["best"]) <= EPS:
                        cls = " class='best'"
                else:
                    cls = " class='empty'"
                tds.append(f"<td{cls}>{e(cell_text(r, m))}</td>")
            tds.append(f"<td class='note'>{e(verdict_text(r))}</td>")
            rows.append(f"<tr>{''.join(tds)}</tr>")
        parts.append(table(["벤치마크", *COLS, "self 기준"], rows))
    parts.append("<h2>캡션 평균 단어 수</h2>")
    parts.append(table(["태스크", *COLS],
                       ["<tr><td>" + e(label) + "</td>" + "".join(f"<td>{vals[m]:.1f}</td>" if m in vals else "<td></td>" for m in COLS)
                        + "</tr>" for label, vals in s["words"]]))
    notes = [e(n).replace("`", "") for n in NOTES]
    if s["skipped"]:
        notes.append(e(f"이 분류에 없는 {', '.join(s['skipped'])}는 베이스라인을 돌리지 않았습니다."))
    parts.append("<h2>읽을 때 주의할 점</h2><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default="lmms_eval_out")
    ap.add_argument("--baselines", default="lmms_eval_baselines")
    ap.add_argument("--out_stem", default=None, help="output path without extension (default <baselines>/prompting_comparison)")
    a = ap.parse_args()
    s = collect(a.checkpoints, a.baselines)
    stem = a.out_stem or os.path.join(a.baselines, "prompting_comparison")
    md = to_markdown(s)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(stem + ".html", "w", encoding="utf-8") as f:
        f.write(to_html(s))
    print(md)
    print(f"wrote {stem}.md and {stem}.html")
    if s["not_run"]:
        print(f"[report] not run yet: {', '.join(s['not_run'])}")


if __name__ == "__main__":
    main()
