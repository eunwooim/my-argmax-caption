"""Offline test of the reward comparison pipeline on CPU: no downloads, no GPU, minutes.

  1. tasks       prepare_tts_tasks.py --mock: CHAIR-500 on synthetic images (its official chair.pkl scores them), the
                 CaptionQA copy written (not run: its data is on the Hub)
  2. sampling    lmms_eval_run.py --prompting base-sample=tts_sample on mock_caption + chair500 with a tiny random
                 Qwen2.5-VL: every item has N candidates in prompting_trace.jsonl, its image bytes saved, and the cell's
                 answer is candidate 0; the same seed gives the same candidates again (resume)
  3. scoring     tts.py score with every selector (S2 raw, self-certainty, LPFV, Long-CLIP / jina as tiny random
                 CLIPs); self-certainty == a step-by-step computation from full forwards
  4. selection   tts.py select: each pick is the first maximum (shortest: minimum) of the first N candidates
  5. replay      tts.py args -> lmms_eval_run.py replay cells: the scored captions are exactly the picks
  6. agreement   tts.py vqa on a CaptionQA-like item, and tts_report.py's agreement on a CaptionQA-like sample file
  7. report      tts_report.py over the cells: the CHAIR-500 numbers equal the task's own results

  python tts_smoke.py --out_dir tts_smoke      # must end with [smoke] OK
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lmms_eval_smoke import CAPTION_YAML, UTILS, run  # noqa: E402

SELFVAL_DIR = os.environ.get("SELFVAL_DIR", "")


def first(x):
    while isinstance(x, list):
        x = x[0]
    return x


def check_self_certainty(model_dir, rec):
    """tts.self_certainty (batched, right-padded, logits_to_keep) == per-token full forwards."""
    import torch
    import tts as T
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(model_dir, min_pixels=3136, max_pixels=50176)
    vlm = T.PB.VLM(T.LocalLM(model, proc, 3136, 50176))
    caps = rec["candidates"][:3]
    got = T.self_certainty(vlm, rec["images"], rec["prompt"], caps, batch=2)
    base = vlm.inputs([vlm.messages(rec["images"], rec["prompt"])])
    end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for c, g in zip(caps, got):
        ids, tt, vals = base["input_ids"], base["mm_token_type_ids"], []
        for t in proc.tokenizer(c, add_special_tokens=False).input_ids + [end]:
            with torch.inference_mode():
                lg = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=base["pixel_values"],
                           image_grid_thw=base["image_grid_thw"], mm_token_type_ids=tt).logits[0, -1].float()
            p = torch.log_softmax(lg, -1)
            vals.append(float(-math.log(lg.shape[-1]) - p.mean()))
            ids = torch.cat([ids, torch.tensor([[t]])], 1)
            tt = torch.cat([tt, torch.zeros((1, 1), dtype=tt.dtype)], 1)
        ref = sum(vals) / len(vals)
        assert abs(ref - g) <= 1e-3 * abs(ref) + 1e-4, f"self-certainty {c!r}: batched {g} != stepwise {ref}"
    print(f"[smoke] self-certainty batched == stepwise for {len(caps)} captions ({[round(x, 4) for x in got]}): ok")


def check_chair_official(tasks, selfval_dir):
    """The chair500 task's counts == Self-Validation's chair.py run as its README says, on captions with objects."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("_smoke_chair500", os.path.join(tasks, "chair500_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(os.path.join(tasks, "chair500.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    caps = ["A man is riding a bicycle next to a car on the street.", "A dog and a cat sit on a sofa near a television.",
            "A plate of pizza with a fork and a knife on a dining table.", "A person holds an umbrella next to a bus."]
    rows, caps = rows[:len(caps)], caps[:len(rows)]
    ours = mod.overall(mod.per_caption([r["image"] for r in rows], caps))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r, c in zip(rows, caps):
            f.write(json.dumps(dict(image=r["image"], text=c)) + "\n")
    d = os.path.join(selfval_dir, "chair500")
    res = subprocess.run([sys.executable, os.path.join(d, "chair.py"), "--cap_file", f.name, "--image_id_key", "image",
                          "--caption_key", "text", "--cache", os.path.join(d, "chair.pkl")], capture_output=True,
                         text=True, cwd=d, check=True)
    off = {l.split(":")[0].strip(): float(l.split(":")[1]) for l in res.stdout.splitlines()
           if ":" in l and "cache" not in l}
    for k, ok in (("CHAIRs", "CHAIRs"), ("CHAIRi", "CHAIRi"), ("Recall", "Recall"), ("Precision", "Precision"), ("F1", "f1")):
        assert abs(ours[k] - off[ok]) < 0.051, (k, ours[k], off[ok])  # chair.py prints one decimal
    print(f"[smoke] chair500 counts == Self-Validation's chair.py ({ {k: round(v, 1) for k, v in ours.items()} }): ok")


def check_agreement(out, tasks):
    """tts_report.cell_items: CaptionQA agreement = chosen option from the caption == the model's own image answer."""
    import tts_report as TR
    cell = os.path.join(out, "fake_cqa", "caption_long")
    os.makedirs(os.path.join(cell, "x"), exist_ok=True)
    open(os.path.join(cell, "done.json"), "w").write("{}")
    qs = [dict(image_id="i0", question_idx=0, score=1.0, chosen=1), dict(image_id="i0", question_idx=1, score=0.0, chosen=2),
          dict(image_id="i0", question_idx=2, score=0.3, chosen=3)]
    with open(os.path.join(cell, "x", "1_samples_captionqa_x_natural.jsonl"), "w") as f:
        f.write(json.dumps(dict(doc_id=0, filtered_resps=["a b c"], captionqa_score=qs)) + "\n")
    vqa = {("i0", 0): 1, ("i0", 1): 0, ("i0", 2): None}  # agree, disagree, no image answer (left out)
    items = TR.cell_items(cell, vqa, TR.ChairScorer(tasks))
    st = TR.stats("captionqa", items["captionqa"][1])
    assert abs(st["cqa_agree"] - 50.0) < 1e-9 and abs(st["cqa_score"] - 1.3 / 3) < 1e-9, st
    print(f"[smoke] CaptionQA agreement / score from the per-question log: ok ({st})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="tts_smoke")
    ap.add_argument("--n", type=int, default=3, help="items per synthetic task")
    ap.add_argument("--selfval_dir", default=SELFVAL_DIR, help="a SelfVal clone (its chair500/ files), offline")
    a = ap.parse_args()
    if not a.selfval_dir:
        raise SystemExit("[smoke] --selfval_dir <SelfVal clone> (or SELFVAL_DIR): CHAIR-500's files, offline")
    out = os.path.abspath(a.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    py = sys.executable
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--dtype", "bfloat16", "--out", f"{out}/base"], env=env)
    base = f"{out}/base"

    # 1. tasks: mock_caption + CHAIR-500 on synthetic images
    import mock_data as M
    tasks = os.path.join(out, "tasks")
    run([py, os.path.join(HERE, "prepare_tts_tasks.py"), "--out_dir", tasks, "--mock", "--mock_n", str(a.n),
         "--selfval_dir", a.selfval_dir, "--only", "captionqa,chair500"], env=env)
    with open(os.path.join(tasks, "caption.jsonl"), "w") as g:
        for i in range(5000, 5000 + a.n):
            s, c, s2, c2 = M._scene(i)
            g.write(json.dumps(dict(id=i, file=f"synth_{i}.png", caption=f"A {c} {s} and a {c2} {s2}.")) + "\n")
    with open(os.path.join(tasks, "mock_caption.yaml"), "w") as f:
        f.write(CAPTION_YAML.format(data=os.path.join(tasks, "caption.jsonl")))
    with open(os.path.join(tasks, "utils.py"), "w") as f:
        f.write(UTILS.format(here=HERE))
    check_chair_official(tasks, a.selfval_dir)
    ref = f"{out}/ref"
    os.makedirs(ref)
    with open(f"{ref}/plan.json", "w") as f:
        json.dump(dict(models={}, groups={"caption_long": dict(tasks=["mock_caption", "chair500"],
                                                               gen_kwargs="max_new_tokens=16,temperature=0")},
                       model_args="min_pixels=3136,max_pixels=50176,device=cpu,device_map=cpu", seed=0, limit=None,
                       batch_size=2, include_path=tasks), f)
    common = ["--settings_from", f"{ref}/plan.json", "--settings_groups", "caption_long", "--gpus", "cpu"]

    # 2. sampling cell
    lm = f"{out}/lmms_tts"
    run([py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", lm, "--models", f"base-sample={base}",
         "--prompting", "base-sample=tts_sample", "--prompting_args", json.dumps({"tts_sample": {"n": 4}})] + common,
        env=env)
    cell = f"{lm}/base-sample/caption_long"
    with open(f"{cell}/prompting_trace.jsonl") as f:
        tr = [json.loads(l) for l in f]
    assert len(tr) == 2 * a.n and all(len(t["candidates"]) == 4 for t in tr), [len(t["candidates"]) for t in tr]
    assert all(os.path.exists(p) for t in tr for p in t["images"]), "images not saved"
    for task in ("mock_caption", "chair500"):
        p = glob.glob(f"{cell}/**/*_samples_{task}.jsonl", recursive=True)[0]
        with open(p) as f:
            resp = {r["doc_id"]: first(r["filtered_resps"]) for r in map(json.loads, f)}
        assert resp == {t["doc_id"]: t["candidates"][0] for t in tr if t["task"] == task}, f"{task}: answer != candidate 0"
    assert len({c for t in tr for c in t["candidates"]}) > len(tr), "candidates of an item are identical (no sampling?)"
    print(f"[smoke] sampling cell: {len(tr)} items x 4 candidates, images saved, answers = candidate 0: ok")
    # tts_vllm.py dump (the vLLM path's items, built on the CPU from lmms-eval's task code) == what the cell saw
    items_dir = f"{out}/items"
    run([py, os.path.join(HERE, "tts_vllm.py"), "dump", "--tasks", "mock_caption,chair500", "--include_path", tasks,
         "--out_dir", items_dir, "--max_new_tokens", "16"], env=env)
    with open(f"{items_dir}/items.jsonl") as f:
        dumped = {(r["task"], r["doc_id"]): r for r in map(json.loads, f)}
    assert set(dumped) == {(t["task"], t["doc_id"]) for t in tr}, "dump has other items than the sampling cell"
    for t in tr:
        d = dumped[(t["task"], t["doc_id"])]
        assert d["prompt"] == t["prompt"], (d["prompt"], t["prompt"])
        assert [open(p, "rb").read() for p in d["images"]] == [open(p, "rb").read() for p in t["images"]], \
            f"{t['task']}/{t['doc_id']}: image bytes differ"
        assert {k: v for k, v in d["doc"].items() if k in t["doc"]} == t["doc"], "item fields differ"
    print(f"[smoke] tts_vllm.py dump == the sampling cell's items (prompt, JPEG bytes, fields) for {len(tr)} items: ok")
    # same seed per item -> the same candidates when sampled again (a resumed cell)
    import torch
    import prompting_baselines as PB
    import tts as T
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16).eval()
    proc = AutoProcessor.from_pretrained(base, min_pixels=3136, max_pixels=50176)

    class Req:
        def __init__(self, t):
            from PIL import Image
            self.doc = {"x": 1}
            self.args = (t["prompt"], {"max_new_tokens": t["max_new_tokens"]},
                         lambda d: [Image.open(p).convert("RGB") for p in t["images"]], t["doc_id"], t["task"], t["split"])
    import lmms_eval.models.simple.qwen2_5_vl as Q
    lmo = T.LocalLM(model, proc, 3136, 50176)
    lmo._encode_image_data_url = lambda im: Q.encode_image_to_data_url(im, image_format="JPEG", mime_type="image/jpeg",
                                                                      convert_rgb=True, quality=85)
    lmo.task_dict = {}
    for t in tr[:2]:
        lmo.task_dict.setdefault(t["task"], {}).setdefault(t["split"], {})[t["doc_id"]] = {}
    s4 = PB.settings("tts_sample", {"n": 4})
    r1 = PB.tts_sample(lmo, PB.VLM(lmo), Req(tr[0]), s4, 0, f"{out}/resample_images")
    PB.tts_sample(lmo, PB.VLM(lmo), Req(tr[1]), s4, 0, f"{out}/resample_images")  # another item in between
    r2 = PB.tts_sample(lmo, PB.VLM(lmo), Req(tr[0]), s4, 0, f"{out}/resample_images")
    assert r1["candidates"] == r2["candidates"], "the same item sampled twice gave other candidates (seed per item)"
    print("[smoke] the seed is per item: an item sampled again (a resumed cell) gets the same candidates: ok")

    # 3. scoring
    td = f"{out}/tts"
    run([py, os.path.join(HERE, "tts.py"), "score", "--sample_dir", cell, "--out", f"{td}/scores.jsonl", "--model", base,
         "--plan", f"{ref}/plan.json", "--reward_raw", "--mock_scorers", "--device", "cpu", "--dtype", "float32",
         "--step1_dir", ""], env=env)
    with open(f"{td}/scores.jsonl") as f:
        sc = [json.loads(l) for l in f]
    for s in T.SCORED:
        assert all(len(r[s]) == 4 and all(math.isfinite(v) for v in r[s]) for r in sc), s
    print(f"[smoke] every candidate scored by {T.SCORED}: ok")
    check_self_certainty(base, tr[0])

    # 4. selection
    run([py, os.path.join(HERE, "tts.py"), "select", "--sample_dir", cell, "--scores", f"{td}/scores.jsonl",
         "--out_dir", f"{td}/selected", "--ns", "2,4"], env=env)
    by = {PB.item_key(r["task"], r["doc_id"]): r for r in sc}
    cand = {PB.item_key(t["task"], t["doc_id"]): t["candidates"] for t in tr}
    for s in T.SELECTORS:
        for n in (2, 4):
            with open(f"{td}/selected/{s}_n{n}.jsonl") as f:
                for x in map(json.loads, f):
                    k = PB.item_key(x["task"], x["doc_id"])
                    assert x["index"] < n and x["caption"] == cand[k][x["index"]]
                    if s == "shortest":
                        assert by[k]["words"][x["index"]] == min(by[k]["words"][:n])
                    elif s in T.SCORED:
                        assert by[k][s][x["index"]] == max(by[k][s][:n]) and by[k][s].index(max(by[k][s][:n])) == x["index"]
    print("[smoke] selection: first maximum (shortest: minimum) of the first N candidates: ok")

    # 5. replay cells
    def targs(field):
        return subprocess.run([py, os.path.join(HERE, "tts.py"), "args", "--selected_dir", f"{td}/selected", "--prefix",
                               "base", "--model", base, "--selectors", "self,shortest", "--field", field],
                              check=True, capture_output=True, text=True, env=env).stdout.strip()
    run([py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", lm, "--models", targs("models"),
         "--prompting", targs("prompting"), "--prompting_args", targs("prompting_args")] + common, env=env)
    for s in ("self", "shortest"):
        for n in (2, 4):
            with open(f"{td}/selected/{s}_n{n}.jsonl") as f:
                want = {(x["task"], x["doc_id"]): x["caption"] for x in map(json.loads, f)}
            c = f"{lm}/base-{s}-n{n}/caption_long"
            for task in ("mock_caption", "chair500"):
                p = glob.glob(f"{c}/**/*_samples_{task}.jsonl", recursive=True)[0]
                with open(p) as f:
                    got = {(task, r["doc_id"]): first(r["filtered_resps"]) for r in map(json.loads, f)}
                assert got == {k: v for k, v in want.items() if k[0] == task}, f"{s} n{n} {task}: scored != picked"
    print("[smoke] replay cells score exactly the picked captions: ok")

    # 6. agreement: the model's own answers (vqa) on a CaptionQA-like item, and the report's agreement
    sys.path.insert(0, tasks)
    import captionqa_ext  # noqa: F401  (imports lmms-eval's CaptionQA utils: the real helpers)
    vdir = f"{out}/vqa_cell"
    os.makedirs(vdir, exist_ok=True)
    doc = dict(id="cq0", shuffle_perms="{}", questions=[
        dict(question="What color is the shape?", choices=["red", "blue", "green"], answer="red", category=["c"]),
        dict(question="Is there a circle?", choices=["Yes", "No"], answer="Yes", category=["c"])])
    with open(f"{vdir}/prompting_trace.jsonl", "w") as f:
        f.write(json.dumps(dict(task="captionqa_x_natural", doc_id=0, split="natural", prompt="Describe this image in detail.",
                                max_new_tokens=16, images=tr[0]["images"], candidates=["x"], doc=doc)) + "\n")
    run([py, os.path.join(HERE, "tts.py"), "vqa", "--sample_dir", vdir, "--out", f"{td}/vqa/base.jsonl", "--model", base,
         "--plan", f"{ref}/plan.json", "--tasks_dir", tasks, "--device", "cpu"], env=env)
    with open(f"{td}/vqa/base.jsonl") as f:
        v = [json.loads(l) for l in f]
    assert len(v) == 2 and all(x["chosen"] is None or 0 <= x["chosen"] < 3 for x in v), v
    print(f"[smoke] own image answers for {len(v)} CaptionQA questions: ok ({[x['answer'][:12] for x in v]})")
    check_agreement(out, tasks)

    # 7. report: a Use 2 cell (base, greedy) + the test-time cells; CHAIR-500 numbers = the task's own results
    ck = f"{out}/lmms_ckpt"
    run([py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", ck, "--models", f"base={base}"] + common, env=env)
    run([py, os.path.join(HERE, "tts_report.py"), "--tts", lm, "--checkpoints", ck, "--vqa_dir", f"{td}/vqa",
         "--tasks_dir", tasks, "--bootstrap", "50", "--out_stem", f"{td}/reward_comparison"], env=env)
    import tts_report as TR
    items = TR.cell_items(f"{ck}/base/caption_long", None, TR.ChairScorer(tasks))
    st = TR.stats("chair500", items["chair500"][1])
    p = glob.glob(f"{ck}/base/caption_long/**/*_results.json", recursive=True)[0]
    with open(p) as f:
        r = json.load(f)["results"]["chair500"]
    for m, k in (("chair_i", "chair500_CHAIRi,none"), ("chair_s", "chair500_CHAIRs,none"), ("chair_f1", "chair500_F1,none")):
        if not (math.isnan(st[m]) and math.isnan(r[k])):
            assert abs(st[m] - r[k]) < 1e-6, (m, st[m], r[k])
    assert os.path.getsize(f"{td}/reward_comparison.md") > 0 and os.path.getsize(f"{td}/reward_comparison.html") > 0
    print(f"[smoke] report written; its CHAIR-500 recomputation == the task's results ({ {k: round(v, 2) for k, v in st.items()} }): ok")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
