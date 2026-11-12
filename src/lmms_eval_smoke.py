"""Offline end-to-end test of the lmms-eval path, on CPU in a few minutes: no downloads, no GPU.

  1. grpo.py --mock                     tiny random Qwen2.5-VL, 3 GRPO steps (self reward)
  2. export_ckpt.py --mock              base as an HF folder; step_2 exported and --check_only (the checks)
  3. two synthetic lmms-eval tasks      mock_yesno (short answer) and mock_caption (open-ended), on the
                                        same synthetic images as --mock, written to <out_dir>/tasks
  4. lmms_eval_run.py                   both models x the three groups (mcq, caption, general) through the real
                                        lmms-eval qwen2_5_vl; the GRPO model is the step_2 adapter itself, merged in
                                        memory by lmms_eval_lora.py; it must give the same outputs as the exported folder
  5. prepare_tasks.py --mock            the DOCCI caption task on synthetic images (CAPTURE replaced by a word
                                        overlap, DOCCI_MOCK_METRICS=1; CIDEr / BLEU / METEOR / ROUGE for real);
                                        the general group is dropped (--drop_groups) and added again with it:
                                        the old cells go to _old_*/, mock_caption replays from the copied cache
  6. --rescore                          one cell scored again from its cache
  7. lmms_eval_report.py                the table; also checks every cell is there

The scores mean nothing (random model). What it shows: checkpoints load in lmms-eval, the plan is frozen,
groups can be dropped and added without losing results, and the report finds every result. Run it once on the
server after installing lmms-eval, before the real (GPU-hours) evaluation.

  python lmms_eval_smoke.py --out_dir lmms_smoke
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

YESNO_YAML = """task: mock_yesno
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.yesno_text
doc_to_target: answer
generation_kwargs:
  max_new_tokens: 8
  temperature: 0
process_results: !function utils.yesno_results
metric_list:
  - metric: mock_accuracy
    aggregation: mean
    higher_is_better: true
"""

CAPTION_YAML = """task: mock_caption
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: generate_until
doc_to_visual: !function utils.doc_to_visual
doc_to_text: !function utils.caption_text
doc_to_target: caption
generation_kwargs:
  max_new_tokens: 16
  temperature: 0.7
  until: []
process_results: !function utils.caption_results
metric_list:
  - metric: mock_words
    aggregation: mean
    higher_is_better: true
"""

UTILS = '''import sys
sys.path.insert(0, {here!r})
import mock_data as M


def doc_to_visual(doc):
    return [M.image(doc["file"])]


def yesno_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\\nAnswer the question using a single word or phrase."


def yesno_results(doc, results):
    return {{"mock_accuracy": float(results[0].strip().lower().startswith(doc["answer"]))}}


def caption_text(doc, lmms_eval_specific_kwargs=None):
    return "Describe this image."


def caption_results(doc, results):
    return {{"mock_words": float(len(results[0].split()))}}
'''


def run(cmd, **kw):
    print("[smoke] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="lmms_smoke")
    ap.add_argument("--n", type=int, default=4, help="items per synthetic task")
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    py = sys.executable
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", DOCCI_MOCK_METRICS="1")

    run([py, os.path.join(HERE, "grpo.py"), "--mock", "--out_dir", f"{out}/grpo"], env=env)
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--out", f"{out}/exports/base"], env=env)
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--adapter", f"{out}/grpo/ckpt/step_2",
         "--out", f"{out}/exports/self_step2"], env=env)
    # bf16, as on the server (lmms-eval loads in bf16): base folder for the in-memory merge, and the same
    # adapter exported from it, to compare the two ways item by item
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--dtype", "bfloat16", "--out", f"{out}/exports/base_bf16"],
        env=env)
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--dtype", "bfloat16", "--adapter",
         f"{out}/grpo/ckpt/step_2", "--out", f"{out}/exports/self_step2_bf16"], env=env)
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--check_only", "--adapter", f"{out}/grpo/ckpt/step_2",
         "--out", f"{out}/merge_checks/self_step2"], env=env)
    assert os.listdir(f"{out}/merge_checks/self_step2") == ["export.json"], "check_only wrote more than export.json"

    import mock_data as M
    tasks = os.path.join(out, "tasks")
    os.makedirs(tasks)
    with open(os.path.join(tasks, "yesno.jsonl"), "w") as f, open(os.path.join(tasks, "caption.jsonl"), "w") as g:
        for i in range(5000, 5000 + a.n):
            s, c, s2, c2 = M._scene(i)
            f.write(json.dumps(dict(id=i, file=f"synth_{i}.png", question=f"Is there a {c} {s} in the image?",
                                    answer="yes")) + "\n")
            g.write(json.dumps(dict(id=i, file=f"synth_{i}.png", caption=f"A {c} {s} and a {c2} {s2}.")) + "\n")
    with open(os.path.join(tasks, "mock_yesno.yaml"), "w") as f:
        f.write(YESNO_YAML.format(data=os.path.join(tasks, "yesno.jsonl")))
    with open(os.path.join(tasks, "mock_caption.yaml"), "w") as f:
        f.write(CAPTION_YAML.format(data=os.path.join(tasks, "caption.jsonl")))
    with open(os.path.join(tasks, "utils.py"), "w") as f:
        f.write(UTILS.format(here=HERE))
    run([py, os.path.join(HERE, "prepare_tasks.py"), "--mock", "--mock_n", str(a.n), "--out_dir", tasks], env=env)

    def run_cmd(mcq, caption, general, *extra):
        return [py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", f"{out}/lmms",
                "--models", f"base={out}/exports/base,self={out}/grpo/ckpt/step_2", "--lora_base", f"{out}/exports/base_bf16",
                "--mcq", mcq, "--caption", caption, "--general", general, "--include_path", tasks,
                "--model_args", "device=cpu,device_map=cpu", "--gpus", "cpu", "--hf_dir", f"{out}/hf"] + list(extra)
    cpu = dict(env, CUDA_VISIBLE_DEVICES="")
    run(run_cmd("mock_yesno", "mock_caption", "mock_caption"), env=cpu)   # plan frozen with mcq + caption + general
    for g in ("mcq", "caption", "general"):
        with open(f"{out}/lmms/self/{g}/run.log") as f:
            assert "[lora] merged" in f.read(), f"self/{g}: adapter was not merged in memory"

    # the frozen plan must refuse a changed task list, and a group left out without --drop_groups
    for args, want in ((("pope", "mock_caption", "mock_caption"), "frozen plan"),
                       (("mock_yesno", "mock_caption", ""), "--drop_groups general"),
                       (("mock_yesno", "mock_caption", "", "--drop_groups", "nope"), "not in the frozen plan")):
        r = subprocess.run(run_cmd(*args) + ["--dry_run"], capture_output=True, text=True, env=env)
        assert r.returncode != 0 and want in r.stdout + r.stderr, f"plan freeze did not stop {args}: {r.stdout[-500:]} {r.stderr[-500:]}"
    print("[smoke] frozen plan refused changed task lists and silent group removal: ok")

    # drop the general group and define it anew with the DOCCI caption task added: the old cells are kept aside,
    # mock_caption's responses replay from the copied cache, docci_caption is generated and scored
    run(run_cmd("mock_yesno", "mock_caption", "mock_caption,docci_caption", "--drop_groups", "general"), env=cpu)
    old = glob.glob(f"{out}/lmms/_old_*")
    assert len(old) == 1 and sorted(os.listdir(old[0])) == ["base_general", "self_general"], old
    assert glob.glob(f"{old[0]}/base_general/**/*_results.json", recursive=True), "dropped cell lost its results"
    with open(f"{out}/lmms/plan.json") as f:
        plan = json.load(f)
    assert list(plan["groups"]) == ["mcq", "caption", "general"] and "added" in plan["groups"]["general"], plan["groups"]
    assert plan["groups"]["general"]["tasks"] == ["mock_caption", "docci_caption"]
    with open(f"{out}/lmms/plan_history.jsonl") as f:
        hist = [json.loads(l) for l in f]
    assert len(hist) == 1 and hist[0]["dropped"] == ["general"], hist
    with open(f"{out}/lmms/base/general/run.log") as f:
        hits = [int(m.group(1)) for m in re.finditer(r"(\d+)/(\d+) cache hits", f.read())]
    assert hits and max(hits) > 0, f"mock_caption did not replay from the copied cache (hits {hits})"
    print(f"[smoke] group dropped and re-added: old cells kept, cache replayed ({max(hits)} hits), docci_caption scored: ok")

    # --rescore: the cell's scores are set aside and it is scored again from its cache (no generation)
    run(run_cmd("mock_yesno", "mock_caption", "mock_caption,docci_caption", "--rescore", "self/general"), env=cpu)
    assert glob.glob(f"{out}/lmms/_rescored_*/self_general/*_results.json"), "rescore did not set the old scores aside"
    assert glob.glob(f"{out}/lmms/self/general/**/*_results.json", recursive=True), "rescore left no new results"
    print("[smoke] --rescore scored a cell again from its cache: ok")

    run([py, os.path.join(HERE, "lmms_eval_report.py"), "--out_dir", f"{out}/lmms"])
    left = [d for d in os.listdir(f"{out}/hf") if d.startswith("datasets_")] if os.path.isdir(f"{out}/hf") else []
    assert not left, f"benchmark data of finished groups not deleted: {left}"
    print("[smoke] per-group benchmark data deleted after use: ok")

    # the in-memory merge must evaluate exactly like the exported folder: same outputs on every item
    exp_out = f"{out}/lmms_exported"
    run([py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", exp_out, "--models", f"self={out}/exports/self_step2_bf16",
         "--mcq", "mock_yesno", "--caption", "mock_caption", "--general", "", "--include_path", tasks,
         "--model_args", "device=cpu,device_map=cpu", "--gpus", "cpu"], env=cpu)
    for g, task in (("mcq", "mock_yesno"), ("caption", "mock_caption")):
        def outs(root):
            p = glob.glob(f"{root}/self/{g}/**/*_samples_{task}.jsonl", recursive=True)
            with open(p[0]) as f:
                return {r["doc_id"]: r["filtered_resps"] for r in map(json.loads, f)}
        mem, exp = outs(f"{out}/lmms"), outs(exp_out)
        assert mem == exp, f"{task}: in-memory merge {mem} != exported folder {exp}"
    print("[smoke] adapter merged in memory = exported folder, item by item: ok")

    with open(f"{out}/lmms/summary.json") as f:
        s = json.load(f)
    acc = s["groups"]["mcq"]["mock_yesno"]["mock_accuracy"]["values"]
    words = s["groups"]["caption"]["mock_caption"]["mock_words"]["values"]
    docci = {k: v["values"] for k, v in s["groups"]["general"]["docci_caption"].items()}
    assert set(acc) == set(words) == {"base", "self"}, (acc, words)
    for m in ("docci_CAPTURE", "docci_CIDEr", "docci_Bleu_4", "docci_METEOR", "docci_ROUGE_L"):
        assert set(docci.get(m, {})) == {"base", "self"}, (m, docci)
    print(f"[smoke] all cells present: accuracy {acc}, words {words}, DOCCI {docci}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
