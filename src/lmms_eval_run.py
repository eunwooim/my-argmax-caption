"""Run lmms-eval on fixed GRPO checkpoints: every model in the plan x every task, same settings for all.

Models are given as name=path. A path is a GRPO LoRA adapter folder (merged into its base model in memory
by lmms_eval_lora.py, nothing written to disk), an HF model folder (export_ckpt.py output) or a hub id:
  base     = Qwen/Qwen2.5-VL-7B-Instruct            (LoRA off, the before-RL model)
  self     = out_grpo/adapter                         (own hidden-state reward)
  external = out_grpo_external/adapter                (outside embedding reward)
  length   = out_grpo_length/adapter                  (length-matched control)
An adapter's base model is the one in its adapter_config.json (--lora_base when that is empty).

Three task groups, run as separate lmms-eval calls because they need different generation settings:
  mcq      multiple-choice / short answer (default POPE, MMStar): each task's own lmms-eval settings
  caption  captioning (default COCO CHAIR, DOCCI captioning = prepare_tasks.py's docci_caption): greedy, up to
           --caption_max_new_tokens (the training limit) for every task, so long captions are not cut by a
           task default
  general  general ability (default MMVP, MMBench, MME, MM-Vet): each task's own settings, but greedy
  caption_long  the reward comparison's caption benchmarks (--caption_long, off by default; CaptionQA, AMBER,
           CHAIR-500 from prepare_tts_tasks.py): greedy, up to --caption_long_max_new_tokens (4096, no cap in
           practice) for every model, also those trained with a 128-token limit
--run_groups runs some groups now (the plan keeps all). --prompting <name>=tts_sample / replay are the test-time
scaling cells (tts.py): N sampled candidates per item, or a caption chosen offline replayed into the task's scorer.

The plan (models, tasks, generation and pixel settings, seed, limit) is written to <out_dir>/plan.json
on the first run and frozen: a later run with other tasks or settings stops, so the benchmark list
cannot drift after results are seen. Adding a new model name or a new task group is allowed. Removing a
group needs --drop_groups a,b: its results move to <out_dir>/_old_<stamp>/ (kept, out of the table), and
when a new group added in the same run has tasks the dropped group had, with the same generation settings,
the dropped cells' response caches are copied over, so those tasks are scored again without generating
again. --replan replaces the whole plan (old plan in plan_history.jsonl, all old results moved aside).

Disk (a shared server): with --hf_dir nothing is written into the HF_HOME cache (shared, read only; a model
already there is used from there). Every download goes under --hf_dir: models/ (kept) and datasets_<group>/
(benchmark data + arrow + xet cache). Groups run one after another for every model, and a group's data folder
is deleted once the group is done for every model, so one group's data (<= ~10 GB) is on disk at a time.
--disk prints what is on disk; --keep_datasets keeps the data. The first failed cell stops the run.

Several workers may run the same plan at once (one per free GPU: rl_paper_commands.sh par N): a worker marks
the cell it is on (<cell>/running.json) and the others skip it, so the cells are shared, not repeated.

Every response is cached as it is generated (<out_dir>/<model>/<group>/cache, lmms-eval's --use_cache), so a
cell that failed in scoring, or was killed, is finished on the next run without generating again
(--rerun deletes the cache). Scorers are best checked before the real run with a short one:
  python lmms_eval_run.py --out_dir lmms_quick --limit 4 --only base ...   (same tasks, 4 items, minutes)

Images: min/max pixels are taken from the checkpoints' training config (the training values, 56*56 and
1024*28*28 by default) and passed to lmms-eval, which otherwise uses its own (256*28*28 .. 1605632).
lmms-eval's --force_simple path is used: it applies these limits to every image (its chat path does not).

Speed: --batch_size 8 (default) generates 8 prompts at once (left padding, greedy) and --attn auto uses
flash attention when flash-attn is installed. Both are frozen with the plan, so every cell of a table uses the
same kernel and batch (bf16 outputs differ slightly between kernels and batch sizes).

  python lmms_eval_run.py --out_dir lmms_out --models base=Qwen/Qwen2.5-VL-7B-Instruct,self=out_grpo/adapter \\
      --judge_url http://localhost:8047/v1 --include_path custom_tasks --gpus 4
  python lmms_eval_run.py --out_dir lmms_out --dry_run           # print the commands of the frozen plan
  python lmms_eval_report.py --out_dir lmms_out                  # the model x benchmark table
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # GPU numbers == nvidia-smi numbers

import argparse
import glob
import json
import shlex
import shutil
import socket
import subprocess
import sys
import time

import gpu

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MCQ = "pope,mmstar"
DEFAULT_CAPTION = "coco_cap_chair,docci_caption"  # docci_caption comes from prepare_tasks.py (--include_path)
DEFAULT_GENERAL = "mmvp,mmbench_en_dev_static,mme,mmvet"  # ship with lmms-eval; MM-Vet is scored by the judge LLM
# the reward comparison's caption benchmarks (README "Reward comparison"): no length cap (--caption_long_max_new_tokens)
DEFAULT_CAPTION_LONG = ("captionqa_x_natural,captionqa_x_document,captionqa_x_ecommerce,captionqa_x_embodiedai,"
                        "amber_g,chair500")
CUSTOM_TASKS = ("docci_caption", "captionqa_x", "chair500")  # need prepare_tasks.py / prepare_tts_tasks.py's folder
JUDGED_TASKS = ("mmvet", "captionqa")  # scored by the judge LLM (captionqa_x too: same prefix)
# python modules each task's scorer imports (lmms-eval does not install them)
# (task name prefix -> modules)
TASK_DEPS = {"docci_caption": ["capture_metric", "pycocoevalcap"], "coco_cap_chair": ["nltk"],
             "detailcaps": ["capture_metric", "pycocoevalcap", "pycocotools"], "coco20": ["pycocoevalcap"],
             "coco_karpathy": ["pycocoevalcap"], "nocaps": ["pycocoevalcap"],
             "mmbench": ["openpyxl"],  # mmbench writes its submission .xlsx
             "amber_g": ["spacy", "nltk"], "chair500": ["nltk"], "captionqa": ["requests"]}
CAPTURE_TASKS = ("docci_caption", "detailcaps")  # CAPTURE scorer (capture_metric, GPU) + pycocoevalcap (Java)
PROMPTING_METHODS = ("selfval_bon", "selfval_fta", "logiccheck", "tts_sample", "replay")  # prompting_baselines.py
GROUP_ORDER = ("mcq", "caption", "general", "caption_long")
FROZEN = ("groups", "model_args", "seed", "limit", "batch_size", "include_path", "force_simple")


def parse_models(spec):
    models = {}
    for part in [p for p in spec.split(",") if p.strip()]:
        if "=" not in part:
            raise SystemExit(f"[plan] --models entry {part!r} is not name=path")
        name, path = (x.strip() for x in part.split("=", 1))
        if not name.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(f"[plan] model name {name!r}: letters, digits, - and _ only")
        if name in models:
            raise SystemExit(f"[plan] model name {name!r} given twice")
        if not os.path.exists(path) and (path.count("/") != 1 or path.startswith((".", "/", "~"))):
            raise SystemExit(f"[plan] model {name}: {path} does not exist (and is not a hub id like org/name)")
        models[name] = os.path.abspath(path) if os.path.exists(path) else path
    return models


def parse_prompting(a, models):
    """--prompting name=method,...: those models run with a prompting baseline (prompting_baselines.py) wrapped
    around lmms-eval's generation; its settings (the official defaults, or --prompting_args) go into the plan."""
    out = {}
    extra = json.loads(a.prompting_args) if a.prompting_args else {}
    for part in [p for p in (a.prompting or "").split(",") if p.strip()]:
        name, method = (x.strip() for x in part.split("=", 1))
        if name not in models:
            raise SystemExit(f"[plan] --prompting {name}: not a model in --models")
        if method not in PROMPTING_METHODS:
            raise SystemExit(f"[plan] --prompting {name}={method}: method must be one of {PROMPTING_METHODS}")
        sys.path.insert(0, HERE)
        import prompting_baselines as PB
        if is_adapter(models[name]):
            print(f"[warn] {name}: {method} on a LoRA checkpoint (the baselines are meant for the base model)", flush=True)
        ap = PB.APPLIES[method]  # a tuple; stored as a list so it equals the copy read back from plan.json
        # --prompting_args: {method: {...}} for every model of the method, {name: {...}} for one model (replay files)
        out[name] = dict(method=method, settings=PB.settings(method, extra.get(name, extra.get(method))),
                         applies=None if ap is None else list(ap))
        if method == "replay" and not out[name]["settings"]["file"]:
            raise SystemExit(f"[plan] --prompting {name}=replay needs --prompting_args '{{\"{name}\": {{\"file\": ...}}}}'")
    return out


def prompting_applies(plan, name, group):
    """False when the model's prompting method changes none of the group's tasks (the cell would be the base
    model's answers again: skipped, shown as empty in the table)."""
    p = (plan.get("prompting") or {}).get(name)
    if not p or p["applies"] is None:
        return True
    return any(t.startswith(tuple(p["applies"])) for t in plan["groups"][group]["tasks"])


def settings_from(a):
    """--settings_from <plan.json>: take the generation / pixel / batch / seed / task settings of another plan, so
    the baselines run with exactly the settings the checkpoints were evaluated with."""
    with open(a.settings_from) as f:
        src = json.load(f)
    m = dict(kv.split("=", 1) for kv in src["model_args"].split(","))
    a.min_pixels, a.max_pixels = int(m.pop("min_pixels")), int(m.pop("max_pixels"))
    attn = m.pop("attn_implementation", None)
    a.attn = attn or "sdpa"
    a.model_args = ",".join(f"{k}={v}" for k, v in m.items())
    a.seed, a.batch_size = src["seed"], src["batch_size"]
    a.limit = a.limit if a.limit is not None else src["limit"]  # an explicit --limit (quick run) wins
    a.include_path = a.include_path or src.get("include_path")
    g = src["groups"]
    want = {x.strip() for x in a.settings_groups.split(",") if x.strip()}
    unknown_g = want - set(GROUP_ORDER)
    if unknown_g:
        raise SystemExit(f"[plan] --settings_groups {sorted(unknown_g)}: groups are {GROUP_ORDER}")
    a.mcq = ",".join(g["mcq"]["tasks"]) if "mcq" in g and "mcq" in want else ""
    a.caption = ",".join(g["caption"]["tasks"]) if "caption" in g and "caption" in want else ""
    a.general = ",".join(g["general"]["tasks"]) if "general" in g and "general" in want else ""
    if "caption" in g:
        a.caption_max_new_tokens = int(dict(kv.split("=") for kv in g["caption"]["gen_kwargs"].split(","))["max_new_tokens"])
    if "caption_long" in want:  # the source plan's, or (not frozen there yet) this run's --caption_long settings
        if "caption_long" in g:
            a.caption_long = ",".join(g["caption_long"]["tasks"])
            a.caption_long_max_new_tokens = int(dict(kv.split("=") for kv in
                                                     g["caption_long"]["gen_kwargs"].split(","))["max_new_tokens"])
        else:
            a.caption_long = a.caption_long or DEFAULT_CAPTION_LONG
    else:
        a.caption_long = ""
    skip = {t.strip() for t in a.skip_tasks.split(",") if t.strip()}
    unknown = skip - {t for spec in g.values() for t in spec["tasks"]} - set(a.caption_long.split(","))
    if unknown:
        raise SystemExit(f"[plan] --skip_tasks {sorted(unknown)}: not tasks of {a.settings_from}")
    a.mcq, a.caption, a.general, a.caption_long = (",".join(t for t in s.split(",") if t and t not in skip)
                                                   for s in (a.mcq, a.caption, a.general, a.caption_long))
    print(f"[plan] settings from {a.settings_from}: {src['model_args']}, seed {a.seed}, batch {a.batch_size}, "
          f"caption max_new_tokens {a.caption_max_new_tokens}, tasks mcq={a.mcq} caption={a.caption} "
          f"general={a.general} caption_long={a.caption_long}", flush=True)


def is_adapter(path):
    return os.path.isabs(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def adapter_config(path):
    with open(os.path.join(path, "adapter_config.json")) as f:
        return json.load(f)


def lora_base(plan, path):
    """Base model an adapter is merged into: its adapter_config.json, else the plan's --lora_base."""
    return adapter_config(path).get("base_model_name_or_path") or plan.get("lora_base")


def export_info(path):
    """Training settings of a checkpoint: export.json of an exported folder, or for an adapter folder the
    run's grpo_train.json / args.json (as export_ckpt.py reads them)."""
    p = os.path.join(path, "export.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    if is_adapter(path):
        cfg = {}
        for c in reversed([os.path.join(path, "grpo_train.json"), os.path.join(path, "..", "args.json"),
                           os.path.join(path, "..", "..", "args.json")]):
            if os.path.exists(c):
                with open(c) as f:
                    cfg.update(json.load(f))
        return dict(min_pixels=56 * 56, max_pixels=cfg.get("max_pixels", 1024 * 28 * 28), train=cfg)
    return None


def pixel_limits(a, models):
    """min/max pixels: --min_pixels/--max_pixels, else the checkpoints' export.json; they must all agree."""
    if a.min_pixels and a.max_pixels:  # given (--settings_from): the replay cells' tiny stand-in model has its own
        return a.min_pixels, a.max_pixels
    seen = {n: (e["min_pixels"], e["max_pixels"]) for n, path in models.items() if (e := export_info(path))}
    if len(set(seen.values())) > 1:
        raise SystemExit(f"[plan] checkpoints were trained with different image sizes: {seen}")
    lo, hi = next(iter(seen.values())) if seen else (56 * 56, 1024 * 28 * 28)
    return a.min_pixels or lo, a.max_pixels or hi


def caption_tokens(a, models):
    if a.caption_max_new_tokens:
        return a.caption_max_new_tokens
    lens = {n: e["train"].get("max_new_tokens") for n, path in models.items()
            if (e := export_info(path)) and e.get("train")}
    if len(set(lens.values())) > 1:
        raise SystemExit(f"[plan] checkpoints were trained with different caption lengths {lens}; "
                         f"pass --caption_max_new_tokens")
    return next(iter(lens.values())) if lens else 4096


def build_plan(a, models):
    lo, hi = pixel_limits(a, models)
    groups = {}
    if a.mcq:
        groups["mcq"] = dict(tasks=a.mcq.split(","), gen_kwargs=None)
    if a.caption:
        groups["caption"] = dict(tasks=a.caption.split(","),
                                 gen_kwargs=f"max_new_tokens={caption_tokens(a, models)},temperature=0")
    if a.general:  # each task's own settings, but greedy (MM-Vet's default is temperature 0.2)
        groups["general"] = dict(tasks=a.general.split(","), gen_kwargs="temperature=0")
    if a.caption_long:  # greedy, no length cap: the same limit for every model and every task of the group
        groups["caption_long"] = dict(tasks=a.caption_long.split(","),
                                      gen_kwargs=f"max_new_tokens={a.caption_long_max_new_tokens},temperature=0")
    model_args = f"min_pixels={lo},max_pixels={hi}" + (f",{a.model_args}" if a.model_args else "")
    attn = resolve_attn(a)
    if attn:
        model_args += f",attn_implementation={attn}"
    return dict(models=models, lora_base=a.lora_base, groups=groups, model_args=model_args, seed=a.seed, limit=a.limit,
                prompting=parse_prompting(a, models),
                skip_tasks=[x.strip() for x in a.skip_tasks.split(",") if x.strip()],
                batch_size=a.batch_size, include_path=a.include_path, force_simple=True,
                lmms_eval_version=lmms_eval_version(a), created=time.strftime("%Y-%m-%d %H:%M:%S"))


def resolve_attn(a):
    """--attn auto -> flash_attention_2 when the flash-attn package imports in --python, else None (sdpa)."""
    if a.attn != "auto":
        return None if a.attn in ("none", "sdpa", "") else a.attn
    r = subprocess.run(shlex.split(a.python) + ["-c", "import flash_attn"], capture_output=True, text=True)
    attn = "flash_attention_2" if r.returncode == 0 else None
    print(f"[plan] attention: {attn or 'sdpa (flash-attn not installed in ' + a.python + ')'}", flush=True)
    return attn


def lmms_eval_version(a):
    """Version (and git commit, for a source checkout) of the lmms-eval that --python runs."""
    code = ("import importlib.metadata as m, os, lmms_eval; "
            "print(m.version('lmms_eval')); print(os.path.dirname(os.path.dirname(lmms_eval.__file__)))")
    r = subprocess.run(shlex.split(a.python) + ["-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[warn] lmms-eval not importable with {a.python}", flush=True)
        return dict(version=None, commit=None)
    v, root = r.stdout.strip().splitlines()[-2:]
    c = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return dict(version=v, commit=c.stdout.strip() if c.returncode == 0 else None)


def freeze(a, new):
    """The plan to run + a function that saves it. First run: the new plan. Later runs: the frozen plan,
    which the new one must match (new model names and new task groups may be added; a group is removed only with
    --drop_groups, which sets its results aside). Nothing is written until save() is called, so a dry run or a
    failed preflight never freezes anything."""
    path = os.path.join(a.out_dir, "plan.json")
    if not os.path.exists(path):
        return new, lambda: write_plan(path, new)
    with open(path) as f:
        old = json.load(f)
    if a.replan:
        def save():
            with open(os.path.join(a.out_dir, "plan_history.jsonl"), "a") as f:
                f.write(json.dumps(old) + "\n")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            for name in old["models"]:
                if os.path.isdir(os.path.join(a.out_dir, name)):
                    os.rename(os.path.join(a.out_dir, name), os.path.join(a.out_dir, f"_old_{stamp}_{name}"))
            print(f"[plan] --replan: old plan saved to plan_history.jsonl, old results moved to _old_{stamp}_*", flush=True)
            write_plan(path, new)
        return new, save
    # --drop_groups: those groups leave the plan (their cells are moved aside in save()); a frozen group missing
    # from this run without it is refused
    frozen = json.loads(json.dumps(old))
    drop = [g.strip() for g in (a.drop_groups or "").split(",") if g.strip()]
    unknown = [g for g in drop if g not in old["groups"]]
    if unknown:
        raise SystemExit(f"[plan] --drop_groups {unknown}: not in the frozen plan (groups: {list(old['groups'])})")
    gone = [g for g in old["groups"] if g not in new["groups"] and g not in drop]
    if gone:
        raise SystemExit(f"[plan] the frozen plan has task group(s) {gone} that this run does not. A frozen group is "
                         f"never dropped silently: pass --drop_groups {','.join(gone)} to set its results aside "
                         f"(<out_dir>/_old_<stamp>/), or run with the frozen groups")
    dropped = {g: old["groups"].pop(g) for g in drop}
    if dropped and old.get("include_path") != new.get("include_path") and not any(
            t.startswith(CUSTOM_TASKS) for g in old["groups"].values() for t in g["tasks"]):
        old["include_path"] = new.get("include_path")  # only the dropped groups used the old task folder
    # a task group the frozen plan does not have yet may be added (frozen from then on, with the time it was
    # added); the groups already there must stay exactly as they are
    new_groups = [g for g in new["groups"] if g not in old["groups"]]
    if new_groups and old.get("include_path") is None and new.get("include_path"):
        old = dict(old, include_path=new["include_path"])  # where the new group's tasks live
    # folders moved since the plan was frozen (old path gone, the new one there): the plan follows them. lmms-eval's
    # response cache keys the model by its config, not its path, so the cells' caches stay valid
    moved = {}
    for n, p_new in new["models"].items():
        p_old = old["models"].get(n)
        if p_old and p_old != p_new and os.path.isabs(p_old) and not os.path.exists(p_old) and os.path.exists(p_new):
            moved[n], old["models"][n] = (p_old, p_new), p_new
    ip_old, ip_new = old.get("include_path"), new.get("include_path")
    if ip_old and ip_new and ip_old != ip_new and not os.path.exists(ip_old) and os.path.exists(ip_new):
        moved["include_path"], old["include_path"] = (ip_old, ip_new), ip_new
    if moved:
        print("[plan] moved since the plan was frozen (old path gone, new one exists), plan updated:\n" + "\n".join(
            f"  {k}: {a} -> {b}" for k, (a, b) in moved.items()), flush=True)

    def strip(groups):
        return {g: {k: v for k, v in spec.items() if k != "added"} for g, spec in groups.items()}
    cmp_old = dict(old, groups=strip(old["groups"]))
    cmp_new = dict(new, groups=strip({g: spec for g, spec in new["groups"].items() if g in old["groups"]}))
    diff = [k for k in FROZEN if cmp_old.get(k) != cmp_new.get(k)]
    changed = [n for n in new["models"] if n in old["models"] and (old["models"][n] != new["models"][n] or
               (old.get("prompting") or {}).get(n) != (new.get("prompting") or {}).get(n))]
    if diff or changed:
        msg = [f"[plan] this differs from the frozen plan {path}:"]
        msg += [f"  {k}: frozen {cmp_old.get(k)!r}, now {cmp_new.get(k)!r}" for k in diff]
        msg += [f"  model {n}: frozen {old['models'][n]}, now {new['models'][n]}" for n in changed]
        msg.append("  Run with the frozen settings, or pass --replan to start a new plan (old results set aside).")
        raise SystemExit("\n".join(msg))
    if old.get("lmms_eval_version") != new.get("lmms_eval_version"):
        print(f"[warn] lmms-eval changed since the plan was frozen: {old.get('lmms_eval_version')} -> "
              f"{new.get('lmms_eval_version')}. Models run before and after may not be comparable.", flush=True)
    added = [n for n in new["models"] if n not in old["models"]]
    old["models"].update({n: new["models"][n] for n in added})
    old.setdefault("prompting", {}).update({n: new["prompting"][n] for n in added if n in (new.get("prompting") or {})})
    if added:
        print(f"[plan] adding model(s) {added} to the frozen plan", flush=True)
    if new_groups:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for g in new_groups:
            old["groups"][g] = dict(new["groups"][g], added=stamp)
        print(f"[plan] adding task group(s) {new_groups} to the frozen plan ({stamp}); "
              f"the groups already there are unchanged", flush=True)
    if dropped:
        print(f"[plan] --drop_groups: {list(dropped)} leave the plan; their results go to {a.out_dir}/_old_<stamp>/", flush=True)

    def save():
        if dropped:
            drop_cells(a, frozen, dropped, {g: old["groups"][g] for g in new_groups})
        if added or new_groups or dropped or moved:
            write_plan(path, old)
    return old, save


def drop_cells(a, frozen, dropped, new_groups):
    """--drop_groups: the old plan goes to plan_history.jsonl, the dropped groups' cells (results, logs, response
    cache) to <out_dir>/_old_<stamp>/<model>_<group>. Where a group added now has tasks a dropped group had, with
    the same generation settings, the dropped cell's response cache is copied into the new cell: lmms-eval keys
    its cache by task, item, generation settings and model, not by our group name, so those responses replay
    and are only scored again."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(a.out_dir, "plan_history.jsonl"), "a") as f:
        f.write(json.dumps(dict(frozen, dropped=list(dropped), dropped_at=stamp)) + "\n")
    old_root = os.path.join(a.out_dir, f"_old_{stamp}")
    for name in frozen["models"]:
        for g in dropped:
            cell = os.path.join(a.out_dir, name, g)
            if os.path.isdir(cell):
                os.makedirs(old_root, exist_ok=True)
                shutil.move(cell, os.path.join(old_root, f"{name}_{g}"))
    print(f"[plan] dropped {list(dropped)}: their cells moved to {old_root}/<model>_<group> (kept, not in the table)", flush=True)
    for ng, spec in new_groups.items():
        for g, old_spec in dropped.items():
            shared = [t for t in spec["tasks"] if t in old_spec["tasks"]]
            if not shared:
                continue
            if old_spec.get("gen_kwargs") != spec.get("gen_kwargs"):
                print(f"[plan] {ng} has {shared} as {g} did, but with other generation settings "
                      f"({old_spec.get('gen_kwargs')!r} -> {spec.get('gen_kwargs')!r}): generated again", flush=True)
                continue
            copied = []
            for name in frozen["models"]:
                src, dst = os.path.join(old_root, f"{name}_{g}", "cache"), os.path.join(a.out_dir, name, ng, "cache")
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)
                    copied.append(name)
            print(f"[plan] {ng}: response cache of {g} copied for {copied}: {shared} replay from it (scored again, "
                  f"not generated again); the other tasks of {ng} are generated", flush=True)


def write_plan(path, plan):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, path)  # never a half-written plan.json for another worker to read
    return plan


_local = {}


def complete_snapshot(d):
    idx = os.path.join(d, "model.safetensors.index.json")
    weights = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
    return all(os.path.exists(os.path.join(d, f)) for f in weights | {"config.json", "preprocessor_config.json"})


def local_copy(a, repo):
    """Local snapshot folder of a hub model id: a complete copy in the HF cache this script sees (the shared one
    included, read only), else in <hf_dir>/models, downloaded there once (~16 GB for 7B). lmms-eval then gets a
    folder path and never downloads a model into its own (per group, deleted) cache."""
    if repo in _local:
        return _local[repo]
    path = repo
    if not os.path.exists(repo) and repo.count("/") == 1:
        from huggingface_hub import snapshot_download
        own = os.path.join(a.hf_dir, "models", "hub")
        for cache in (None, own):  # None = the caches HF_HOME / HF_HUB_CACHE point to
            try:
                d = snapshot_download(repo, cache_dir=cache, local_files_only=True)
                if complete_snapshot(d):
                    path = d
                    break
            except Exception:
                pass
        if path == repo and not a.dry_run:
            need_gb(a, own, 20, f"downloading {repo}")
            print(f"[model] {repo}: no complete local copy; downloading into {own}", flush=True)
            path = snapshot_download(repo, cache_dir=own, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"])
            if not complete_snapshot(path):
                raise SystemExit(f"[model] {repo}: download incomplete at {path}")
    print(f"[model] {repo}: " + (f"local copy {path}" if path != repo else "no local copy (dry run: would download)"),
          flush=True)
    _local[repo] = path
    return path


def free_gb(path):
    while not os.path.exists(path):
        path = os.path.dirname(path) or "/"
    return shutil.disk_usage(path).free / 2**30


def need_gb(a, path, gb, what):
    free = free_gb(path)
    if free < gb:
        raise SystemExit(f"[disk] {free:.0f} GB free at {path}, {what} needs ~{gb} GB: free space first "
                         f"(this disk is shared; see `python lmms_eval_run.py --out_dir {a.out_dir} --disk`)")


def group_env(a, env, group):
    """Environment of one lmms-eval call: every download (benchmark data, hub files, xet cache, datasets' arrow
    cache) goes to <hf_dir>/datasets_<group>, which is deleted when the group is done for every model, so the
    disk holds one group's data at a time, never the shared HF cache (read only here) and never ~/.cache."""
    if not a.hf_dir:
        return env
    home = os.path.join(a.hf_dir, f"datasets_{group}")
    e = dict(env, HF_HOME=home, HF_HUB_CACHE=os.path.join(home, "hub"), HF_XET_CACHE=os.path.join(home, "xet"),
             HF_DATASETS_CACHE=os.path.join(home, "datasets"), LMMS_EVAL_DATASETS_CACHE=os.path.join(home, "datasets"),
             HF_HUB_DISABLE_XET="1")  # plain http downloads: the xet writer fails on some network disks
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if k in os.environ:
            e[k] = os.environ[k]
    if "HF_TOKEN" not in e:  # the token file of the original HF_HOME is not under the new one
        tok = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")
        if os.path.exists(tok):
            with open(tok) as f:
                e["HF_TOKEN"] = f.read().strip()
    return e


def disk_report(a):
    """What this evaluation holds on disk, and where."""
    paths = [("results + logs", a.out_dir)]
    if a.hf_dir:
        paths.append(("model downloads", os.path.join(a.hf_dir, "models")))
        if os.path.isdir(a.hf_dir):
            paths += [(f"benchmark data {g[9:]}", os.path.join(a.hf_dir, g)) for g in sorted(os.listdir(a.hf_dir))
                      if g.startswith("datasets_")]
    rows = []
    for label, path in paths:
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
                       if os.path.exists(os.path.join(r, f)))
            rows.append(f"  {size / 2**30:6.1f} GB  {label}: {path}")
    print(f"[disk] {free_gb(a.hf_dir or a.out_dir):.0f} GB free at {a.hf_dir or a.out_dir}; this evaluation holds:\n"
          + ("\n".join(rows) if rows else "  nothing yet"), flush=True)


def command(a, plan, name, group, n_gpus):
    g = plan["groups"][group]
    out = os.path.join(a.out_dir, name, group)
    base = shlex.split(a.python) + (["-m", "accelerate.commands.launch", f"--num_processes={n_gpus}",
                                     f"--main_process_port={a.port}"] if n_gpus > 1 else [])
    path = plan["models"][name]
    entry = [os.path.join(HERE, "lmms_eval_lora.py")]  # lmms-eval + scorer fixes (+ LoRA merged in memory)
    if is_adapter(path):
        entry, pretrained = entry + ["--lora", path], lora_base(plan, path)
    else:
        pretrained = path
    p = (plan.get("prompting") or {}).get(name)
    if p:
        entry += ["--prompting", p["method"], "--prompting_args", json.dumps(p["settings"])]
    if a.hf_dir:
        pretrained = local_copy(a, pretrained)
    cmd = base + entry + ["--model", "qwen2_5_vl", "--force_simple",
                          "--model_args", f"pretrained={pretrained},{plan['model_args']}",
                  "--tasks", ",".join(g["tasks"]), "--batch_size", str(plan["batch_size"]),
                  "--seed", str(plan["seed"]), "--log_samples", "--output_path", out,
                  "--use_cache", os.path.join(out, "cache")]  # every response, as generated (sqlite); see main()
    # lmms-eval exits 0 even when a task crashed (unless --fail_on_error, which older versions lack): main()
    # counts a call as done only when its results file was written
    if g["gen_kwargs"]:
        cmd += ["--gen_kwargs", g["gen_kwargs"]]
    if plan["limit"]:
        cmd += ["--limit", str(plan["limit"])]
    if plan["include_path"]:
        cmd += ["--include_path", plan["include_path"]]
    if a.launcher_args and any(t.startswith(JUDGED_TASKS) for t in g["tasks"]):
        cmd += ["--launcher_args", a.launcher_args]
    return cmd, out


def running_elsewhere(out):
    """True when another live process on this host is evaluating this cell (its running.json). A marker left by
    a dead process (killed job) is removed."""
    p = os.path.join(out, "running.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            r = json.load(f)
        if r.get("host") == socket.gethostname() and r.get("pid") != os.getpid():
            os.kill(int(r["pid"]), 0)  # raises when the process is gone
            return True
    except (OSError, ValueError, KeyError):
        pass
    if os.path.exists(p):
        print(f"[run] stale running.json in {out} (process gone): removed", flush=True)
        os.remove(p)
    return False


def reserved_gpus(a):
    """GPUs another live worker of this out_dir picked but may not have loaded its model on yet (<out_dir>/gpu_<i>.json).
    Markers of dead processes are removed."""
    taken = []
    for p in glob.glob(os.path.join(a.out_dir, "gpu_*.json")):
        try:
            with open(p) as f:
                r = json.load(f)
            if r.get("host") == socket.gethostname() and r.get("pid") != os.getpid():
                os.kill(int(r["pid"]), 0)
                taken.append(int(r["gpu"]))
                continue
        except (OSError, ValueError, KeyError):
            pass
        if os.path.exists(p):
            os.remove(p)
    return taken


def reserve_gpu(a, i):
    with open(os.path.join(a.out_dir, f"gpu_{i}.json"), "w") as f:
        json.dump(dict(gpu=i, pid=os.getpid(), host=socket.gethostname(), started=time.strftime("%Y-%m-%d %H:%M:%S")), f)


def set_aside_scores(a, plan):
    """--rescore: move a cell's done.json and lmms-eval result / sample files out of the cell (the response cache
    stays), so the next run of the cell replays the cached responses and scores them again."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for cell in [c.strip() for c in a.rescore.split(",") if c.strip()]:
        try:
            name, group = cell.split("/")
        except ValueError:
            raise SystemExit(f"[rescore] {cell!r}: expected model/group, e.g. length/caption")
        if name not in plan["models"] or group not in plan["groups"]:
            raise SystemExit(f"[rescore] {cell}: not in the plan (models {list(plan['models'])}, groups {list(plan['groups'])})")
        out = os.path.join(a.out_dir, name, group)
        if running_elsewhere(out):
            raise SystemExit(f"[rescore] {cell} is being run by another process; stop it first")
        set_aside_cell_scores(a, out, name, group, stamp)


def set_aside_cell_scores(a, out, name, group, stamp=None):
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    files = [os.path.join(out, "done.json")] + glob.glob(os.path.join(out, "**", "*_results.json"), recursive=True) \
        + glob.glob(os.path.join(out, "**", "*_samples_*.jsonl"), recursive=True)
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print(f"[rescore] {name}/{group}: nothing to set aside (not scored yet); it will run", flush=True)
        return
    dst = os.path.join(a.out_dir, f"_rescored_{stamp}", f"{name}_{group}")
    os.makedirs(dst, exist_ok=True)
    for f in files:
        shutil.move(f, os.path.join(dst, os.path.basename(f)))
    has_cache = os.path.isdir(os.path.join(out, "cache"))
    print(f"[rescore] {name}/{group}: {len(files)} file(s) moved to {dst}; "
          f"{'responses replay from the cache, scoring only' if has_cache else 'NO cache: it generates again'}", flush=True)


def judge_up(url):
    import urllib.request
    try:
        urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10).read()
        return True
    except Exception:
        return False


def wait_judge(a, tasks, poll_s=60, max_h=12, needs_llm=False):
    """Before a cell with judge-scored tasks (or a LogicCheckGPT cell, whose text LLM is the judge server): wait
    until the judge answers (it may have died since preflight, e.g. killed on a shared GPU); scoring without it
    silently produces zeros."""
    if not a.judge_url or not (needs_llm or any(t.startswith(JUDGED_TASKS) for t in tasks)):
        return
    waited = 0
    while not judge_up(a.judge_url):
        if waited == 0:
            print(f"[judge] {a.judge_url} does not answer; waiting for it (restart it: bash rl_paper_commands.sh judge)",
                  flush=True)
        if waited >= max_h * 3600:
            raise SystemExit(f"[judge] still not answering after {max_h} h; stopped")
        time.sleep(poll_s)
        waited += poll_s
    if waited:
        print(f"[judge] answering again after {waited // 60} min", flush=True)


def judge_sanity(path):
    """Judge-scored metrics as lmms-eval writes them when every judge call failed: a cell with these is not done."""
    try:
        with open(path) as f:
            res = json.load(f).get("results", {})
    except (OSError, ValueError):
        return []
    bad = []
    for task, vals in res.items():
        v = {k.split(",")[0]: x for k, x in vals.items()}
        def num(k):
            try:
                return float(v[k])
            except (KeyError, TypeError, ValueError):
                return None
        if task == "mmvet" and num("gpt_eval_score") == 0:
            bad.append(f"{task}: score 0 (lmms-eval scores an unparsable or missing judge answer as 0)")
        if task.startswith("captionqa") and num("captionqa_score") == 0:
            bad.append(f"{task}: score 0 (every judge call failed: CaptionQA scores a missing answer as 0)")
    return bad


def results_file(out, since=0):
    """lmms-eval's <out>/<model>/<date>_results.json, written only when every task of the call finished."""
    import glob
    return [p for p in glob.glob(os.path.join(out, "**", "*_results.json"), recursive=True) if os.path.getmtime(p) >= since]


def error_lines(log_path, since=0, n=25):
    """The Traceback / ERROR lines of this run from lmms-eval's log (the log is appended across runs)."""
    try:
        with open(log_path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""
    starts = [i for i, l in enumerate(lines) if "Traceback (most recent call last)" in l or "| ERROR" in l]
    if not starts:
        return "  (no Traceback / ERROR line in the log; see its end)\n  " + "\n  ".join(lines[-10:])
    i = starts[-1]
    if "Traceback" not in lines[i]:
        return "  " + "\n  ".join(lines[max(0, i - n):i + 1])
    # the traceback: its first lines and, whatever its length, its end (the exception itself)
    end = i + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or end == i + 1):
        end += 1
    end = min(end + 2, len(lines))  # the exception line (+ a note after it)
    chunk = lines[i:end] if end - i <= n else lines[i:i + n // 2] + ["  ..."] + lines[end - n // 2:end]
    return "  " + "\n  ".join(chunk)


OOM_RETRIES = 3  # a cell that dies of CUDA out of memory (shared GPU) is run again this many times


def out_of_memory(log_path, since_byte=0):
    """True when the part of the log written by this run contains a CUDA out-of-memory error."""
    try:
        with open(log_path, "rb") as f:
            f.seek(since_byte)
            tail = f.read().decode(errors="replace")
    except OSError:
        return False
    return "CUDA out of memory" in tail or "OutOfMemoryError" in tail or "CUBLAS_STATUS_ALLOC_FAILED" in tail


def pick_gpu(a, ids):
    """One free GPU for this worker (auto mode), skipping GPUs other live workers of this out_dir picked."""
    taken = reserved_gpus(a)  # a worker started minutes ago may still be loading data, its GPU still looks empty
    if taken:
        print(f"[gpu] GPU(s) {sorted(taken)} already picked by other workers of {a.out_dir}", flush=True)
    free = [i for i in ids if i not in taken] or ids
    i = gpu.pick(free, wait=True)
    reserve_gpu(a, i)
    return [i]


def preflight(a, plan):
    """Stop before hours of generation when a task's scorer is not set up."""
    tasks = ",".join(t for g in plan["groups"].values() for t in g["tasks"])
    problems = []
    judged = sorted({t for t in tasks.split(",") if t.startswith(JUDGED_TASKS)})
    if judged and not (a.judge_url or a.launcher_args):
        problems.append(f"{', '.join(judged)} need the judge LLM: --judge_url http://host:port/v1 of a running "
                        "OpenAI-compatible server (e.g. vllm serve Qwen/Qwen2.5-14B-Instruct)")
    custom = sorted({t for t in tasks.split(",") if t.startswith(CUSTOM_TASKS)})
    if custom and not (plan.get("include_path") and all(
            os.path.exists(os.path.join(plan["include_path"], f"{t}.yaml")) for t in custom)):
        problems.append(f"{', '.join(custom)}: run prepare_tasks.py --out_dir <dir> and pass --include_path <dir>")
    # things a scorer needs only after hours of generation: check them now
    if any(t.startswith(CAPTURE_TASKS + ("coco20", "coco_karpathy", "nocaps")) for t in tasks.split(",")) \
            and not shutil.which("java"):
        problems.append("caption scoring (pycocoevalcap METEOR / PTB tokenizer) needs Java: "
                        "conda install -c conda-forge openjdk=11 (in this environment), then check `java -version`")
    if any(t.startswith(CAPTURE_TASKS) for t in tasks.split(",")) and not os.environ.get("DOCCI_MOCK_METRICS"):
        # capture_metric pulls in more (factual_scene_graph, nltk data)
        r = subprocess.run(shlex.split(a.python) + ["-c", "import capture_metric.capture"], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("the CAPTURE scorer does not import: pip install capture_metric (with its dependencies) "
                            f"({next((l.strip() for l in reversed(r.stderr.splitlines()) if l.strip()), '?')[:200]})")
    if any(t.startswith("coco_cap_chair") for t in tasks.split(",")):
        code = ("import nltk; from nltk.stem import WordNetLemmatizer; "
                "WordNetLemmatizer().lemmatize('cats'); nltk.pos_tag(nltk.word_tokenize('A cat sits.'))")
        r = subprocess.run(shlex.split(a.python) + ["-c", code], capture_output=True, text=True)
        if r.returncode != 0:
            problems.append("CHAIR scoring needs NLTK data: python -m nltk.downloader punkt punkt_tab "
                            "averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet omw-1.4 "
                            f"({next((l.strip() for l in r.stderr.splitlines() if 'Resource' in l or 'Error' in l), '?')[:200]})")
    if any(t.startswith("amber_g") for t in tasks.split(",")):
        meta = os.path.join(os.environ.get("AMBER_BASE_DIR", "./data/amber"), "metadata")
        miss = [f for f in ("annotations.json", "relation.json", "safe_words.txt", "metrics.txt")
                if not os.path.exists(os.path.join(meta, f))]
        if miss:
            problems.append(f"amber_g reads AMBER's metadata from $AMBER_BASE_DIR/metadata ({meta}): {miss} missing. "
                            "Run prepare_tts_tasks.py and export AMBER_BASE_DIR=<task folder>/amber")
    if a.hf_dir:
        free = free_gb(a.hf_dir)
        if free < 10:
            problems.append(f"only {free:.0f} GB free at {a.hf_dir}; one task group's benchmark data (download + arrow "
                            f"cache) takes up to ~10 GB. Free space first (python lmms_eval_run.py --out_dir ... --disk)")
        elif free < 20:
            print(f"[warn] {free:.0f} GB free at {a.hf_dir}: enough for one group's data at a time, not much more", flush=True)
    mods = sorted({m for t in tasks.split(",") for k, ms in TASK_DEPS.items() if t.startswith(k) for m in ms})
    if os.environ.get("DOCCI_MOCK_METRICS"):  # smoke test: docci_caption's CAPTURE is replaced by a word overlap
        mods = [m for m in mods if m != "capture_metric"]
    if mods:
        code = "import importlib.util as u; print(','.join(m for m in %r if u.find_spec(m) is None))" % (mods,)
        r = subprocess.run(shlex.split(a.python) + ["-c", code], capture_output=True, text=True)
        miss = r.stdout.strip() if r.returncode == 0 else ",".join(mods)
        if miss:
            problems.append(f"python modules the task scorers need are missing in {a.python}: {miss} "
                            f"(pip install capture_metric pycocoevalcap pycocotools nltk openpyxl)")
    if judged and a.judge_url:
        import urllib.request
        try:
            urllib.request.urlopen(a.judge_url.rstrip("/") + "/models", timeout=10).read()
        except Exception as e:
            problems.append(f"judge {a.judge_url} does not answer ({e}); start it first")
    methods = {p["method"] for p in (plan.get("prompting") or {}).values()}
    if "logiccheck" in methods and not a.judge_url and not any(
            p["settings"].get("llm_url") for p in plan["prompting"].values()):
        problems.append("logiccheck needs its text LLM: --judge_url of the OpenAI-compatible judge server")
    if methods:
        mods = ["qwen_vl_utils"] + (["openai"] if "logiccheck" in methods else []) + (["nltk"] if methods - {"logiccheck"} else [])
        code = "import importlib.util as u; print(','.join(m for m in %r if u.find_spec(m) is None))" % (mods,)
        r = subprocess.run(shlex.split(a.python) + ["-c", code], capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip():
            problems.append(f"the prompting baselines need: {r.stdout.strip() or mods} (pip install qwen-vl-utils openai nltk)")
    for name, path in plan["models"].items():
        if is_adapter(path):
            if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
                problems.append(f"model {name}: {path} has adapter_config.json but no adapter_model.safetensors")
            if not lora_base(plan, path):
                problems.append(f"model {name}: adapter {path} names no base model; pass --lora_base")
        elif os.path.isabs(path) and not os.path.exists(os.path.join(path, "config.json")):
            problems.append(f"model {name}: {path} is neither a LoRA adapter nor a model folder")
        elif os.path.isabs(path) and export_info(path) is None:
            print(f"[warn] model {name}: {path} has no export.json (not made by export_ckpt.py?)", flush=True)
    return problems


def wait_free(ids, max_mem_mb=None, poll_s=60):
    """Wait until every GPU in ids is free (other people's jobs: never load a 7B model next to them)."""
    max_mem_mb = gpu.FREE_MEM_MB if max_mem_mb is None else max_mem_mb
    while True:
        g = gpu.query()
        busy = [i for i in ids if g.get(i, {}).get("used", 1 << 30) > max_mem_mb]
        if not busy:
            return ids
        print(f"[gpu] waiting for GPU(s) {busy} to free up: " + ", ".join(
            f"{i}: {g[i]['used']} MiB" if i in g else f"{i}: not found" for i in busy), flush=True)
        time.sleep(poll_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="", help="name=path,... (default: the frozen plan's models)")
    ap.add_argument("--only", default="", help="run only these model names (comma list); the plan is unchanged")
    ap.add_argument("--mcq", default=DEFAULT_MCQ, help="multiple-choice tasks, lmms-eval names ('' = none)")
    ap.add_argument("--caption", default=DEFAULT_CAPTION,
                    help="captioning tasks ('' = none); docci_caption needs --include_path from prepare_tasks.py")
    ap.add_argument("--general", default=DEFAULT_GENERAL, help="general-ability tasks, greedy ('' = none)")
    ap.add_argument("--caption_long", default="", help="caption tasks of the reward comparison, greedy, up to "
                    f"--caption_long_max_new_tokens ('' = none, the default; the comparison uses {DEFAULT_CAPTION_LONG})")
    ap.add_argument("--caption_long_max_new_tokens", type=int, default=4096,
                    help="caption_long group generation limit (no cap in practice; the training limit is not used)")
    ap.add_argument("--run_groups", default="", help="run only these task groups now (comma list); the plan is "
                    "unchanged, the other groups' cells stay to do")
    ap.add_argument("--settings_groups", default="mcq,caption,general",
                    help="with --settings_from: the groups to take over (e.g. caption_long for the test-time cells)")
    ap.add_argument("--drop_groups", default="", help="task groups to remove from the frozen plan (comma list); their "
                    "results move to <out_dir>/_old_<stamp>/ and the matching tasks of a group added now reuse their "
                    "response caches")
    ap.add_argument("--caption_max_new_tokens", type=int, default=None,
                    help="caption group generation limit (default: the checkpoints' training limit, else 4096)")
    ap.add_argument("--min_pixels", type=int, default=None, help="default: the checkpoints' training value")
    ap.add_argument("--max_pixels", type=int, default=None, help="default: the checkpoints' training value")
    ap.add_argument("--lora_base", default=None,
                    help="base model for adapters whose adapter_config.json names none (mock runs)")
    ap.add_argument("--model_args", default="", help="extra qwen2_5_vl model args, e.g. attn_implementation=flash_attention_2")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="prompts generated at once (left-padded batch, greedy): 8 is ~5x faster than 1 on captions")
    ap.add_argument("--attn", default="auto",
                    help="attention kernel in the plan: auto = flash_attention_2 if flash-attn is installed, else sdpa; "
                         "or flash_attention_2 / sdpa / none (frozen with the plan like batch_size)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N items per task: smoke tests only (frozen too)")
    ap.add_argument("--include_path", default=None, help="extra lmms-eval task folder")
    ap.add_argument("--judge_url", default=None, help="judge LLM (MM-Vet), OpenAI-compatible base url .../v1")
    ap.add_argument("--judge_model", default="Qwen/Qwen2.5-14B-Instruct", help="judge model name on that server")
    ap.add_argument("--launcher_args", default=None, help="lmms-eval --launcher_args for its own judge server")
    ap.add_argument("--gpus", default="auto",
                    help="nvidia-smi indices to use, e.g. 0 or 0,4 (data parallel; waits until they are free); "
                         "auto = the first free GPU (waits); cpu = no GPU (smoke test). Never 0, 1, 2, 3")
    ap.add_argument("--port", type=int, default=29517, help="accelerate port for multi-GPU runs")
    ap.add_argument("--hf_dir", "--hub_cache", default=None,
                    help="own folder for every download: <hf_dir>/models (hub models not in the HF_HOME cache, kept) "
                         "and <hf_dir>/datasets_<group> (benchmark data, deleted once the group is done for all "
                         "models). Models already in the HF_HOME cache (shared, read only) are used from there")
    ap.add_argument("--keep_datasets", action="store_true", help="do not delete <hf_dir>/datasets_<group> afterwards")
    ap.add_argument("--disk", action="store_true", help="print what this evaluation holds on disk and exit")
    ap.add_argument("--python", default=sys.executable, help="python of the environment with lmms-eval")
    ap.add_argument("--replan", action="store_true", help="replace the frozen plan (old results set aside)")
    ap.add_argument("--rerun", action="store_true",
                    help="run again even where done.json exists (the cell's response cache is deleted: fresh generation)")
    ap.add_argument("--keep_going", action="store_true", help="after a failed cell, try the remaining ones (default: stop)")
    ap.add_argument("--rescore", default="", help="cells (model/group, comma list) whose scoring failed or must be redone: "
                    "their done.json and result files are set aside (<out_dir>/_rescored_<stamp>/) and the cell runs "
                    "again, replaying the generated responses from its cache (scoring only)")
    ap.add_argument("--prompting", default="", help="name=method,...: run these models (normally the base model "
                    f"under another name) with a prompting baseline from another paper, one of {PROMPTING_METHODS} "
                    "(prompting_baselines.py, BASELINES.md)")
    ap.add_argument("--prompting_args", default="", help='JSON {method: {setting: value}} over the official defaults')
    ap.add_argument("--settings_from", default="", help="plan.json of the checkpoints' evaluation: use its tasks and "
                    "generation / pixel / batch / seed settings (for the baselines' own out_dir)")
    ap.add_argument("--skip_tasks", default="", help="with --settings_from: leave these tasks out of this plan (comma "
                    "list), e.g. tasks a baseline comparison does not report. Recorded in the plan")
    ap.add_argument("--dry_run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    if any(os.path.exists(os.path.join(a.out_dir, x)) for x in ("train_log.jsonl", "adapter", "adapter_config.json")):
        raise SystemExit(f"[plan] --out_dir {a.out_dir} is a GRPO run / adapter folder; use a separate folder")
    if a.hf_dir:
        a.hf_dir = os.path.abspath(a.hf_dir)
    if a.skip_tasks and not a.settings_from:
        ap.error("--skip_tasks needs --settings_from")
    if a.settings_from:
        settings_from(a)
    if a.disk:
        return disk_report(a)
    os.makedirs(a.out_dir, exist_ok=True)

    plan_path = os.path.join(a.out_dir, "plan.json")
    save = None
    if a.models:
        plan, save = freeze(a, build_plan(a, parse_models(a.models)))
    elif os.path.exists(plan_path):
        with open(plan_path) as f:
            plan = json.load(f)
    else:
        raise SystemExit("[plan] no plan yet: pass --models name=path,...")
    print("[plan] " + json.dumps({k: plan[k] for k in ("models", "groups", "model_args", "limit")}, indent=1), flush=True)

    problems = preflight(a, plan)
    if problems:
        print("\n".join(f"[preflight] {p}" for p in problems), flush=True)
        if not a.dry_run:
            raise SystemExit("[preflight] fix the above first (or leave those tasks out: --caption/--general/--mcq); "
                             "nothing was run and no plan was frozen")
    if save and not a.dry_run:
        save()
    if not a.dry_run:
        print(f"[plan] frozen: {plan_path}", flush=True)  # `par` in rl_paper_commands.sh waits for this line

    env = dict(os.environ)
    if a.judge_url:  # MM-Vet reads MODEL_VERSION / OPENAI_API_URL, CaptionQA CAPTIONQA_JUDGE_MODEL / OPENAI_API_URL
        env.update(OPENAI_API_URL=a.judge_url, MODEL_VERSION=a.judge_model, API_TYPE="openai",
                   OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY") or "EMPTY", CAPTIONQA_JUDGE_MODEL=a.judge_model)
    n_gpus = 1
    if a.gpus == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif not a.dry_run and shutil.which("nvidia-smi"):
        ids = gpu.candidates(a.gpus)
        if not ids:
            raise SystemExit("[gpu] no usable GPU in --gpus")
        ids = pick_gpu(a, ids) if a.gpus == "auto" else wait_free(ids)
        env["CUDA_VISIBLE_DEVICES"], n_gpus = ",".join(map(str, ids)), len(ids)
        print(f"[gpu] using GPU(s) {ids}", flush=True)
    # a shared GPU: other people's jobs come and go next to ours; an out-of-memory failure is retried below on a
    # GPU that is free then. (No PYTORCH_CUDA_ALLOC_CONF=expandable_segments here: the CAPTURE scorer shares CUDA
    # tensors with worker processes, which with expandable segments needs pidfd_getfd, blocked on this server.)
    if "expandable_segments:True" in env.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        print("[warn] PYTORCH_CUDA_ALLOC_CONF has expandable_segments: the CAPTURE scorer's worker processes may fail "
              "with 'pidfd_getfd: Operation not permitted'; unset it if they do", flush=True)

    if a.rescore and not a.dry_run:
        set_aside_scores(a, plan)
    names = [n for n in plan["models"] if not a.only or n in a.only.split(",")]
    run_groups = [g.strip() for g in a.run_groups.split(",") if g.strip()]
    unknown = [g for g in run_groups if g not in plan["groups"]]
    if unknown:
        raise SystemExit(f"[run] --run_groups {unknown}: not in the plan (groups: {list(plan['groups'])})")
    failed, elsewhere = [], []
    for group in plan["groups"]:  # group-major: a group's data is downloaded once, used by every model, then deleted
        if run_groups and group not in run_groups:
            continue
        genv = group_env(a, env, group)
        for name in names:
            if not prompting_applies(plan, name, group):
                print(f"[run] {name}/{group}: {plan['prompting'][name]['method']} changes none of these tasks "
                      f"(the base model's answers): skipped", flush=True)
                continue
            cmd, out = command(a, plan, name, group, n_gpus)
            done = os.path.join(out, "done.json")
            if os.path.exists(done) and not a.rerun:
                if results_file(out):
                    print(f"[run] {name}/{group}: done already", flush=True)
                    continue
                print(f"[run] {name}/{group}: done.json but no results file (an earlier run failed quietly): "
                      f"running again", flush=True)
                os.remove(done)
            if not a.dry_run and running_elsewhere(out):
                print(f"[run] {name}/{group}: running in another process (see {out}/running.json): skipped", flush=True)
                elsewhere.append(f"{name}/{group}")
                continue
            print(f"[run] {name}/{group}: {shlex.join(cmd)}", flush=True)
            if a.dry_run:
                continue
            if a.hf_dir:
                need_gb(a, a.hf_dir, 10, f"the {group} benchmark data")
            pm = (plan.get("prompting") or {}).get(name)
            wait_judge(a, plan["groups"][group]["tasks"], needs_llm=bool(pm and pm["method"] == "logiccheck"))
            if a.rerun and os.path.isdir(os.path.join(out, "cache")):
                shutil.rmtree(os.path.join(out, "cache"))  # --rerun = generate again, not replay the cache
            os.makedirs(out, exist_ok=True)
            t0 = time.time()
            # lmms-eval's response cache: each response is committed as it is generated, and a fixed run id
            # makes a later call of the same cell open the same cache. So when a scorer crashes after hours of
            # generation (or the job is killed), the next run replays the responses and only scores.
            # The cache lives in the cell's own folder: no reuse across models (LoRA cells share pretrained=base).
            run_log = os.path.join(out, "run.log")
            for attempt in range(1, OOM_RETRIES + 2):
                log_pos = os.path.getsize(run_log) if os.path.exists(run_log) else 0
                with open(os.path.join(out, "running.json"), "w") as f:  # lets other workers skip this cell
                    json.dump(dict(pid=os.getpid(), host=socket.gethostname(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                   gpus=env.get("CUDA_VISIBLE_DEVICES")), f)
                try:
                    with open(run_log, "a") as log:
                        cenv = dict(genv, LMMS_CACHE_RUN_ID="cell")
                        if pm:  # the baseline's per-item steps, for checking it (and for reviewers)
                            cenv.update(PROMPTING_TRACE=os.path.join(out, "prompting_trace.jsonl"),
                                        PROMPTING_SEED=str(plan["seed"]))
                            if a.judge_url:
                                cenv.setdefault("PROMPTING_LLM_URL", a.judge_url)
                                cenv.setdefault("PROMPTING_LLM_MODEL", a.judge_model)
                        rc = subprocess.run(cmd, env=cenv, stdout=log,
                                            stderr=subprocess.STDOUT).returncode
                finally:
                    try:
                        os.remove(os.path.join(out, "running.json"))
                    except OSError:
                        pass
                if rc == 0 and not results_file(out, since=t0):
                    rc = "0 but no results file"
                elif rc == 0:
                    bad = judge_sanity(max(results_file(out, since=t0), key=os.path.getmtime))
                    if bad:
                        rc = "0 but the judge scored nothing"
                        print(f"[run] {name}/{group}: judge-scored tasks came back empty ({'; '.join(bad)}): the judge "
                              f"was down while scoring. Start it (bash rl_paper_commands.sh judge) and run again: the "
                              f"cell only scores again (responses replay from its cache)", flush=True)
                        set_aside_cell_scores(a, out, name, group)
                if rc == 0 or attempt > OOM_RETRIES or a.gpus == "cpu" or not out_of_memory(run_log, log_pos):
                    break
                # someone else's job filled our GPU. The responses generated so far are in the cache, so a retry
                # only generates the rest: wait for a free GPU (maybe another one) and run the cell again.
                print(f"[run] {name}/{group}: CUDA out of memory on GPU {env.get('CUDA_VISIBLE_DEVICES')} (another job "
                      f"on the same GPU?). Retry {attempt}/{OOM_RETRIES} on a free GPU; generated responses are kept",
                      flush=True)
                time.sleep(120)
                if a.gpus == "auto" and shutil.which("nvidia-smi"):
                    for p in glob.glob(os.path.join(a.out_dir, "gpu_*.json")):  # drop our own reservation
                        try:
                            with open(p) as f:
                                if json.load(f).get("pid") == os.getpid():
                                    os.remove(p)
                        except (OSError, ValueError):
                            pass
                    ids = pick_gpu(a, gpu.candidates(a.gpus))
                    env["CUDA_VISIBLE_DEVICES"] = genv["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))
                    print(f"[gpu] using GPU(s) {ids}", flush=True)
                elif a.gpus != "auto" and shutil.which("nvidia-smi"):
                    wait_free([int(x) for x in env["CUDA_VISIBLE_DEVICES"].split(",")])
            if rc != 0:
                failed.append(f"{name}/{group}")
                print(f"[run] {name}/{group} FAILED (exit {rc}); see {out}/run.log", flush=True)
                print(error_lines(os.path.join(out, "run.log"), since=t0), flush=True)
                if not a.keep_going:
                    raise SystemExit(f"[run] stopped at the first failure (the other cells would most likely fail "
                                     f"the same way); fix it and run again, or pass --keep_going")
                continue
            with open(done, "w") as f:
                json.dump(dict(cmd=cmd, sec=round(time.time() - t0), finished=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2)
            print(f"[run] {name}/{group} done in {time.time() - t0:.0f}s", flush=True)
        # the group is finished for every model in the plan: its benchmark data is not needed any more
        ddir = os.path.join(a.hf_dir, f"datasets_{group}") if a.hf_dir else None
        if ddir and os.path.isdir(ddir) and not a.dry_run and not a.keep_datasets and not a.only and \
                all(results_file(os.path.join(a.out_dir, n, group)) for n in plan["models"]
                    if prompting_applies(plan, n, group)):
            size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(ddir) for f in fs
                       if os.path.exists(os.path.join(r, f))) / 2**30
            shutil.rmtree(ddir, ignore_errors=True)
            print(f"[disk] {group} done for every model: deleted its benchmark data ({size:.1f} GB, {ddir}); "
                  f"a later run downloads it again (--keep_datasets to keep it)", flush=True)
    if failed:
        raise SystemExit(f"[run] failed: {failed}")
    if elsewhere:
        print(f"[run] nothing left for this worker; still running in other workers: {elsewhere}", flush=True)
    elif not a.dry_run:
        print(f"[run] all done. Table: python lmms_eval_report.py --out_dir {a.out_dir}")


if __name__ == "__main__":
    main()
