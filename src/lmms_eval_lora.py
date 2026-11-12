"""lmms-eval entry point of lmms_eval_run.py: fixes for scorer packages, and a GRPO LoRA adapter merged into the
base model in memory (no exported model folder on disk).

  python lmms_eval_lora.py [--lora out_grpo/adapter] [--prompting selfval_bon|selfval_fta|logiccheck
                           [--prompting_args '{"n": 10}']] <the usual lmms-eval arguments, pretrained=<base model>>

With --prompting, lmms-eval's Qwen2_5_VL.generate_until is wrapped with a prompting baseline from another paper
(prompting_baselines.py, settings and deviations in BASELINES.md); the model, processor and tasks are unchanged.

With --lora, the base model is loaded by lmms-eval's own qwen2_5_vl code (bf16), then PeftModel +
merge_and_unload, the same steps export_ckpt.py takes before saving, so the weights are the same as an exported
folder's. Everything else (processor, chat template, generation) is lmms-eval's, untouched.

Scorer fixes (applied in every call, they change nothing when the package is absent):
  capture_metric 0.1.13 (DetailCaps CAPTURE) averages per-sample scores with statistics.mean over a mix of
  Python floats and numpy float32, which raises "don't know how to coerce float and float32" after the
  whole generation pass; its mean is replaced by a numpy mean (same value).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_scorers():
    try:
        import capture_metric.capture as C
        import numpy as np
        C.mean = lambda xs: float(np.mean([float(x) for x in xs]))
        print("[fix] capture_metric: statistics.mean -> numpy mean (float / float32 mix)", flush=True)
    except ImportError:
        pass
    patch_mmvet_judge()


MMVET_MARK = "Compare the ground truth and prediction from AI models"  # start of lmms-eval's MM-Vet judge prompt


def mmvet_score_text(content):
    """lmms-eval reads the MM-Vet judge's answer as float(first word); a local judge often explains first or
    writes 'Score: 0.5'. Reduce the answer to its last number in [0, 1] (the verdict), else leave it."""
    found = re.findall(r"(?<![\d.])(1(?:\.0+)?|0(?:\.\d+)?)(?!\.?\d)", content or "")
    return found[-1] if found else content


def patch_mmvet_judge():
    """Wrap lmms_eval.llm_judge.get_server so MM-Vet judge answers are normalised before lmms-eval parses them.
    (Task modules are loaded from files, so the task's own code cannot be patched directly.)"""
    try:
        import lmms_eval.llm_judge as J
    except ImportError:
        return
    orig = J.get_server

    def get_server(*args, **kw):
        server = orig(*args, **kw)
        evaluate = server.evaluate

        def wrapped(request, *a, **k):
            resp = evaluate(request, *a, **k)
            try:
                text = " ".join(str(m.get("content", "")) for m in (request.messages or []))
                if MMVET_MARK in text and resp is not None and resp.content:
                    resp.content = mmvet_score_text(resp.content)
            except Exception as e:  # never break scoring over the fix itself
                print(f"[fix] mmvet judge answer: {e}", flush=True)
            return resp
        server.evaluate = wrapped
        return server
    J.get_server = get_server
    print("[fix] MM-Vet: judge answer reduced to its score number before parsing", flush=True)


def patch_per_task():
    """lmms-eval hands generate_until the requests of every task of a call at once, and its Qwen2.5-VL code looks up
    every image of a batch in the task of the batch's first item (task = task[0]). A batch that straddles two tasks
    with the same generation settings therefore captions the second task's items with the first task's images
    (seen 2026-11-07 in the baselines, and 2026-11-11 in the caption_long group, whose six tasks all share one
    setting). Each task's requests are passed on their own; nothing else changes."""
    cls = None
    for mod in ("lmms_eval.models.simple.qwen2_5_vl", "lmms_eval.models.qwen2_5_vl"):
        try:
            import importlib
            cls = getattr(importlib.import_module(mod), "Qwen2_5_VL")
            break
        except (ImportError, AttributeError):
            continue
    if cls is None or getattr(cls, "_per_task_patched", False):
        return
    orig = cls.generate_until

    def generate_until(self, requests):
        by_task = {}
        for i, r in enumerate(requests):
            by_task.setdefault(r.args[4], []).append(i)
        if len(by_task) <= 1:
            return orig(self, requests)
        out = [None] * len(requests)
        for ids in by_task.values():
            for i, res in zip(ids, orig(self, [requests[i] for i in ids])):
                out[i] = res
        return out

    cls.generate_until = generate_until
    cls._per_task_patched = True
    print("[fix] Qwen2_5_VL.generate_until: one call per task (a batch never mixes two tasks' images)", flush=True)


def patch_lora(adapter):
    import transformers

    cls = transformers.Qwen2_5_VLForConditionalGeneration
    orig = cls.from_pretrained.__func__
    state = dict(done=False)

    def from_pretrained(c, *args, **kw):
        model = orig(c, *args, **kw)
        if c is cls and not state["done"]:  # the model lmms-eval evaluates; only the first load
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
            state["done"] = True
            name = args[0] if args else kw.get("pretrained_model_name_or_path")
            print(f"[lora] merged {adapter} into {name} in memory ({next(model.parameters()).dtype})", flush=True)
        return model

    cls.from_pretrained = classmethod(from_pretrained)


def main():
    argv = sys.argv[1:]
    if "--lora" in argv:
        i = argv.index("--lora")
        patch_lora(argv[i + 1])
        del argv[i:i + 2]
    prompting = None
    if "--prompting" in argv:
        i = argv.index("--prompting")
        prompting = argv[i + 1]
        del argv[i:i + 2]
    pargs = {}
    if "--prompting_args" in argv:
        i = argv.index("--prompting_args")
        pargs = json.loads(argv[i + 1])
        del argv[i:i + 2]
    patch_per_task()  # before the prompting wrapper, which then calls the per-task version
    if prompting:
        import prompting_baselines
        prompting_baselines.patch(prompting, pargs)
    if prompting == "replay" and "--model_args" in argv:
        # a replay cell generates nothing: its model (a tiny stand-in, rl_paper_commands.sh) is loaded on the CPU, so
        # the cell needs no GPU; the task's prompts, items and scorer are unchanged
        i = argv.index("--model_args")
        argv[i + 1] += ",device=cpu,device_map=cpu"
    sys.argv = [sys.argv[0]] + argv
    patch_scorers()
    from lmms_eval.cli.dispatch import main as lmms_main
    lmms_main()


if __name__ == "__main__":
    main()
