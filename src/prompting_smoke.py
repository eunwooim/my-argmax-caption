"""Offline test of the prompting baselines (prompting_baselines.py) on CPU: no downloads, no GPU.

  1. LPFV check        SelfVal's object probability, computed batched (prompt repeated, objects right-padded), must
                       equal the official way: one sequence per object, step by step (top-k 50, temperature 1)
  2. step functions    detect_non_exist / filter_caption / caption_score / the LogicCheckGPT filters on fixed inputs
  3. lmms-eval run     tiny random Qwen2.5-VL (grpo.py --mock base, exported) x the synthetic tasks of
                       lmms_eval_smoke.py (mock_yesno, mock_caption), through lmms_eval_run.py --prompting with
                       base, selfval_bon, selfval_fta, logiccheck. LogicCheckGPT's text LLM is a fake
                       OpenAI-compatible server here that answers each prompt type in the official format.
                       Checks: every cell scored, Self-Validation skips the yes/no group (not a caption task),
                       a prompting_trace.jsonl per baseline cell with the steps of each item, and the baseline's
                       answers are the trace's finals.

The scores mean nothing (random model). Run it once on the server after installing lmms-eval:
  python prompting_smoke.py --out_dir prompting_smoke
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompting_baselines as PB  # noqa: E402
from lmms_eval_smoke import CAPTION_YAML, UTILS, YESNO_YAML, run  # noqa: E402


def first(x):
    while isinstance(x, list):
        x = x[0]
    return x


def fake_llm_answer(system, prompt):
    """Answers in the format each LogicCheckGPT prompt asks for (content made up from the prompt)."""
    if system == PB.SYS_PROMPT_REWRITER:
        return prompt.split("Passage:\n")[-1].split("\n\nRewritten passage:")[0]
    if system == PB.SYS_PROMPT_EXTRACTOR and "[Entity]:" not in prompt:
        sent = prompt.split("[Sentence]:\n")[-1].lower()
        ents = [w for w in ("circle", "square", "triangle", "image", "red", "blue") if w in sent]
        return ".".join(ents) if ents else "None"
    if system == PB.SYS_PROMPT_EXTRACTOR:
        ent = prompt.split("[Entity]:\n")[-1].split("\n")[0]
        return (f"The {ent} is red.&The object is red.\nThe {ent} is in the middle.&The object is in the middle.\n"
                f"The {ent} is a {ent}.&The object is a object.")
    if system == PB.SYS_PROMPT_REPHRASER:
        stm = prompt.split("[Statements]:\n")[-1].split("\n[Response]:")[0].splitlines()
        return "\n".join(f"Could you tell me all the objects that {s.split('The object ')[-1].rstrip('.')} in the image?"
                         for s in stm)
    if system == PB.SYS_PROMPT_ANSWER:
        return "Yes" if len(prompt) % 2 else "No"
    if system == PB.SYS_PROMPT_REFINER:
        return prompt.split("[Passage]:\n")[-1].split("\n[Supplementary Information]:")[0] + " (refined)"
    return "None"


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_GET(self):  # /v1/models
        self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next(m["content"] for m in msgs if m["role"] == "user")
        Handler.calls.append(dict(system=system, temperature=body.get("temperature")))
        text = fake_llm_answer(system, user)
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def check_lpfv(model_dir):
    """Batched LPFV == per-object teacher forcing with a fresh forward per step (the official sample() loop)."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import mock_data as M

    class LM:  # what VLM reads from lmms-eval's Qwen2_5_VL
        pass
    lm = LM()
    lm.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    lm.processor = AutoProcessor.from_pretrained(model_dir, min_pixels=56 * 56, max_pixels=64 * 28 * 28)
    lm.tokenizer = lm.processor.tokenizer
    lm.image_min_pixels, lm.image_max_pixels, lm.system_prompt = 56 * 56, 64 * 28 * 28, None
    vlm = PB.VLM(lm)
    vlm.batch = 3  # several chunks
    img = [M.image("synth_5000.png")]
    objs = ["Dog", "Dining table", "Person", "Cat", "Traffic light"]
    base = vlm.inputs([vlm.messages(img, PB.LPFV_PROMPT)])
    # top-k 50 (the official setting) mostly gives 0 on a random model, so check without the cut-off too
    for top_k in (50, None):
      got = vlm.continuation_probs(img, PB.LPFV_PROMPT, objs, top_k=top_k, temperature=1.0)
      for o in objs:
        ids = base["input_ids"]
        tt = base["mm_token_type_ids"]  # image tokens marked, as the processor gives them (3-D positions)
        prob = 1.0
        for t in lm.tokenizer(o, add_special_tokens=False).input_ids:
            with torch.inference_mode():
                logits = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=base["pixel_values"],
                                  image_grid_thw=base["image_grid_thw"], mm_token_type_ids=tt).logits[0, -1].float()
            if top_k:
                kth = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            prob *= torch.softmax(logits, -1)[t].item()
            ids = torch.cat([ids, torch.tensor([[t]])], 1)
            tt = torch.cat([tt, torch.zeros((1, 1), dtype=tt.dtype)], 1)
        assert abs(prob - got[o]) <= 1e-3 * prob + 1e-12, f"LPFV {o} top_k {top_k}: batched {got[o]} != stepwise {prob}"
      print(f"[smoke] LPFV batched == stepwise (top_k {top_k}) for {objs}: ok ({ {k: f'{v:.2e}' for k, v in got.items()} })")
    # the teacher-forced forward must read the image the way generate() does: first-step distribution of
    # generate (output_scores, greedy) == the forward's distribution at the same position
    with torch.inference_mode():
        g = lm.model.generate(**base, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True,
                              temperature=None, top_p=None, top_k=None)
    dist = torch.softmax(g.scores[0][0].float(), -1)
    checked = 0
    for tid in torch.topk(dist, 50).indices.tolist():  # tokens whose text tokenizes back to the same single token
        name = lm.tokenizer.decode([tid])
        if not name.strip() or lm.tokenizer(name, add_special_tokens=False).input_ids != [tid]:
            continue
        p_fwd = vlm.continuation_probs(img, PB.LPFV_PROMPT, [name], top_k=None, temperature=1.0)[name]
        assert abs(dist[tid].item() - p_fwd) <= 1e-3 * dist[tid].item() + 1e-9, \
            f"LPFV forward reads the image differently from generate(): {name!r} {p_fwd} vs {dist[tid].item()}"
        checked += 1
        if checked == 3:
            break
    assert checked, "no single-token continuation to compare with generate()"
    print(f"[smoke] LPFV forward == generate() first-step probability for {checked} tokens: image positions as in "
          f"generation: ok")
    # sampling really samples (Qwen's generation_config would make it greedy)
    outs = vlm.generate([(img, "Describe this image.")] * 4, 12, sample=True, temperature=0.5, top_k=50)
    assert len(set(outs)) > 1, f"sampled candidates are identical: {outs}"
    print("[smoke] Self-Validation candidates differ (sampling on): ok")


class FakeVLM:
    """Stands in for the LVLM to run each method's full logic on text a random model never writes."""
    CAPTIONS = ["A dog sits on a sofa next to a cat. A lamp is on the table.",
                "A dog lies on a sofa, a bottle stands on the table.",
                "A dog and a person are in a living room."]

    def __init__(self):
        self.calls = []

    def generate(self, items, max_new_tokens, sample=False, temperature=1.0, top_k=50):
        self.calls += [(p, sample) for _, p in items]
        out = []
        for i, (_, p) in enumerate(items):
            if p.startswith("Could you please describe the"):
                ent = p.split("describe the ")[1].split(" in the picture")[0]
                out.append(f"The {ent} is brown and is in the middle of the room.")
            elif p.startswith("Could you tell me all the objects"):
                out.append("There is a dog and a sofa." if "brown" in p else "I see a lamp.")
            elif "Description from different sources" in p:
                out.append("AGGREGATED: " + p.split("sources:")[1].split("Given")[0].strip())
            else:
                out.append(self.CAPTIONS[i % 3] if sample else self.CAPTIONS[0])
        return out

    def continuation_probs(self, images, prompt, names, top_k=50, temperature=1.0):
        assert prompt == PB.LPFV_PROMPT
        p = {"Dog": 0.3, "Sofa": 0.2, "Cat": 0.001, "Table": 0.05, "Bottle": 0.004, "Person": 0.1, "Lamp": 0.02}
        return {n: p.get(n, 0.0) for n in names}


def check_methods(url):
    v = FakeVLM()
    docs = [([], "Please describe this image in detail.")]
    finals, tr = PB.selfval(None, v, docs, "selfval_bon", PB.settings("selfval_bon", {"n": 3}), 32, 0)
    # scores: (dog .3 + sofa .2 + cat .001 + table .05) / 4 = .138; (.3+.2+.004+.05)/4 = .1385; (.3+.1)/2 = .2
    assert tr[0]["chosen"] == 2 and finals[0] == FakeVLM.CAPTIONS[2], tr
    finals, tr = PB.selfval(None, v, docs, "selfval_fta", PB.settings("selfval_fta", {"n": 3}), 32, 0)
    assert sorted(tr[0]["removed"][0]) == ["cat"] and tr[0]["removed"][1] == ["bottle"], tr[0]["removed"]
    assert not any("cat" in k or "bottle" in k for k in tr[0]["kept"]) and finals[0].startswith("AGGREGATED:"), tr
    print(f"[smoke] Self-Validation logic: BoN picks the best-verified caption, FtA drops cat/bottle clauses: ok")
    s = PB.settings("logiccheck", {"llm_url": url, "llm_model": "fake"})
    llm = PB.LLM(s)
    docs = [([], "Please describe this image in detail."), ([], "Is there a red circle in the image?"),
            ([], "Is there a dog?"), ([], "Which option? A. dog B. cat")]
    answers = ["A red circle is next to a blue square.", "Yes", "No, there is no dog.", "B"]
    finals, tr = PB.logiccheck(None, v, llm, docs, answers, s, 0)
    assert [t["exit"] for t in tr] == ["refined", "refined", "non_exist", "no_entities"], [t.get("exit") for t in tr]
    assert tr[1]["entity_source"] == "query" and finals[2] == answers[2] and finals[3] == "B"
    d = tr[0]["dialogues"][0]
    assert len(d["answers"]) == 3 and d["torture_questions"] and len(d["NLI"]) == len(d["torture_questions"]), d
    assert "There is" in tr[0]["sup_info"] and finals[0].endswith("(refined)"), tr[0]
    print(f"[smoke] LogicCheckGPT logic: full closed loop + refinement, early exits, bare-yes uses the query: ok "
          f"(entities {tr[0]['entities']}, rates {[round(x['rate'], 2) for x in tr[0]['dialogues']]})")


def check_steps():
    assert PB.detect_non_exist("No, there is no dog.") and not PB.detect_non_exist("Yes, there is a dog.")
    assert PB.bare_yes("Yes") and PB.bare_yes(" yes. ") and not PB.bare_yes("Yes, a dog")
    kept, removed = PB.filter_caption("A dog sits on a sofa, next to a cat. The room is bright.", ["dog", "cat"],
                                      {"Dog": 0.5, "Cat": 0.001}, 0.01)
    assert removed == ["cat"] and kept == ["A dog sits on a sofa,", "The room is bright.", ""], kept
    assert PB.caption_score([], {}) == 1.0 and PB.caption_score(["dog", "cat"], {"Dog": 0.4, "Cat": 0.2}) == 0.30000000000000004
    t = PB.remove_irrelevant(PB.remove_identity_exposion(
        [["The object is red.", "a"], ["The object is a truck.", "b"], ["The truck is red.", "c"], ["It is red.", "d"]],
        "truck"))
    assert t == [["The object is red.", "a"]], t
    assert PB.chair_words("A man rides a bicycle next to a dining table.") == ["man", "bicycle", "table"]
    print("[smoke] step functions on fixed inputs: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="prompting_smoke")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    out = os.path.abspath(a.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    py = sys.executable
    env = dict(os.environ, HF_DATASETS_OFFLINE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")
    run([py, os.path.join(HERE, "export_ckpt.py"), "--mock", "--dtype", "bfloat16", "--out", f"{out}/base"], env=env)

    check_steps()
    check_lpfv(f"{out}/base")

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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    check_methods(url)
    # the baselines' own out_dir, settings taken from the checkpoints' plan (as rl_paper_commands.sh baselines does)
    ref = f"{out}/lmms_ref"
    os.makedirs(ref)
    with open(f"{ref}/plan.json", "w") as f:
        json.dump(dict(models={}, groups={"mcq": dict(tasks=["mock_yesno"], gen_kwargs=None),
                                          "caption": dict(tasks=["mock_caption"], gen_kwargs="max_new_tokens=16,temperature=0")},
                       model_args="min_pixels=3136,max_pixels=50176,device=cpu,device_map=cpu", seed=0, limit=None,
                       batch_size=2, include_path=tasks), f)
    prompting_args = json.dumps({"selfval_bon": {"n": 3}, "logiccheck": {"lvlm_max_new_tokens": 12}})
    run([py, os.path.join(HERE, "lmms_eval_run.py"), "--out_dir", f"{out}/lmms", "--settings_from", f"{ref}/plan.json",
         "--models", ",".join(f"{n}={out}/base" for n in ("base", "selfval_bon", "selfval_fta", "logiccheck")),
         "--prompting", "selfval_bon=selfval_bon,selfval_fta=selfval_fta,logiccheck=logiccheck",
         "--prompting_args", prompting_args, "--judge_url", url, "--judge_model", "fake", "--gpus", "cpu"],
        env=dict(env, PROMPTING_GEN_BATCH="4"))
    with open(f"{out}/lmms/plan.json") as f:
        plan = json.load(f)
    assert plan["prompting"]["selfval_bon"]["settings"]["n"] == 3 and plan["prompting"]["selfval_fta"]["settings"]["n"] == 3
    assert plan["groups"]["caption"]["gen_kwargs"] == "max_new_tokens=16,temperature=0", plan["groups"]
    for name in ("selfval_bon", "selfval_fta"):
        assert not os.path.exists(f"{out}/lmms/{name}/mcq"), f"{name}: the yes/no group should be skipped"
    print("[smoke] Self-Validation cells only on the caption group; settings copied from the reference plan: ok")
    for name, groups in (("selfval_bon", ["caption"]), ("selfval_fta", ["caption"]), ("logiccheck", ["mcq", "caption"])):
        for g in groups:
            cell = f"{out}/lmms/{name}/{g}"
            with open(f"{cell}/run.log") as f:
                assert f"wrapped with {name}" in f.read(), f"{name}/{g}: not wrapped"
            with open(f"{cell}/prompting_trace.jsonl") as f:
                tr = [json.loads(l) for l in f]
            assert len(tr) == a.n, f"{name}/{g}: {len(tr)} traces for {a.n} items"
            task = "mock_yesno" if g == "mcq" else "mock_caption"
            p = glob.glob(f"{cell}/**/*_samples_{task}.jsonl", recursive=True)
            with open(p[0]) as f:
                resp = {r["doc_id"]: first(r.get("filtered_resps", r.get("resps"))) for r in map(json.loads, f)}
            assert {t["doc_id"]: t["final"] for t in tr} == resp, f"{name}/{g}: answers are not the traces' finals"
            if name == "selfval_bon":
                assert all(len(t["trace"]["candidates"]) == 3 for t in tr)
            if name == "logiccheck":
                exits = [t["trace"]["exit"] for t in tr]
                print(f"[smoke] logiccheck/{g} exits: {exits}")
    systems = {c["system"] for c in Handler.calls}
    assert PB.SYS_PROMPT_REFINER in systems and PB.SYS_PROMPT_REPHRASER in systems, systems
    assert any(c["temperature"] == 0.0 for c in Handler.calls if c["system"] == PB.SYS_PROMPT_ANSWER) or \
        PB.SYS_PROMPT_ANSWER not in systems
    print(f"[smoke] LogicCheckGPT ran its full loop through the text LLM ({len(Handler.calls)} calls): ok")
    run([py, os.path.join(HERE, "lmms_eval_report.py"), "--out_dir", f"{out}/lmms"])
    server.shutdown()
    print("[smoke] OK")


if __name__ == "__main__":
    main()
