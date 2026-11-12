"""Training-free prompting baselines from other papers, run on the base model inside lmms-eval.

Both methods are put on top of lmms-eval's own qwen2_5_vl model: its generate_until is wrapped, so the
processor, chat template, image limits (min/max pixels) and the task's prompt, scorer and generation settings
stay exactly the ones the GRPO checkpoints were evaluated with. lmms_eval_lora.py --prompting <method> does
the wrapping; lmms_eval_run.py --prompting name=method puts a baseline into a plan.

  selfval_bon  Self-Validation, Best-of-N            arXiv 2601.22451, github.com/Liushiyu-0709/SelfVal
  selfval_fta  Self-Validation, Filter-then-Aggregate (same paper, same code: llava/eval/aggreagate.py)
  logiccheck   LogicCheckGPT (logical closed loop)   arXiv 2402.11622, github.com/CRIPAC-DIG/LogicCheckGPT

Each follows the official code (selection.py / aggreagate.py / myutils.py for SelfVal, check_llava.py for
LogicCheckGPT) with its default settings; the prompts are copied verbatim. Every place where we had to deviate
(Qwen2.5-VL instead of LLaVA, the text LLM, the token limit) is listed in BASELINES.md and in DEVIATIONS below,
and written into every cell's prompting_trace.jsonl together with each item's intermediate steps.

Tasks a method does not apply to are answered by the unchanged base model (the paper's method is not defined
there); BASELINES.md says which. Self-Validation is a captioning method (the paper evaluates only COCO captions),
LogicCheckGPT a response-rewriting method for any free-form answer (POPE, MME, open-ended QA, captions).
"""
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import time

# ----------------------------------------------------------------------------------------------------------------
# settings (defaults = the official code's)

DEFAULTS = {
    "selfval_bon": dict(n=10, temperature=0.5, top_k=50, lpfv_top_k=50, lpfv_temperature=1.0),
    "selfval_fta": dict(n=3, temperature=0.5, top_k=50, lpfv_top_k=50, lpfv_temperature=1.0, alpha=0.01),
    "logiccheck": dict(threshold=0.4, extra_samples=2, sample_temperature=1.0, sample_top_k=50, lvlm_max_new_tokens=512,
                       llm_temperature=0.2, llm_max_tokens=1024, max_attr_questions=5, llm_url=None, llm_model=None,
                       llm_workers=16),
    # test-time scaling (tts.py): N candidates per image with Self-Validation's sampling (selection.py /
    # aggreagate.py: temperature 0.5, HF's top-k 50 and top-p 1); the cell's own answer is candidate 0 (= N 1)
    "tts_sample": dict(n=16, temperature=0.5, top_k=50),
    # a caption chosen offline (tts.py select) for every item: lmms-eval scores it exactly like a generated one
    "replay": dict(file=""),
}
# tasks each method changes; any other task gets the base model's answer (see BASELINES.md)
CAPTION_TASKS = ("coco_cap_chair", "docci_caption", "mock_caption")
# the caption benchmarks of the reward comparison (test-time scaling and training): CaptionQA (our copy of the
# lmms-eval task that also logs the chosen option, prepare_tts_tasks.py), AMBER, Self-Validation's CHAIR-500
CAPTION_LONG_TASKS = ("captionqa_x", "amber_g", "chair500", "mock_caption")
APPLIES = {
    "selfval_bon": CAPTION_TASKS,
    "selfval_fta": CAPTION_TASKS,
    "logiccheck": None,  # every generate_until task (the method rewrites whatever the model answered)
    "tts_sample": CAPTION_LONG_TASKS,
    "replay": CAPTION_LONG_TASKS,
}
DEVIATIONS = {
    "selfval_bon": [
        "LVLM: Qwen2.5-VL-7B (chat template, min/max pixels of our evaluation) instead of LLaVA-1.5's conv template",
        "candidate length: the task's max_new_tokens of our plan (the same limit every model gets) instead of 512",
        "sampling: temperature 0.5 + top-k 50 + top-p 1 set explicitly (the HF defaults the official LLaVA code "
        "ran with), not inherited from Qwen's generation_config.json (near-greedy top-k / top-p)",
        "LPFV: one batched teacher-forced forward per image over the unique objects of all N candidates "
        "(same probabilities as the official per-object generate() with top-k 50, temperature 1)",
    ],
    "logiccheck": [
        "LVLM: Qwen2.5-VL-7B instead of LLaVA-1.5 (same questions, greedy + 2 samples at temperature 1, 512 tokens)",
        "text LLM: the local judge server (default Qwen2.5-14B-Instruct) instead of gpt-3.5-turbo-0125; same "
        "prompts, system prompts, temperature 0.2 (0.0 for the closed-loop check), max_tokens 1024",
        "sentence split: spaCy en_core_web_lg when installed (as the code), else en_core_web_sm, else a regex",
        "an extracted entity 'None' (the extractor's 'nothing found' answer) is dropped, and an answer with no "
        "entities is returned unchanged (the code would ask about an object called 'None')",
        "a bare yes answer (Qwen answers 'Yes' to POPE/MME prompts, LLaVA in the paper answered 'Yes, there is a X "
        "in the image.') is checked with the entities of the question, so the method can act on yes/no tasks",
    ],
}
DEVIATIONS["selfval_fta"] = DEVIATIONS["selfval_bon"] + [
    "filter threshold alpha 0.01 (the paper's ablation value; the main-table value is not stated)",
]
DEVIATIONS["tts_sample"] = [
    "N candidates per image with Self-Validation's sampling (temperature 0.5, top-k 50, top-p 1), length up to the "
    "task's max_new_tokens of our plan (no 512 cap); seeded per item, so a resumed cell samples the same candidates",
    "the cell answers with candidate 0 (the N = 1 point of every selector); all N candidates, the exact image bytes "
    "and the item's fields go to prompting_trace.jsonl / images/ for tts.py",
]
DEVIATIONS["replay"] = ["no generation: the caption tts.py chose for the item is returned and scored by the task"]


def settings(method, overrides=None):
    s = dict(DEFAULTS[method])
    s.update({k: v for k, v in (overrides or {}).items() if k in s})
    return s


def applies(method, task):
    tasks = APPLIES[method]
    return tasks is None or task.startswith(tasks)


# ----------------------------------------------------------------------------------------------------------------
# the LVLM, through lmms-eval's loaded model (self of its Qwen2_5_VL class)

class VLM:
    def __init__(self, lm):
        self.lm = lm
        self.model = lm.model
        self.processor = lm.processor
        self.tok = lm.tokenizer
        self.min_px = getattr(lm, "image_min_pixels", None) or getattr(lm, "min_pixels", 256 * 28 * 28)
        self.max_px = getattr(lm, "image_max_pixels", None) or getattr(lm, "max_pixels", 1605632)
        self.system = getattr(lm, "system_prompt", None)
        self.batch = max(1, int(os.environ.get("PROMPTING_GEN_BATCH", "16")))

    @property
    def device(self):
        return next(self.model.parameters()).device

    def messages(self, images, text):
        # the image as lmms-eval's own path passes it (its JPEG data URL when it has one), so the baseline's
        # every LVLM call sees exactly the pixels the base model's answers were made from
        enc = getattr(self.lm, "_encode_image_data_url", None)
        content = [{"type": "image", "image": enc(im) if enc else im, "min_pixels": self.min_px, "max_pixels": self.max_px}
                   for im in (images or [])]
        msg = ([{"role": "system", "content": self.system}] if self.system else [])
        return msg + [{"role": "user", "content": content + [{"type": "text", "text": text}]}]

    def inputs(self, batch_msgs, padding_side="left"):
        from qwen_vl_utils import process_vision_info
        texts = self.processor.apply_chat_template(batch_msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batch_msgs)[:2]
        self.processor.tokenizer.padding_side = padding_side
        return self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True,
                              return_tensors="pt").to(self.device)

    def generate(self, items, max_new_tokens, sample=False, temperature=1.0, top_k=50):
        """items: [(images, prompt)] -> [text]. Greedy, or sampling with exactly temperature + top-k (top-p 1)."""
        import torch
        out = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            inp = self.inputs([self.messages(im, t) for im, t in chunk])
            kw = dict(do_sample=True, temperature=temperature, top_k=top_k, top_p=1.0) if sample else \
                dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                ids = self.model.generate(**inp, max_new_tokens=max_new_tokens, num_beams=1, use_cache=True,
                                          eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id, **kw)
            ids = ids[:, inp["input_ids"].shape[1]:]
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True,
                                                                   clean_up_tokenization_spaces=False)]
        return out

    def continuation_probs(self, images, prompt, continuations, top_k=50, temperature=1.0):
        """Probability of each continuation's tokens right after the assistant turn starts, as SelfVal's sample()
        measures it: softmax of the top-k / temperature warped logits at each teacher-forced step, product over
        tokens. One forward per image for all continuations (prompt repeated, continuation right-padded)."""
        import torch
        if not continuations:
            return {}
        base = self.inputs([self.messages(images, prompt)], padding_side="right")
        p_ids = base["input_ids"][0]
        toks = [self.tok(c, add_special_tokens=False).input_ids for c in continuations]
        res = {}
        for i in range(0, len(toks), self.batch):
            chunk, names = toks[i:i + self.batch], continuations[i:i + self.batch]
            L = len(p_ids) + max(len(t) for t in chunk)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(chunk), L), pad, dtype=p_ids.dtype, device=self.device)
            att = torch.zeros((len(chunk), L), dtype=base["attention_mask"].dtype, device=self.device)
            for b, t in enumerate(chunk):
                n = len(p_ids) + len(t)
                ids[b, :n] = torch.cat([p_ids, torch.tensor(t, dtype=p_ids.dtype, device=self.device)])
                att[b, :n] = 1
            kw = {}
            if "pixel_values" in base:
                kw["pixel_values"] = base["pixel_values"].repeat(len(chunk), 1)
                kw["image_grid_thw"] = base["image_grid_thw"].repeat(len(chunk), 1)
            if "mm_token_type_ids" in base:
                # which tokens are image tokens: without it transformers 5.x cannot build the 3-D (mRoPE) positions
                # and gives every token a 1-D position, so the image is read with the wrong positions (fixed
                # 2026-11-10; generate() always got it from the processor, this teacher-forced forward did not)
                tt = torch.zeros((len(chunk), L), dtype=base["mm_token_type_ids"].dtype, device=self.device)
                tt[:, :len(p_ids)] = base["mm_token_type_ids"][0]
                kw["mm_token_type_ids"] = tt
            # only the positions that predict the continuation (from the prompt's last token on): the full-sequence
            # logits of an image prompt are [batch, ~1000+ tokens, 152k] floats, tens of GB (OOM on 2026-11-11)
            keep = L - (len(p_ids) - 1)
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=keep, **kw).logits.float()
            for b, (t, name) in enumerate(zip(chunk, names)):
                prob = 1.0
                for k, tok_id in enumerate(t):
                    step = logits[b, k] / temperature  # kept position k = sequence position len(p_ids) - 1 + k
                    if top_k and top_k < step.shape[-1]:
                        kth = torch.topk(step, top_k).values[-1]
                        step = step.masked_fill(step < kth, float("-inf"))
                    prob *= torch.softmax(step, dim=-1)[tok_id].item()
                res[name] = prob
        return res


def seed_all(seed):
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------
# Self-Validation (arXiv 2601.22451). Object extraction: the default "mscoco" rule (CHAIR's caption_to_words over
# the MSCOCO objects + synonyms), loaded from lmms-eval's own coco_cap_chair task so it is the same word list.

LPFV_PROMPT = "Describe any element of the image with only one word or phrase."  # selection.py / aggreagate.py


_chair = None


def chair_words(caption):
    global _chair
    if _chair is None:
        import lmms_eval
        path = os.path.join(os.path.dirname(lmms_eval.__file__), "tasks", "coco_cap_chair", "utils.py")
        spec = importlib.util.spec_from_file_location("_selfval_chair_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _chair = mod.caption_to_words
    return _chair(caption)[0]  # evaluator.caption_to_words(caption)[0]: the MSCOCO words as written


def lpfv(vlm, images, objects, s):
    names = sorted({o.capitalize() for o in objects})  # object_name = obj.capitalize()
    return vlm.continuation_probs(images, LPFV_PROMPT, names, top_k=s["lpfv_top_k"], temperature=s["lpfv_temperature"])


def caption_score(words, probs):
    """f(y) = mean over the caption's objects of c_j; a caption with no objects scores 1.0 (cal_selfconsistency)."""
    return sum(probs[w.capitalize()] for w in words) / len(words) if words else 1.0


def filter_caption(caption, words, probs, alpha):
    """aggreagate.py filter_caption: every object word with c <= alpha is replaced by [IDX] in the caption, the
    caption is split after punctuation (incl. commas) and the pieces holding [IDX] are dropped."""
    remove = [w for w in dict.fromkeys(words) if probs[w.capitalize()] <= alpha]
    for w in remove:
        if w in caption:
            caption = caption.replace(w, "[IDX]")
    parts = re.split(r"(?<=[。！？；：，.!?;:,])\s*", caption)
    return [p for p in parts if "[IDX]" not in p], remove


def aggregate_prompt(pieces):
    """aggreagate.py regenerate(): the f-string as written there (its indentation included)."""
    return f'''
                Description from different sources:
                {'||'.join(str(c) for c in pieces)}
                Given the materials above, give the final description:
                '''


def selfval(lm, vlm, docs, method, s, max_new_tokens, seed):
    """docs: [(images, prompt)] -> ([final caption], [trace])."""
    seed_all(seed)
    cands = vlm.generate([d for d in docs for _ in range(s["n"])], max_new_tokens, sample=True,
                         temperature=s["temperature"], top_k=s["top_k"])
    finals, traces = [], []
    for j, (images, prompt) in enumerate(docs):
        cs = cands[j * s["n"]:(j + 1) * s["n"]]
        words = [chair_words(c) for c in cs]
        probs = lpfv(vlm, images, [w for ws in words for w in ws], s)
        if method == "selfval_bon":
            scores = [caption_score(ws, probs) for ws in words]
            best = max(range(len(cs)), key=lambda i: scores[i])  # np.argmax: first maximum
            finals.append(cs[best])
            traces.append(dict(candidates=cs, objects=words, lpfv=probs, scores=scores, chosen=best))
        else:
            kept, removed = [], []
            for c, ws in zip(cs, words):
                k, r = filter_caption(c, ws, probs, s["alpha"])
                kept += k
                removed.append(r)
            finals.append(None)
            traces.append(dict(candidates=cs, objects=words, lpfv=probs, removed=removed, kept=kept))
    if method == "selfval_fta":
        todo = [(docs[j][0], aggregate_prompt(traces[j]["kept"])) for j in range(len(docs))]
        finals = vlm.generate(todo, max_new_tokens)  # do_sample=False, with the image
    return finals, traces


# ----------------------------------------------------------------------------------------------------------------
# LogicCheckGPT (arXiv 2402.11622): prompts verbatim from logiccheckgpt/check_llava.py

PROMPT_TEMPLATE_PREPROCESS = """
Given a passage, you are required to replace pronouns such as "they" with the actual entities they refer to based on the context, then output the passage after replacement.
Only replace the pronouns if there is any, and do not change anything else in the original passage.
If there is nothing to replace, then keep the original sentences unchanged, do not add any new sentence.
The modification should be as small as possible, and the output passage should have the same number of sentences as the original passage.

Examples:
Passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. He appears to be smoking.

Rewritten passage:
The image depicts a kitchen scene, with a man wearing a t-shirt, a pair of shorts, and a baseball cap standing in the middle of it. The man appears to be smoking.

Passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right.

Rewritten passage:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed. The dog is in the center of the group, while the cats are positioned to the left and right.


Please only output the passage, without any explanation.
Passage:
{text}

Rewritten passage:
"""

PROMPT_TEMPLATE_TARGET = """"
You are given a sentence, extract the entities within the sentence for me.
[Task]
Your task is to extract the common objects and summarize them as general categories without repetition, merging essentially similar objects.
Avoid extracting abstract or non-specific entities.
Extract entity in the singular form. Output all the extracted types of items in one line and separate each object type with a period. If there is nothing to output, then output a single "None".
DO NOT RESPOND WITH ANYTHING ELSE.

Here are examples:
[Sentence]:
The image depicts a man laying on the ground next to a motorcycle, which appears to have been involved in a crash.
[Response]:
man.motorcycle
[Sentence]:
There are a few people around, including one person standing close to the motorcyclist and another person further away.
[Response]:
person.motorcyclist
[Sentence]:
No, there is no car in the image.
[Response]:
car
[Sentence]:
The image depicts a group of animals, with a black dog, a white kitten, and a gray cat, sitting on a bed.
[Response]:
dog.cat.bed

Now complete the following:
[Sentence]:
{sent}
[Response]:
"""

PROMPT_TEMPLATE_FACTOR_ATTRIBUTE = """
You will receive a piece of text that describes an object, and the given object.
[Task]
Your task is to accurately identify and extract every attribute associated with the given object in the provided text. Each claim should be concise (less than 15 words) and self-contained, corresponding to only one attribute.
You MUST only respond in the format as required. Each line should contain the original claim and the modified claim with all the mentions of the given object being replaced with "the object".
DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED.

[Response Format]
original claim&modified claim

Here are examples:
[Text]:
The man is wearing a baseball cap and appears to be smoking.
[Entity]:
man
[Response]:
The man is wearing a baseball cap.&The object is wearing a baseball cap.
The man appears to be smoking.&The object appears to be smoking.
[Text]:
The snowboard in the picture is a red and white snowboard, which is being ridden by a person on a snowy slope. The snowboarder is wearing a helmet and is in the middle of the slope, likely enjoying the thrill of skiing down the mountain.
[Entity]:
snowboard
[Response]:
The snowboard in the picture is red and white.&The object in the picture is red and white.
The snowboard is being ridden by a person on a snowy slope.&The object is being ridden by a person on a snowy slope.
[Text]:
The truck in the picture is a red and black pickup truck, which is parked in a driveway. It is a small truck, likely a compact or mid-size model, and appears to be in good condition.
[Entity]:
truck
[Response]:
The truck is red and black.&The object is red and black.
The truck is a pickup truck.&The object is a pickup object.
The truck is parked in a driveway.&The object is parked in a driveway.
The truck is a small truck.&The object is a small object.
The truck appears to be in good condition.&The object appears to be in good condition.
[Text]:
The dining table in the image is a wooden table with a red and black checkered tablecloth.
[Entity]:
dining table
[Response]:
The dining table in the image is made of wood.&The object in the image is made of wood.
The dining table has a red and black checkered tablecloth.&The object has a red and black checkered tablecloth.


Now complete the following:
[Text]:
{sent}
[Entity]:
{entity}
[Response]:
"""

PROMPT_TEMPLATE_REPHRASE_TORTURE_2 = """
You will receive a list of statements of objects in an image.
[Task]
Your task is to rephrase each line of statement into a question following the below question template.
In specific, extract attributes of the object to fill in the attribute slot of the template to form questions.
DO NOT RESPOND WITH ANYTHING ELSE.  DO NOT CHANGE THE QUESTION TEMPLATE.

[Template]:
Could you tell me all the objects that (ATTRIBUTE SLOT) in the image?

[Response Format]:
Could you tell me all the objects that are yellow in the image?
Could you tell me all the objects that are parked near a marketplace in the image?
Could you tell me all the objects that are made of wood in the image?
Could you tell me all the objects that are walking down the street in the image?
Could you tell me all the objects that have tablecloths on them on the street in the image?

Now complete the following:
[Statements]:
{statement}
[Response]:
"""

PROMPT_TEMPLATE_CHECK_EXIST_ENTITY = """
You are given a statement and a question.
[Task]
Your task is to answer the question based on the statement. The statement is about some objects. The question is to ask whether some specific object exists.
1. Your response should be limited to one of the following two choices: "Yes"/"No".
2. Note that instances of a certain category can also belong to its super-categories. For example, a baseball is a subclass of the sports ball.
3. Note that the table is equivalent to the dining table here.
4. DO NOT RESPOND WITH ANYTHING ELSE.

[Response Format]
Yes/No

Now complete the following:
[Statement]:
{statement}
[Question]:
Is there a {object} in the statement?
[Response]:
"""

PROMPT_TEMPLATE_REFINE_ENTITY = """
You are given a query, a passage and supplementary information.
[Task]
You are required to correct and output the refined passage in a fluent and natural style, following these rules:
1. Correct the sentences in the passage if they are inconsistent with the supplementary information. Remove the objects that are confirmed to not exist in the supplementary information.
2. Do not modify correct sentences and introduce additional information.
3. When giving refined passage, also pay attention to the given query. The refined passage should be a reasonable answer to the query.
4. Note the dining table is equivalent to the table.
Output only the corrected passage, without introducing extra contents.

Here are examples:
[Query]:
Is there a snowboard in the image?
[Passage]:
Yes, there is a snowboard in the image. The image shows a person skiing down a snow-covered slope.
[Supplementary Information]:
There is a snowboard.
There is a person.
There is
[Response]:
Yes, there is a snowboard in the image. The image shows a person skiing down a snow-covered slope.
[Query]:
Is there a sports ball in the image?
[Passage]:
Yes, there is a sports ball in the image, and it appears to be a soccer ball.
[Supplementary Information]:
There is no sports ball in the image.
[Response]:
No, there is no sports ball in the image.
[Query]:
Is there a ball in the image?
[Passage]:
Yes, there is a ball in this image.
[Supplementary Information]:
There is no ball in the image.
[Response]:
No, there is no ball in the image.
[Query]:
Describe this image.
[Passage]:
The image features a brown dog, leaping into the air to catch a frisbee. The dog is in the process of jumping, with its front paws in the air and its back legs stretched out behind it. The frisbee is visible in the air, close to the dog's mouth, as it attempts to catch it.
There are several other people in the scene, with some standing closer to the dog and others further away. A car is parked in the background, and a few chairs can be seen scattered around the area. The scene appears to be a backyard setting, with the dog and its owner enjoying a fun outdoor activity.
[Supplementary Information]:
There is no chair.
There is no people.
There is a car.
There is a dog.
There is a frisbee.
[Response]:
The image captures a brown dog leaping into the air to catch a frisbee. The dog is in the process of jumping, with its front paws in the air and its back legs stretched out behind it. The frisbee is visible in the air, close to the dog's mouth, as it attempts to catch it.
Surrounding the dog, There are no people in the scene. A car is parked in the background, and there are no chairs scattered around the area. The scene appears to be a backyard setting, with the dog and its owner enjoying a fun outdoor activity.

Now complete the following:
[Query]:
{query}
[Passage]:
{passage}
[Supplementary Information]:
{sup_info}
[Response]:
"""

SYS_PROMPT_REWRITER = "You are a language assistant that helps to rewrite a passage according to instructions."
SYS_PROMPT_EXTRACTOR = "You are a language assistant that helps to extract information from given sentences."
SYS_PROMPT_REPHRASER = "You are a language assistant that helps to rephrase sentences from given sentences."
SYS_PROMPT_ANSWER = "You are a language assistant that helps to answer the question according to instructions."
SYS_PROMPT_REFINER = "You are a language assistant that helps to refine a passage according to instructions."

GENERAL_QUESTION = ("Could you please describe the {entity} in the picture? Provide as much detailed information as "
                    "possible, including its attributes, location, or state.")


class LLM:
    """The text-side LLM (get_response of check_llava.py) on an OpenAI-compatible server."""

    def __init__(self, s):
        from openai import OpenAI
        url = s["llm_url"] or os.environ.get("PROMPTING_LLM_URL") or os.environ.get("OPENAI_API_URL")
        if not url:
            raise SystemExit("[logiccheck] no text LLM: pass llm_url (or PROMPTING_LLM_URL / --judge_url)")
        url = url.rstrip("/")
        url = url[: -len("/chat/completions")] if url.endswith("/chat/completions") else url
        self.client = OpenAI(base_url=url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self.model = s["llm_model"] or os.environ.get("PROMPTING_LLM_MODEL") or os.environ.get("MODEL_VERSION")
        self.s = s
        self.pool = cf.ThreadPoolExecutor(max_workers=s["llm_workers"])

    def __call__(self, prompt, sys_prompt, temperature=None):
        t = self.s["llm_temperature"] if temperature is None else temperature
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=t, max_tokens=self.s["llm_max_tokens"],
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}])
                return r.choices[0].message.content or ""
            except Exception as e:  # the official code retries forever; we give up after a few minutes
                print(f"[logiccheck] LLM call failed ({e}); retry {attempt + 1}", flush=True)
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError("[logiccheck] text LLM unreachable")

    def map(self, jobs):
        """jobs: [(prompt, sys_prompt, temperature)] -> answers, in parallel."""
        return list(self.pool.map(lambda j: self(*j), jobs))


_nlp = None


def split_sents(passage):
    global _nlp
    if _nlp is None:
        try:
            import spacy
            for name in ("en_core_web_lg", "en_core_web_sm"):
                try:
                    _nlp = spacy.load(name)
                    break
                except OSError:
                    pass
        except ImportError:
            pass
        _nlp = _nlp or False
    if _nlp:
        return [s.text.strip() for s in _nlp(passage).sents if s.text.strip()]
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if s.strip()]


def detect_non_exist(text):
    if text.find(".") != -1:
        text = text.split(".")[0]
    words = text.replace(",", "").split(" ")
    return "No" in words or "not" in words or "no" in words


def bare_yes(text):
    return re.fullmatch(r"\W*yes\W*", text.strip(), flags=re.I) is not None


def remove_duplicates(res):
    seen, out = set(), []
    for qs, ent in res:
        if qs not in seen:
            out.append([qs, ent])
            seen.add(qs)
    return out


def remove_identity_exposion(res, entity):
    out = []
    for qs, ans in res:
        if " is a " in qs or " is an " in qs or " be a " in qs or " be an " in qs:
            continue
        words = qs.replace(".", "").replace(",", "").split()
        if entity in words or entity + "s" in words:
            continue
        out.append([qs, ans])
    return out


def remove_irrelevant(res):
    return [[qs, ent] for qs, ent in res if "The object" in qs or "the object" in qs]


def string_match(entity, s):
    return entity in s.replace(".", "").replace(",", "").split()


def logiccheck(lm, vlm, llm, docs, answers, s, seed):
    """docs: [(images, query)], answers: the base model's greedy answers -> ([final answer], [trace]).
    Steps of check_llava.py main(), run stage by stage over a chunk of items (LVLM calls batched, LLM calls in
    parallel); each item's result is what the per-item code computes."""
    seed_all(seed)
    S = [dict(query=q, input_desc=a, images=im) for (im, q), a in zip(docs, answers)]
    active = []
    for x in S:
        if detect_non_exist(x["input_desc"]):  # early exit, as the code
            x["output"], x["exit"] = x["input_desc"], "non_exist"
        else:
            active.append(x)
    # get_rewritten: pronoun resolution, then sentence split (original and rewritten)
    for x, r in zip(active, llm.map([(PROMPT_TEMPLATE_PREPROCESS.format(text=x["input_desc"]), SYS_PROMPT_REWRITER, None)
                                     for x in active])):
        x["split_sents"] = split_sents(r)
    # get_target: entities per sentence (a bare "Yes" has none: use the query's, see DEVIATIONS)
    for x in active:
        x["entity_source"] = "answer"
        if bare_yes(x["input_desc"]):
            x["split_sents"], x["entity_source"] = split_sents(x["query"]), "query"
    jobs = [(x, PROMPT_TEMPLATE_TARGET.format(sent=sent)) for x in active for sent in x["split_sents"]]
    for (x, _), r in zip(jobs, llm.map([(p, SYS_PROMPT_EXTRACTOR, None) for _, p in jobs])):
        ents = [e.strip() for e in r.strip().split(".")]
        x.setdefault("entities", [])
        x["entities"] += [e for e in ents if e and e.lower() != "none" and e not in x["entities"]]
    for x in active:
        x.setdefault("entities", [])
        if not x["entities"]:
            x["output"], x["exit"] = x["input_desc"], "no_entities"
    active = [x for x in active if "output" not in x]
    dialogues = [(x, dict(entity=e)) for x in active for e in x["entities"]]
    for x, d in dialogues:
        x.setdefault("dialogues", []).append(d)
    # object-to-attribute inquiring: 1 greedy + 2 sampled answers per entity
    qs = [(x["images"], GENERAL_QUESTION.format(entity=d["entity"])) for x, d in dialogues]
    greedy = vlm.generate(qs, s["lvlm_max_new_tokens"])
    sampled = vlm.generate([q for q in qs for _ in range(s["extra_samples"])], s["lvlm_max_new_tokens"], sample=True,
                           temperature=s["sample_temperature"], top_k=s["sample_top_k"]) if s["extra_samples"] else []
    for i, (x, d) in enumerate(dialogues):
        d["answers"] = [greedy[i]] + sampled[i * s["extra_samples"]:(i + 1) * s["extra_samples"]]
    # attribute-to-object questions (get_question_torture): claims from the answers in turn, until > 5 questions
    def torture(d):
        cur = []
        for ans in d["answers"]:
            r = llm(PROMPT_TEMPLATE_FACTOR_ATTRIBUTE.format(sent=ans, entity=d["entity"]), SYS_PROMPT_EXTRACTOR)
            t = [item.split("&") for item in r.splitlines() if item.lower() != "none"]
            t = [[item[1], item[0]] for item in t if len(item) == 2]
            t = remove_irrelevant(remove_identity_exposion(t, d["entity"]))
            cur = remove_duplicates(cur + t)
            if len(cur) > s["max_attr_questions"]:
                break
        if not cur:
            return []
        rq = llm(PROMPT_TEMPLATE_REPHRASE_TORTURE_2.format(statement="\n".join(q[0] for q in cur)),
                 SYS_PROMPT_REPHRASER).splitlines()
        return [[rq[i], cur[i][1]] for i in range(min(len(cur), len(rq)))]
    for (x, d), t in zip(dialogues, llm.pool.map(lambda xd: torture(xd[1]), dialogues)):
        d["torture_questions"] = t
    # answers to the attribute-to-object questions (greedy)
    tq = [(x["images"], q) for x, d in dialogues for q, _ in d["torture_questions"]]
    ta = iter(vlm.generate(tq, s["lvlm_max_new_tokens"]) if tq else [])
    for x, d in dialogues:
        d["torture_answers"] = [next(ta) for _ in d["torture_questions"]]
    # logical closed-loop check: string match, else ask the LLM (temperature 0)
    checks = []
    for x, d in dialogues:
        d["NLI"] = [None] * len(d["torture_answers"])
        for k, ans in enumerate(d["torture_answers"]):
            if string_match(d["entity"], ans):
                d["NLI"][k] = "Yes"
            else:
                checks.append((d, k, PROMPT_TEMPLATE_CHECK_EXIST_ENTITY.format(statement=ans, object=d["entity"])))
    for (d, k, _), r in zip(checks, llm.map([(p, SYS_PROMPT_ANSWER, 0.0) for _, _, p in checks])):
        d["NLI"][k] = r.strip()
    # get_refinement: closed-loop rate >= threshold -> exists; refine the passage with the supplementary info
    jobs = []
    for x in active:
        pos, neg = [], []
        for d in x["dialogues"]:
            r = sum(1 for lab in d["NLI"] if lab == "Yes") / len(d["NLI"]) if d["NLI"] else 0
            d["rate"], d["entity_label"] = r, "Yes" if r >= s["threshold"] else "No"
            (pos if d["entity_label"] == "Yes" else neg).append(
                f"There is a {d['entity']}." if d["entity_label"] == "Yes" else f"There is no {d['entity']}.")
        x["sup_info"] = "\n".join(neg) + "\n" + "\n".join(pos)
        jobs.append((PROMPT_TEMPLATE_REFINE_ENTITY.format(query=x["query"], passage=x["input_desc"],
                                                          sup_info=x["sup_info"]), SYS_PROMPT_REFINER, None))
    for x, r in zip(active, llm.map(jobs)):
        x["output"], x["exit"] = r.strip(), "refined"
    traces = [{k: v for k, v in x.items() if k != "images"} for x in S]
    return [x["output"] for x in S], traces


# ----------------------------------------------------------------------------------------------------------------
# test-time scaling: sampling cell and replay cell (tts.py does the scoring and the selection offline)

def item_key(task, doc_id):
    return f"{task}/{doc_id}"


def item_seed(seed, task, doc_id):
    import zlib
    return (seed * 1_000_003 + zlib.crc32(item_key(task, doc_id).encode())) % (2**31)


def save_images(lm, images, img_dir, task, doc_id):
    """The exact bytes the model saw: lmms-eval's Qwen2.5-VL passes every image as its JPEG data URL (quality 85),
    and VLM.messages does the same, so those bytes are written. Without that encoder (none here), PNG."""
    import base64
    os.makedirs(os.path.join(img_dir, task), exist_ok=True)
    enc = getattr(lm, "_encode_image_data_url", None)
    paths = []
    for k, im in enumerate(images):
        if enc:
            url = enc(im)
            p = os.path.join(img_dir, task, f"{doc_id}_{k}.jpg")
            with open(p, "wb") as f:
                f.write(base64.b64decode(url.split(",", 1)[1]))
        else:
            p = os.path.join(img_dir, task, f"{doc_id}_{k}.png")
            im.save(p)
        paths.append(p)
    return paths


def doc_fields(doc):
    """The item's JSON-serialisable fields (CaptionQA: id, questions, shuffle_perms; AMBER: question_id, ...)."""
    out = {}
    for k, v in doc.items():
        try:
            json.dumps(v)
        except (TypeError, ValueError):
            continue  # images and other objects
        out[k] = v
    return out


def read_trace(path):
    """Items a sampling cell has done already: item key -> candidates (a killed cell resumes, nothing is redone)."""
    done = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line cut by a kill
                done[item_key(r["task"], r["doc_id"])] = r["candidates"]
    return done


def read_replay(path):
    if not path or not os.path.exists(path):
        raise SystemExit(f"[prompting] replay: caption file {path!r} not found (tts.py select writes it)")
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return {item_key(r["task"], r["doc_id"]): r["caption"] for r in rows}


def tts_sample(lm, vlm, req, s, seed, img_dir):
    context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args[:6]
    doc = lm.task_dict[task][split][doc_id]
    images = [v for v in (doc_to_visual(doc) or []) if hasattr(v, "convert")]
    prompt = strip_visual_placeholders(context)
    mnt = int(gen_kwargs.get("max_new_tokens", 1024))
    seed_all(item_seed(seed, task, doc_id))
    cands = vlm.generate([(images, prompt)] * s["n"], mnt, sample=True, temperature=s["temperature"], top_k=s["top_k"])
    return dict(task=task, doc_id=doc_id, split=split, prompt=prompt, max_new_tokens=mnt,
                images=save_images(lm, images, img_dir, task, doc_id), candidates=cands, doc=doc_fields(doc))


# ----------------------------------------------------------------------------------------------------------------
# lmms-eval hook

def strip_visual_placeholders(context):
    return re.sub(r"<image(?: \d+)?>\s*", "", context).strip()


def patch(method, overrides=None):
    """Wrap lmms-eval's Qwen2_5_VL.generate_until with the method (both module layouts of lmms-eval)."""
    s = settings(method, overrides)
    cls = None
    for mod in ("lmms_eval.models.simple.qwen2_5_vl", "lmms_eval.models.qwen2_5_vl"):
        try:
            cls = getattr(importlib.import_module(mod), "Qwen2_5_VL")
            break
        except (ImportError, AttributeError):
            continue
    if cls is None:
        raise SystemExit("[prompting] lmms-eval's Qwen2_5_VL model class not found")
    orig = cls.generate_until
    state = {}
    trace_path = os.environ.get("PROMPTING_TRACE")
    chunk = max(1, int(os.environ.get("PROMPTING_CHUNK", "8")))
    seed = int(os.environ.get("PROMPTING_SEED", "0"))

    def generate_until(self, requests):
        if "vlm" not in state:
            state["vlm"] = VLM(self)
            state["llm"] = LLM(s) if method == "logiccheck" else None
            print(f"[prompting] {method} {json.dumps(s)}", flush=True)
        vlm, llm = state["vlm"], state["llm"]
        out = [None] * len(requests)
        mine = [i for i, r in enumerate(requests) if applies(method, r.args[4])]
        rest = [i for i in range(len(requests)) if i not in set(mine)]
        need_base = rest + (mine if method == "logiccheck" else [])
        if need_base:  # the unchanged base model (lmms-eval's own path): other tasks, and LogicCheckGPT's input
            # one call per task: lmms-eval hands over the requests of every task at once, and the Qwen2.5-VL model
            # code looks up every image of a batch in the task of the batch's first item (task = task[0]), so a batch
            # straddling two tasks with the same generation settings (coco_cap_chair -> docci_caption) gets the
            # wrong images for the second task's items. Seen in baselines_quick on 2026-11-07.
            by_task = {}
            for i in need_base:
                by_task.setdefault(requests[i].args[4], []).append(i)
            for ids in by_task.values():
                base = orig(self, [requests[i] for i in ids])
                for i, b in zip(ids, base):
                    out[i] = b
        if method == "replay":
            if "replay" not in state:
                state["replay"] = read_replay(s["file"])
                print(f"[prompting] replay: {len(state['replay'])} captions from {s['file']}", flush=True)
            for i in mine:
                task, doc_id = requests[i].args[4], requests[i].args[3]
                key = item_key(task, doc_id)
                if key not in state["replay"]:
                    raise SystemExit(f"[prompting] replay: no caption for {key} in {s['file']}")
                out[i] = state["replay"][key]
            return out
        if method == "tts_sample":
            if not trace_path:
                raise SystemExit("[prompting] tts_sample needs PROMPTING_TRACE (lmms_eval_run.py sets it)")
            if "done" not in state:
                state["done"] = read_trace(trace_path)
                if state["done"]:
                    print(f"[prompting] tts_sample: {len(state['done'])} items sampled already (resumed)", flush=True)
            img_dir = os.path.join(os.path.dirname(trace_path), "images")
            t0, new = time.time(), 0
            with open(trace_path, "a") as fh:
                for i in mine:
                    key = item_key(requests[i].args[4], requests[i].args[3])
                    if key not in state["done"]:
                        rec = tts_sample(self, vlm, requests[i], s, seed, img_dir)
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        state["done"][key] = rec["candidates"]
                        new += 1
                        if new % 20 == 0:
                            print(f"[prompting] tts_sample: {new} items sampled ({time.time() - t0:.0f}s)", flush=True)
                    out[i] = state["done"][key][0]
            return out
        fh = open(trace_path, "a") if trace_path else None
        for c in range(0, len(mine), chunk):
            idx = mine[c:c + chunk]
            docs, meta = [], []
            for i in idx:
                context, gen_kwargs, doc_to_visual, doc_id, task, split = requests[i].args[:6]
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                images = [v for v in (visuals or []) if hasattr(v, "convert")]
                docs.append((images, strip_visual_placeholders(context)))
                meta.append(dict(task=task, doc_id=doc_id, max_new_tokens=int(gen_kwargs.get("max_new_tokens", 1024))))
            mnt = max(m["max_new_tokens"] for m in meta)
            item_seed = seed * 1_000_003 + int(meta[0]["doc_id"])
            if method == "logiccheck":
                finals, traces = logiccheck(self, vlm, llm, docs, [out[i] for i in idx], s, item_seed)
            else:
                finals, traces = selfval(self, vlm, docs, method, s, mnt, item_seed)
            for i, f, t, m in zip(idx, finals, traces, meta):
                out[i] = f
                if fh:
                    fh.write(json.dumps(dict(m, method=method, final=f, trace=t), default=str) + "\n")
            if fh:
                fh.flush()
        if fh:
            fh.close()
        return out

    def response_cache_identity(self):
        base = orig_identity(self) if orig_identity else None
        return dict(base or {}, prompting=method, prompting_settings=s)

    orig_identity = getattr(cls, "response_cache_identity", None)
    cls.generate_until = generate_until
    cls.response_cache_identity = response_cache_identity
    print(f"[prompting] lmms-eval Qwen2_5_VL.generate_until wrapped with {method}; "
          f"deviations from the official code: " + " | ".join(DEVIATIONS[method]), flush=True)
    return s
