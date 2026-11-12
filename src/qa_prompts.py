"""Prompts and JSON schemas: Caption 2, Positive/Negative/Hallucinated QA generation, checking, benchmark.

The QA generation prompt itself lives in pnh_prompt.md (edit it there). Everything above the
USER MESSAGE marker is the system message; the part below it is the user message, with {CAPTION_1}
(DOCCI human description) and {CAPTION_2} (AI caption) filled in.
"""
import os

SETS = ["positive", "negative", "hallucinated"]
CATEGORIES = ["object", "attribute", "action", "count", "spatial", "scene", "text", "relational"]
NEG_TYPES = ["contradiction", "omission"]
FORMATS = ["open-ended", "mcq"]
LETTERS = ["A", "B", "C", "D"]
PNH_PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnh_prompt.md")
_USER_MARK = "<!-- ===== USER MESSAGE"


# ------------------------------------------------------------------ Caption 2 (AI caption of the image)
# One version, "simple" (caption.py --prompt_version); "argmax" is the GRPO prompt.
CAPTION_PROMPTS = {
    "simple": "Describe this image in detail.",
    "argmax": "Describe this image.",  # the prompt grpo.py trained and evaluated the captioner with
}
CAPTION_PROMPT = CAPTION_PROMPTS["simple"]


def caption_messages(image_url, prompt=CAPTION_PROMPT):
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": prompt}]}]


# ------------------------------------------------------------------ QA generation
def load_pnh_prompt(path=PNH_PROMPT):
    with open(path) as f:
        text = f.read()
    system, user = text.split(_USER_MARK, 1)
    user = user.split("-->", 1)[1]
    return system.strip(), user.strip()


# From the prompt's "Recommendations for Downstream Data Processing": distractor rules for MCQ.
MCQ_ADDON = """

### Format for this run: MCQ
Use "format": "mcq" for every item. Add "options": {"A": "...", "B": "...", "C": "...", "D": "..."} and set
"answer" to the letter of the correct option.
- For positive and negative, take the distractors from the same semantic category as the answer
  (other colors for a color, other objects for an object, other numbers for a count).
- For hallucinated, the correct letter is the Caption 2 claim. Do not list the Caption 1 truth as an
  option{truth_rule}."""

OPEN_ADDON = """

### Format for this run: open-ended
Use "format": "open-ended" for every item."""

MIXED_ADDON = """

### Format for this run: mixed
Use "format": "open-ended" for some items and "mcq" for others. For each "mcq" item add
"options": {"A": "...", "B": "...", "C": "...", "D": "..."} and set "answer" to the correct letter.
Distractors come from the same semantic category as the answer. For hallucinated MCQ items, the correct
letter is the Caption 2 claim{truth_rule_mixed}."""


def gen_messages(caption1, caption2, qa_format="open-ended", prompt_path=PNH_PROMPT, hal_mcq_truth=False):
    system, user = load_pnh_prompt(prompt_path)
    if qa_format == "mcq":  # .replace, not .format: the add-ons contain JSON braces
        system += MCQ_ADDON.replace("{truth_rule}", " (exception for this run: DO include it as one of the "
                                    "other options, to test which one a model prefers)" if hal_mcq_truth else "")
    elif qa_format == "mixed":
        system += MIXED_ADDON.replace("{truth_rule_mixed}", ", and the Caption 1 truth is one of the other options"
                                      if hal_mcq_truth else ", and the Caption 1 truth is not an option")
    else:
        system += OPEN_ADDON
    user = user.replace("{CAPTION_1}", caption1).replace("{CAPTION_2}", caption2)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _item(extra_props, extra_required):
    props = {"id": {"type": "string"}, "question": {"type": "string"}, "answer": {"type": "string"},
             "category": {"type": "string", "enum": CATEGORIES},
             "format": {"type": "string", "enum": FORMATS},
             "options": {"type": "object", "properties": {k: {"type": "string"} for k in LETTERS}}}
    props.update(extra_props)
    return {"type": "object", "properties": props,
            "required": ["question", "answer", "category", "format"] + extra_required}


GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "positive": {"type": "array", "items": _item({}, [])},
        "negative": {"type": "array", "items": _item({"neg_type": {"type": "string", "enum": NEG_TYPES}},
                                                     ["neg_type"])},
        "hallucinated": {"type": "array", "items": _item({"caption1_truth": {"type": "string"}},
                                                         ["caption1_truth"])},
    },
    "required": ["positive", "negative", "hallucinated"],
}


# ------------------------------------------------------------------ checking (independent judge)
VERIFY_SYSTEM = """You check one question-answer item that was generated from two captions of the same image.
Caption 1 is written by a human and is the ground truth. Caption 2 is written by an AI model and may omit
or invent things.

The item belongs to one set:
- positive: both captions state the answer (same or equivalent meaning).
- negative: Caption 1 states the answer. With neg_type "contradiction", Caption 2 says something that
  conflicts with it; with neg_type "omission", Caption 2 does not mention it at all.
- hallucinated: the answer is a claim made by Caption 2 that conflicts with Caption 1, and caption1_truth
  is what Caption 1 says instead.

Decide:
- "valid": the item really fits its set (and neg_type), and the answer (and caption1_truth, options) are
  right for the captions.
- "atomic": the question asks about exactly one fact.
Output JSON only."""

VERIFY_USER = """Caption 1 (human, ground truth):
\"\"\"{caption1}\"\"\"

Caption 2 (AI model):
\"\"\"{caption2}\"\"\"

Item:
{item}

Return exactly this JSON:
{{"reason": "<one short sentence>", "valid": true|false, "atomic": true|false}}"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "valid": {"type": "boolean"}, "atomic": {"type": "boolean"}},
    "required": ["reason", "valid", "atomic"],
}


def item_text(q):
    keys = ["set", "neg_type", "question", "answer", "caption1_truth", "options", "format"]
    import json
    return json.dumps({k: q[k] for k in keys if q.get(k) not in (None, "", {})}, ensure_ascii=False)


def verify_messages(caption1, caption2, q):
    return [{"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": VERIFY_USER.format(caption1=caption1, caption2=caption2, item=item_text(q))}]


# ------------------------------------------------------------------ benchmark: the VLM answers from the image
OPEN_SUFFIX = " Answer in a few words."
MCQ_SUFFIX = "\nAnswer with the letter of the correct option only."


def bench_text(q):
    if q["format"] == "mcq":
        opts = "\n".join(f"{k}. {q['options'][k]}" for k in LETTERS if k in q["options"])
        return f"{q['question']}\n{opts}{MCQ_SUFFIX}"
    return q["question"] + OPEN_SUFFIX


def bench_messages(q, image_url=None):
    text = bench_text(q)
    if image_url is None:  # blind baseline: same question, no image
        return [{"role": "user", "content": text}]
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                         {"type": "text", "text": text}]}]


# ------------------------------------------------------------------ benchmark: grading open-ended answers
GRADE_SYSTEM = """You grade a short answer to a question about an image.
You get the question, the true answer, sometimes a known false answer (a claim an AI caption made that is
wrong for this image), and the model's answer.
- "correct": the model's answer means the same as the true answer (wording, synonyms, singular/plural,
  extra harmless detail are fine).
- "false_claim": the model's answer means the same as the known false answer.
- "wrong": anything else, including no answer or "I don't know".
Output JSON only."""

GRADE_USER = """Question: {question}
True answer: {truth}
{false_line}Model's answer: {pred}

Return exactly this JSON:
{{"reason": "<one short sentence>", "label": "correct|false_claim|wrong"}}"""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "label": {"type": "string", "enum": ["correct", "false_claim", "wrong"]}},
    "required": ["reason", "label"],
}


def grade_messages(question, truth, false, pred):
    false_line = f"Known false answer: {false}\n" if false else ""
    return [{"role": "system", "content": GRADE_SYSTEM},
            {"role": "user", "content": GRADE_USER.format(question=question, truth=truth, false_line=false_line,
                                                          pred=pred)}]
