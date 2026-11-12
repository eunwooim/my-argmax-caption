"""Prompt strings for the four scores (S0-S3), exactly as the Step 1 document specifies.

Every builder returns the full model input string, already wrapped in the Qwen2.5-VL chat
template. The image is written as IMAGE_TOKEN; the processor expands <|image_pad|> into the
real number of image tokens.
"""

IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
USER = "<|im_start|>user\n"
END_USER_START_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"


def qwen_chat(user_content: str, prefill: str = "") -> str:
    """Single-turn Qwen2.5-VL chat template with add_generation_prompt=True (+ optional prefill)."""
    return SYSTEM + USER + user_content + END_USER_START_ASSISTANT + prefill


# ---------------------------------------------------------------- S0: notebook score (floor)
def s0_image() -> str:
    return qwen_chat(IMAGE_TOKEN)


def s0_caption(caption: str) -> str:
    return qwen_chat(caption)


# ---------------------------------------------------------------- S1: E5-V
# Official E5-V prompts, wrapped in the Qwen chat template instead of Llama-3.
def s1_image() -> str:
    return qwen_chat(IMAGE_TOKEN + "\nSummary above image in one word:")


def s1_caption(caption: str) -> str:
    return qwen_chat(f"{caption}\nSummary above sentence in one word:")


# ---------------------------------------------------------------- S2: FreeRet (embedding part)
# Copied verbatim from FreeRet mmeb_evaluation/input_format_oneword.py, entry 'MSCOCO_i2t'.
# String assembly follows mmeb_evaluation/run_mmeb.py lines 185 / 263:
#   system + user + instruction + content + rule text + <|im_end|>\n<|im_start|>assistant\n
FREERET_INSTRUCTION = "You are required to determine if the provided caption matches the given image."
FREERET_CAPTION_RULE = (
    "Based on the given <caption>, output exactly one word that follows these rules:\n"
    " - Output a single lowercase noun that best captures the semantics of the <caption>.\n"
    " - You may use world knowledge and paraphrasing if helpful.\n"
    " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
    " - Favor the word of the <caption> if possible.\n"
)
FREERET_IMAGE_RULE = (
    "Based on the given <image>, output exactly one word that follows these rules:\n"
    " - Output a single lowercase noun that best captures the semantics of the <image>.\n"
    " - You may use world knowledge and paraphrase if needed.\n"
    " - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.\n"
    " - Favor the caption of the <image> if possible.\n"
)


def s2_image() -> str:
    return qwen_chat(FREERET_INSTRUCTION + f"<photo>: {IMAGE_TOKEN}\n" + FREERET_IMAGE_RULE)


def s2_caption(caption: str) -> str:
    return qwen_chat(FREERET_INSTRUCTION + f"<caption>: {caption}\n" + FREERET_CAPTION_RULE)


# ---------------------------------------------------------------- S3: Lens (our version)
# User side = task instruction, role mapping, "Answer format:", readout phrase,
# <task-relevant semantics>, semantic guidance, input. Assistant side prefilled with the readout phrase.
LENS_READOUT = "The scene content is:"
LENS_INSTRUCTION = "You are required to determine if the provided caption matches the given image."


def _lens(role_line: str, guidance_line: str, input_line: str) -> str:
    user = "\n".join([
        LENS_INSTRUCTION,
        role_line,
        "Answer format:",
        LENS_READOUT,
        "<scene content>",
        guidance_line,
        input_line,
    ])
    return qwen_chat(user, prefill=LENS_READOUT)


def s3_image() -> str:
    return _lens(
        "The image provides the visual content to be matched against a caption.",
        "Describe the objects, their attributes, and their relations that are visible in the image.",
        f"<photo>: {IMAGE_TOKEN}",
    )


def s3_caption(caption: str) -> str:
    return _lens(
        "The caption provides the textual content to be matched against an image.",
        "Describe the objects, their attributes, and their relations that the caption states.",
        f"<caption>: {caption}",
    )


BUILDERS = {
    "S0": (s0_image, s0_caption),
    "S1": (s1_image, s1_caption),
    "S2": (s2_image, s2_caption),
    "S3": (s3_image, s3_caption),
}

# Where each score reads its embedding.
#   "hidden": hidden_states[layer][:, pos]   (S0, S1; default layer = hidden_states[-1], as in the doc)
#   "attn":   last decoder layer after self-attention + residual, before post_attention_layernorm
#             (forward pre-hook on post_attention_layernorm; S2, S3)
READ_POINT = {"S0": "hidden", "S1": "hidden", "S2": "attn", "S3": "attn"}
# Pooling: "last" = last position, "lastn" = mean over last n tokens, n = len(tokens(LENS_READOUT))
POOLING = {"S0": "last", "S1": "last", "S2": "last", "S3": "lastn"}
