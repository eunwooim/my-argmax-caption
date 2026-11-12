"""Frozen Qwen2.5-VL embedder for the four scores (doc sections 1-2).

embed(score, image=...) or embed(score, caption=...) returns a [n_layers, d] float32 tensor:
row l-1 is the embedding read at decoder layer l (1..L), at the score's read point and pooling.
The doc's default reading is row L-1 (the last layer).

Read points
  hidden (S0, S1): hidden_states[l][:, pos]. hidden_states[L] == hidden_states[-1], which in
                   transformers already has the final RMSNorm applied, exactly as the doc's
                   hidden_states[-1][:, -1, :].
  attn   (S2, S3): input of layer l's post_attention_layernorm, i.e. after self-attention and its
                   residual, before the norm and the MLP (forward pre-hook, as the doc says).
"""
import importlib.util
import os

import torch

import prompts as P


def attn_implementation():
    """Attention kernel for Qwen2.5-VL: ATTN_IMPL env (flash_attention_2 / sdpa / eager), else flash_attention_2
    when the flash-attn package is installed, else None (transformers' default, sdpa). Same numbers up to
    bf16 rounding; flash attention is ~1.5-2x faster on the long image-token inputs."""
    impl = os.environ.get("ATTN_IMPL", "auto")
    if impl == "auto":
        impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") else None
    return impl or None


def _decoder_layers(model):
    m = model.model
    lm = getattr(m, "language_model", m)
    return lm.layers


class Embedder:
    def __init__(self, model, processor, device="cuda"):
        self.model = model.eval()
        self.processor = processor
        self.tok = processor.tokenizer
        self.tok.padding_side = "left"  # doc 1: irrelevant at batch size 1, set anyway
        self.device = device
        self.layers = _decoder_layers(model)
        self.L = len(self.layers)
        # S3: n = number of tokens in "The scene content is:"
        self.lens_n = len(self.tok(P.LENS_READOUT, add_special_tokens=False)["input_ids"])
        self._captured = None
        self._pool = None
        for i, layer in enumerate(self.layers):
            layer.post_attention_layernorm.register_forward_pre_hook(self._make_hook(i))

    def _make_hook(self, i):
        def hook(module, args):
            if self._captured is not None:
                self._captured[i] = self._pool(args[0])
        return hook

    def _pool_fn(self, score):
        if P.POOLING[score] == "last":
            return lambda h: h[0, -1, :].float()
        n = self.lens_n
        return lambda h: h[0, -n:, :].float().mean(0)

    @torch.no_grad()
    def embed(self, score, image=None, caption=None):
        img_b, cap_b = P.BUILDERS[score]
        if image is not None:
            text, images = img_b(), [image]
        else:
            text, images = cap_b(caption), None
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        pool = self._pool_fn(score)
        if P.READ_POINT[score] == "attn":
            self._captured, self._pool = [None] * self.L, pool
            try:
                self.model.model(**inputs, use_cache=False)
            finally:
                out, self._captured = self._captured, None
            return torch.stack(out).cpu()
        hs = self.model.model(**inputs, use_cache=False, output_hidden_states=True).hidden_states
        return torch.stack([pool(h) for h in hs[1:]]).cpu()


# ------------------------------------------------------------------ loading
def load_real(model_name, max_pixels, device, dtype="bfloat16"):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_name, size={"shortest_edge": 56 * 56, "longest_edge": max_pixels})
    attn = attn_implementation()
    print(f"[model] attention: {attn or 'sdpa (transformers default)'}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=getattr(torch, dtype), device_map=device, **({"attn_implementation": attn} if attn else {}))
    model.requires_grad_(False)
    _check_template(processor)
    return Embedder(model, processor, device)


def _check_template(processor):
    """Warn if our hand-built chat string differs from the processor's own template."""
    try:
        ref = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}]}], add_generation_prompt=True, tokenize=False)
        if ref != P.s0_image():
            print("[warn] processor chat template differs from prompts.qwen_chat():")
            print("  processor:", repr(ref))
            print("  ours     :", repr(P.s0_image()))
    except Exception as e:  # template check is advisory only
        print("[warn] could not check chat template:", e)


def load_mock(max_pixels, device="cpu", seed=0, num_layers=4):
    """Tiny randomly initialised Qwen2.5-VL with a byte-level tokenizer. No downloads.

    Same architecture code path, hooks and processor as the real model, so the whole pipeline
    can be smoke-tested on CPU. Its scores are meaningless (expect ~50%).
    """
    from transformers import (Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                              Qwen2_5_VLProcessor, Qwen2TokenizerFast)
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    try:
        from transformers.convert_slow_tokenizer import bytes_to_unicode
    except ImportError:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    torch.manual_seed(seed)
    vocab = {ch: i for i, ch in enumerate(bytes_to_unicode().values())}
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>",
                "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
    tok = Qwen2TokenizerFast(vocab=vocab, merges=[])
    tok.add_special_tokens({"additional_special_tokens": specials})
    ids = {s: tok.convert_tokens_to_ids(s) for s in specials}

    improc = Qwen2VLImageProcessorPil(size={"shortest_edge": 56 * 56, "longest_edge": max_pixels})
    processor = Qwen2_5_VLProcessor(image_processor=improc, tokenizer=tok,
                                    video_processor=Qwen2VLVideoProcessor())

    hidden, heads = 64, 4
    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=hidden, intermediate_size=128, num_hidden_layers=num_layers,
                         num_attention_heads=heads, num_key_value_heads=2, vocab_size=len(tok),
                         max_position_embeddings=4096, bos_token_id=ids["<|endoftext|>"],
                         eos_token_id=ids["<|im_end|>"],
                         rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[2, 3, 3])),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=hidden, fullatt_block_indexes=[1]),
        bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"],
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
    )
    model = Qwen2_5_VLForConditionalGeneration(cfg).to(device)
    model.requires_grad_(False)
    return Embedder(model, processor, device)
