"""The comparison rewards for grpo.py (the default reward, the model's own S2 score, is grpo.Reward; the
self-certainty reward needs the policy's logits and lives in grpo.py too).

  ExternalReward  cos(image, caption) from a separate image-text embedding model (CLIP / SigLIP family, or a
                  remote-code model with encode_image / encode_text such as jina-clip-v2). Frozen, other weights.
  LongCLIPReward  cos(image, caption) from Long-CLIP (Zhang et al., ECCV 2024; github.com/beichenzbc/Long-CLIP),
                  its official code and checkpoint (longclip-L.pt); text limit 248 tokens, longer captions are cut
                  (truncate=True, as its README's tokenize) and counted. Long-CLIPScore is 2.5 * max(cos, 0),
                  the same ranking; the reward is the cosine, like ExternalReward.
  LengthReward    -|words - target| / target: GRPO only moves caption length, to the length the self-reward run
                  reached at the same step (length-matched control). No image signal at all.
  RandomReward    uniform [0, 1) per caption, independent of image and text: the no-signal control.

All have the interface grpo.py uses: reward(key, image, captions) -> list of floats, and .name.
"""
import json
import os
import random
import sys

import torch


def _patch_clip_loss():
    """jina-clip-v2's remote code imports clip_loss from transformers' CLIP module, which newer transformers
    dropped. It is only the training loss, never called for embeddings, so put the old definition back."""
    import transformers.models.clip.modeling_clip as mc
    if hasattr(mc, "clip_loss"):
        return

    def contrastive_loss(logits):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    mc.contrastive_loss = getattr(mc, "contrastive_loss", contrastive_loss)
    mc.clip_loss = lambda similarity: (mc.contrastive_loss(similarity) + mc.contrastive_loss(similarity.t())) / 2


def _rebuild_vision_tower(model, dtype, device):
    """Newer transformers build models on the meta device and then fill in only the saved weights, so buffers
    that a remote-code __init__ computes but never saves (jina-clip-v2's EVA rotary tables) come out as garbage
    and the image embedding is NaN. The text tower is fine. Build the vision tower again for real with the
    model's own builder, copy the loaded weights into it, and swap it in."""
    import sys
    mod = sys.modules[type(model).__module__]
    if not hasattr(mod, "_build_vision_tower") or not hasattr(model, "vision_model"):
        raise SystemExit("[reward] image embeddings are NaN and this model has no _build_vision_tower to rebuild "
                         "with; run it with transformers 4.x or pick another --external_model")
    print("[reward] rebuilding the vision tower (its unsaved buffers were not initialised)", flush=True)
    fresh = mod._build_vision_tower(model.config.vision_config)
    fresh.load_state_dict(model.vision_model.state_dict(), strict=True)
    model.vision_model = fresh.to(device, dtype).eval()
    model.requires_grad_(False)
    return model


class ExternalReward:
    def __init__(self, model_name, device, dtype="float32", mock=False):
        self.device, self.img_cache = device, {}
        self.n_texts = self.n_truncated = 0
        if mock or model_name == "mock":
            self._load_mock()
            self.name = "ext:mock"
            return
        from transformers import AutoModel
        dt = getattr(torch, dtype)  # float32 by default: small model, and bf16 can give NaN / coarse cosines
        _patch_clip_loss()
        self.model = AutoModel.from_pretrained(model_name, dtype=dt, trust_remote_code=True).to(device).eval()
        self.model.requires_grad_(False)
        self.name = "ext:" + model_name.rstrip("/").split("/")[-1]
        if hasattr(self.model, "encode_text") and hasattr(self.model, "encode_image"):
            self.kind = "remote"  # jina-clip-v2 and other remote-code embedders: their documented API
            self.tok = None
            if not self._finite():
                self.model = _rebuild_vision_tower(self.model, dt, device)
                if not self._finite():
                    raise SystemExit(f"[reward] {model_name} gives NaN embeddings even after a rebuild; "
                                     "run it with transformers 4.x or pick another --external_model")
        elif hasattr(self.model, "get_text_features"):  # CLIP, SigLIP, SigLIP2, ...
            from transformers import AutoImageProcessor, AutoTokenizer
            self.kind = "hf"
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.improc = AutoImageProcessor.from_pretrained(model_name)
            cfg = getattr(self.model.config, "text_config", self.model.config)
            self.max_len = cfg.max_position_embeddings
            # SigLIP was trained on text padded to the full length; CLIP-style models take plain padding
            self.pad = "max_length" if "siglip" in self.model.config.model_type else True
        else:
            raise SystemExit(f"[reward] {model_name}: no encode_text / get_text_features, not an image-text embedder")
        print(f"[reward] external {self.name} ({self.kind}"
              f"{f', text cut at {self.max_len} tokens' if self.kind == 'hf' else ''})", flush=True)

    def _finite(self):
        """One text and one image through the model: are the embeddings real numbers?"""
        from PIL import Image
        with torch.no_grad():
            t = self._unit(self.model.encode_text(["a photo of a cat"]))
            i = self._unit(self.model.encode_image([Image.new("RGB", (224, 224), (120, 80, 40))]))
        ok_t, ok_i = bool(torch.isfinite(t).all()), bool(torch.isfinite(i).all())
        if not (ok_t and ok_i):
            print(f"[reward] non-finite embeddings (text ok: {ok_t}, image ok: {ok_i})", flush=True)
        return ok_t and ok_i

    def _load_mock(self):
        """Tiny random CLIP on the mock Qwen tokenizer's byte vocabulary: offline smoke test only."""
        from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

        import embedder as E
        self.tok = E.load_mock(64 * 28 * 28).processor.tokenizer
        torch.manual_seed(0)
        cfg = CLIPConfig(text_config=dict(vocab_size=len(self.tok), hidden_size=32, intermediate_size=64,
                                          num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64),
                         vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=2, image_size=32, patch_size=8),
                         projection_dim=16)
        self.model = CLIPModel(cfg).to(self.device).eval()
        self.improc = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
        self.kind, self.max_len, self.pad = "hf", 64, True

    @staticmethod
    def _unit(v):
        if not torch.is_tensor(v):  # remote-code models return numpy
            v = torch.as_tensor(v)
        return torch.nn.functional.normalize(v.float(), dim=-1)

    @staticmethod
    def _features(out):
        """get_*_features returns the embedding itself, or (newer transformers) an output object holding it."""
        if torch.is_tensor(out):
            return out
        return out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]

    @torch.no_grad()
    def _img(self, image):
        if self.kind == "remote":
            return self._unit(self.model.encode_image([image]))[0].cpu()
        pv = self.improc(images=[image], return_tensors="pt")["pixel_values"].to(self.device, self.model.dtype)
        return self._unit(self._features(self.model.get_image_features(pixel_values=pv)))[0].cpu()

    @torch.no_grad()
    def _txt(self, captions):
        self.n_texts += len(captions)
        if self.kind == "remote":
            return self._unit(self.model.encode_text(list(captions))).cpu()
        self.n_truncated += sum(len(x) > self.max_len for x in self.tok(list(captions))["input_ids"])
        t = self.tok(list(captions), padding=self.pad, truncation=True, max_length=self.max_len, return_tensors="pt")
        kw = dict(input_ids=t["input_ids"].to(self.device))
        if "attention_mask" in t and self.pad is True:  # SigLIP takes no mask (its padding is part of the input)
            kw["attention_mask"] = t["attention_mask"].to(self.device)
        return self._unit(self._features(self.model.get_text_features(**kw))).cpu()

    def truncated_frac(self):
        return self.n_truncated / max(self.n_texts, 1)

    def __call__(self, key, image, captions):
        if key not in self.img_cache:
            self.img_cache[key] = self._img(image)
        return (self._txt(captions) @ self.img_cache[key]).tolist()


LONGCLIP_COMMIT = "3966af9ae9331666309a22128468b734db4672a7"  # github.com/beichenzbc/Long-CLIP


def _import_longclip(repo):
    """Long-CLIP's model/ package from a clone of its repo (it is not on pip), under its own module name so it does
    not clash with any other 'model' package. Its longclip.py imports packaging through pkg_resources, which newer
    setuptools no longer has: give it the packaging module."""
    import importlib.util
    pkg = os.path.join(repo, "model")
    if not os.path.exists(os.path.join(pkg, "longclip.py")):
        raise SystemExit(f"[reward] {pkg}/longclip.py not found: --longclip_repo is a clone of github.com/beichenzbc/"
                         f"Long-CLIP (commit {LONGCLIP_COMMIT[:7]}; bash rl_paper_commands.sh longclip_setup)")
    try:
        import pkg_resources
        if not hasattr(pkg_resources, "packaging"):
            raise ImportError
    except ImportError:
        import types

        import packaging
        import packaging.version
        mod = sys.modules.get("pkg_resources") or types.ModuleType("pkg_resources")
        mod.packaging = packaging
        sys.modules["pkg_resources"] = mod
    if "longclip_pkg" not in sys.modules:
        spec = importlib.util.spec_from_file_location("longclip_pkg", os.path.join(pkg, "__init__.py"),
                                                      submodule_search_locations=[pkg])
        m = importlib.util.module_from_spec(spec)
        sys.modules["longclip_pkg"] = m
        spec.loader.exec_module(m)
    import importlib
    return importlib.import_module("longclip_pkg.longclip")


class LongCLIPReward:
    def __init__(self, repo, ckpt, device, mock=False):
        self.device, self.img_cache = device, {}
        self.n_texts = self.n_truncated = 0
        if mock:  # offline smoke test: the tiny random CLIP of ExternalReward under this reward's interface
            self._ext = ExternalReward("mock", device, mock=True)
            self.name = "longclip:mock"
            return
        self._ext = None
        if not os.path.exists(ckpt):
            raise SystemExit(f"[reward] Long-CLIP checkpoint {ckpt} not found (BeichenZhang/LongCLIP-L, longclip-L.pt)")
        self.lc = _import_longclip(repo)
        self.model, self.preprocess = self.lc.load(ckpt, device=device)
        self.model.eval().requires_grad_(False)
        self.ctx = 248  # tokenize()'s default context_length (77 * 4 - 60)
        self.name = "longclip:" + os.path.basename(ckpt).replace(".pt", "")
        print(f"[reward] {self.name} ({ckpt}), text cut at {self.ctx} tokens", flush=True)

    @torch.no_grad()
    def _img(self, image):
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1)[0].cpu()

    @torch.no_grad()
    def _txt(self, captions):
        self.n_texts += len(captions)
        tok = getattr(self.lc, "_tokenizer", None)  # the module's SimpleTokenizer (as in OpenAI's clip.py)
        if tok is not None:
            self.n_truncated += sum(len(tok.encode(c)) + 2 > self.ctx for c in captions)  # + start / end tokens
        t = self.lc.tokenize(list(captions), truncate=True).to(self.device)
        return torch.nn.functional.normalize(self.model.encode_text(t).float(), dim=-1).cpu()

    def truncated_frac(self):
        return self.n_truncated / max(self.n_texts, 1)

    def __call__(self, key, image, captions):
        if self._ext is not None:
            return self._ext(key, image, captions)
        if key not in self.img_cache:
            self.img_cache[key] = self._img(image)
        return (self._txt(captions) @ self.img_cache[key]).tolist()


class RandomReward:
    """Uniform random reward (seeded): GRPO with advantages that carry no information about the caption."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.name = f"random(seed {seed})"

    def __call__(self, key, image, captions):
        return [self.rng.random() for _ in captions]


class LengthReward:
    """Reward = -|words - target| / target. The target follows the self-reward run's mean rollout length per step
    (its train_log.jsonl), or is one fixed number of words."""

    def __init__(self, target_log=None, target_words=None):
        self.by_step = {}
        if target_log:
            with open(target_log) as f:
                self.by_step = {r["step"]: r["words"] for r in map(json.loads, f)}
            if not self.by_step:
                raise SystemExit(f"[reward] {target_log} has no steps")
            self.last = self.by_step[max(self.by_step)]
            self.name = f"length:matched({len(self.by_step)} steps, final {self.last:.0f} words)"
        elif target_words:
            self.last = float(target_words)
            self.name = f"length:{self.last:.0f}words"
        else:
            raise SystemExit("[reward] --reward length needs --length_target_log or --length_target_words")
        self.target = self.last  # evaluation (no step) uses the final length

    def set_step(self, step):
        known = [s for s in self.by_step if s <= step]
        self.target = self.by_step[max(known)] if known else (self.by_step[min(self.by_step)] if self.by_step else self.last)

    def __call__(self, key, image, captions):
        t = max(self.target, 1.0)
        return [-abs(len(c.split()) - t) / t for c in captions]
