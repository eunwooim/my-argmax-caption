"""Chat backends: vLLM in-process, any OpenAI-compatible server, or a mock function.

Every backend takes a list of conversations (OpenAI-style message lists, images as data URLs)
and returns one reply string per conversation, in order.
"""
import concurrent.futures as cf
import os


def add_llm_args(ap, default_model, default_max_tokens):
    g = ap.add_argument_group("model")
    g.add_argument("--backend", default="vllm", choices=["vllm", "openai", "mock"],
                   help="vllm = load the model in this process; openai = call a running server "
                        "(vllm serve, SGLang, Ollama, ...); mock = no model, for smoke tests")
    g.add_argument("--model", default=default_model)
    g.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    g.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    g.add_argument("--concurrency", type=int, default=32, help="parallel requests for --backend openai")
    g.add_argument("--max_model_len", type=int, default=8192)
    g.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    g.add_argument("--dtype", default="auto")
    g.add_argument("--max_tokens", type=int, default=None,
                   help=f"default {default_max_tokens} (x2 for gpt-oss, whose reasoning also counts)")
    g.add_argument("--temperature", type=float, default=None, help="default: the model card's recommended value")
    g.add_argument("--top_p", type=float, default=None)
    g.add_argument("--top_k", type=int, default=None)
    g.add_argument("--repetition_penalty", type=float, default=None, help="default: none (1.0)")
    g.add_argument("--thinking", action="store_true",
                   help="Qwen3 thinking mode (slower; turns off JSON-schema constrained decoding)")
    g.add_argument("--reasoning_effort", default="low", choices=["low", "medium", "high"],
                   help="gpt-oss only. It always reasons; low is enough for writing questions")
    g.add_argument("--no_json_schema", action="store_true", help="do not constrain output to the JSON schema")
    g.add_argument("--seed", type=int, default=0)
    return g


def is_gpt_oss(a):
    return "gpt-oss" in a.model.lower()


def reasons(a):
    """Model writes reasoning before the answer, so the reply cannot be JSON-constrained from token 1."""
    return a.thinking or is_gpt_oss(a)


def resolve_defaults(a, default_max_tokens):
    if a.max_tokens is None:
        a.max_tokens = default_max_tokens * (2 if is_gpt_oss(a) else 1)


def sampling_defaults(a):
    """(temperature, top_p, top_k) from the model cards unless set on the command line.

    gpt-oss: T=1.0 top_p=1.0, no top_k. Qwen3: thinking T=0.6 top_p=0.95, non-thinking T=0.7
    top_p=0.8, top_k=20 for both.
    """
    if is_gpt_oss(a):
        t, p, k = 1.0, 1.0, -1
    else:
        t, p, k = (0.6, 0.95, 20) if a.thinking else (0.7, 0.8, 20)
    pick = lambda x, d: d if x is None else x
    return pick(a.temperature, t), pick(a.top_p, p), pick(a.top_k, k)


def template_kwargs(a):
    if is_gpt_oss(a):
        return {"reasoning_effort": a.reasoning_effort}
    return {"enable_thinking": a.thinking}


class VLLMChat:
    def __init__(self, a, n_gpus=1, limit_images=0):
        from vllm import LLM
        kw = dict(model=a.model, tensor_parallel_size=n_gpus, max_model_len=a.max_model_len,
                  gpu_memory_utilization=a.gpu_memory_utilization, dtype=a.dtype, seed=a.seed)
        if limit_images:
            kw["limit_mm_per_prompt"] = {"image": limit_images}
        self.llm = LLM(**kw)
        self.a = a

    def _params(self, schema, temperature):
        from vllm import SamplingParams
        t, p, k = sampling_defaults(self.a)
        kw = dict(temperature=t if temperature is None else temperature, top_p=p, top_k=k,
                  max_tokens=self.a.max_tokens, seed=self.a.seed)
        if self.a.repetition_penalty:
            kw["repetition_penalty"] = self.a.repetition_penalty
        if is_gpt_oss(self.a):  # keep the harmony channel markers so strip_think can find the final answer
            kw["skip_special_tokens"] = False
        if schema is None or reasons(self.a) or self.a.no_json_schema:
            return SamplingParams(**kw)
        try:  # vLLM >= 0.10.2
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema), **kw)
        except (ImportError, TypeError):
            pass
        try:  # older vLLM
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **kw)
        except (ImportError, TypeError):
            print("[vllm] constrained decoding not available, relying on the prompt", flush=True)
            return SamplingParams(**kw)

    def __call__(self, convs, schema=None, temperature=None):
        outs = self.llm.chat(convs, self._params(schema, temperature), use_tqdm=True,
                             chat_template_kwargs=template_kwargs(self.a))
        return [o.outputs[0].text for o in outs]


class OpenAIChat:
    def __init__(self, a):
        from openai import OpenAI
        self.client = OpenAI(base_url=a.base_url, api_key=a.api_key)
        self.a = a

    def _one(self, msgs, schema, temperature):
        t, p, k = sampling_defaults(self.a)
        extra = {"top_k": k, "chat_template_kwargs": template_kwargs(self.a)}
        if self.a.repetition_penalty:
            extra["repetition_penalty"] = self.a.repetition_penalty
        if is_gpt_oss(self.a):  # vllm serve puts only the final channel in message.content
            extra["reasoning_effort"] = self.a.reasoning_effort
        kw = dict(model=self.a.model, messages=msgs, max_tokens=self.a.max_tokens,
                  temperature=t if temperature is None else temperature, top_p=p, seed=self.a.seed,
                  extra_body=extra)
        if schema is not None and not reasons(self.a) and not self.a.no_json_schema:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "out", "schema": schema}}
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:  # network hiccup or server overload: retry, then give up on this item
                if attempt == 2:
                    print(f"[openai] request failed: {e}", flush=True)
                    return ""

    def __call__(self, convs, schema=None, temperature=None):
        with cf.ThreadPoolExecutor(self.a.concurrency) as ex:
            return list(ex.map(lambda m: self._one(m, schema, temperature), convs))


class MockChat:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, convs, schema=None, temperature=None):
        return [self.fn(m) for m in convs]


class LazyChat:
    """Loads the model on the first call, so a rerun with nothing left to do never touches a GPU."""

    def __init__(self, make):
        self.make, self.chat = make, None

    def __call__(self, convs, schema=None, temperature=None):
        if self.chat is None:
            self.chat = self.make()
        return self.chat(convs, schema=schema, temperature=temperature)


def make_backend(a, mock_fn=None, limit_images=0):
    if a.backend == "mock":
        return MockChat(mock_fn)
    if a.backend == "openai":
        return OpenAIChat(a)
    return VLLMChat(a, n_gpus=a.n_gpus, limit_images=limit_images)
