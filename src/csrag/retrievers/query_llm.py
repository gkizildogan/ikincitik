"""Query-side LLM methods (plan section 4.5) and the F1 filter extraction (4.4).

Qwen3.8-27B through vLLM, thinking off, temperature 0. Every call is cached in
``data/cache/llm.jsonl`` under the hash of the full request (model, messages,
parameters), with its output, token usage and latency. The query-side outputs
don't depend on the index, so both indexes share them.
"""

from __future__ import annotations

import json
import os
import time

from ..runs import sha

VLLM_MODEL_DEFAULT = "RedHatAI/Qwen3.8-27B-INT4"

HYDE_SYSTEM = (
    "You write passages in the style of EASA CS-E (Certification Specifications for Engines, "
    "Amendment 9) and its AMC (Acceptable Means of Compliance)."
)
HYDE_USER = (
    "Write one passage of about 120 to 200 words, in the style of a CS-E specification or an AMC "
    "paragraph, that would answer the question below. Use the regulation's own vocabulary, numbering "
    "style and units. Write only the passage: no title, no preamble, and do not say that it is "
    "hypothetical.\n\nQuestion: {question}"
)

KEYWORDS_SYSTEM = "You help a keyword search engine find passages in EASA CS-E Amendment 9 (engine certification)."
KEYWORDS_USER = (
    "List up to 10 terms or short phrases that the CS-E regulation itself would likely use for the "
    "concepts in this question: regulatory terminology, synonyms, abbreviations and spelling variants "
    "(for example \"flame out\" -> \"flame-out\", \"relight\"). Do not repeat words of the question in "
    "the same form. Return JSON.\n\nQuestion: {question}"
)
KEYWORDS_SCHEMA = {
    "type": "object",
    "properties": {"terms": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
    "required": ["terms"],
    "additionalProperties": False,
}

FILTER_SYSTEM = "You classify questions about EASA CS-E Amendment 9 (engine certification specifications)."
FILTER_USER = (
    "Which engine type is this question specifically about?\n"
    "- \"piston\": only piston (reciprocating) engines;\n"
    "- \"turbine\": only turbine engines (turbojet, turbofan, turboprop, turboshaft);\n"
    "- null: all engines, both types, or the type is not clear from the question.\n"
    "Return JSON.\n\nQuestion: {question}"
)
FILTER_SCHEMA = {
    "type": "object",
    "properties": {"engine_type": {"anyOf": [{"type": "string", "enum": ["piston", "turbine"]}, {"type": "null"}]}},
    "required": ["engine_type"],
    "additionalProperties": False,
}


def prompt_hashes() -> dict:
    """Hashes of every prompt template; stamped into the run configs."""
    from . import agentic

    return {
        "hyde": sha([HYDE_SYSTEM, HYDE_USER], 8),
        "keywords": sha([KEYWORDS_SYSTEM, KEYWORDS_USER, KEYWORDS_SCHEMA], 8),
        "filter": sha([FILTER_SYSTEM, FILTER_USER, FILTER_SCHEMA], 8),
        "agent": sha([agentic.AGENT_SYSTEM, agentic.AGENT_USER], 8),
    }


class LLM:
    """OpenAI-compatible client for the vLLM container, with a request-hash cache."""

    def __init__(self, cache, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 900.0):
        self.cache = cache
        self.base_url = base_url or os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.model = model or os.environ.get("VLLM_MODEL", VLLM_MODEL_DEFAULT)
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    def chat(self, messages: list[dict], max_tokens: int = 512, schema: dict | None = None, name: str = "out") -> dict:
        """``{"content", "usage": {"prompt_tokens", "completion_tokens"}, "seconds", "finish_reason", "cached"}``."""
        req = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            req["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
        key = sha(req, 32)
        hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        kwargs = {k: v for k, v in req.items() if k not in ("chat_template_kwargs",)}
        t0 = time.time()
        r = self.client.chat.completions.create(**kwargs, extra_body={"chat_template_kwargs": req["chat_template_kwargs"]})
        out = {
            "content": r.choices[0].message.content or "",
            "usage": {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens},
            "seconds": round(time.time() - t0, 3),
            "finish_reason": r.choices[0].finish_reason,
        }
        self.cache.put(key, out)
        return {**out, "cached": False}

    # ------------------------------------------------------------ the three query-side calls

    def hyde(self, question: str) -> tuple[str, dict]:
        r = self.chat(
            [{"role": "system", "content": HYDE_SYSTEM}, {"role": "user", "content": HYDE_USER.format(question=question)}],
            max_tokens=400,
        )
        return r["content"].strip(), r

    def keywords(self, question: str) -> tuple[list[str], dict]:
        r = self.chat(
            [{"role": "system", "content": KEYWORDS_SYSTEM}, {"role": "user", "content": KEYWORDS_USER.format(question=question)}],
            max_tokens=200,
            schema=KEYWORDS_SCHEMA,
            name="keywords",
        )
        try:
            terms = [t.strip() for t in json.loads(r["content"])["terms"] if t and t.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            terms = []
        return terms[:10], r

    def extract_filter(self, question: str) -> tuple[str | None, dict]:
        r = self.chat(
            [{"role": "system", "content": FILTER_SYSTEM}, {"role": "user", "content": FILTER_USER.format(question=question)}],
            max_tokens=30,
            schema=FILTER_SCHEMA,
            name="filter",
        )
        try:
            v = json.loads(r["content"])["engine_type"]
        except (json.JSONDecodeError, KeyError, TypeError):
            v = None
        return (v if v in ("piston", "turbine") else None), r


def vllm_healthy(base_url: str | None = None, timeout: float = 3.0) -> bool:
    """Is the vLLM server answering /health? (The GPU phases must not overlap.)"""
    import urllib.request

    base = (base_url or os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")).rsplit("/v1", 1)[0]
    try:
        urllib.request.urlopen(base + "/health", timeout=timeout)
        return True
    except Exception:
        return False


def usage_of(rs: list[dict]) -> dict:
    """Summed tokens and latency of LLM results (cached results report their original cost)."""
    return {
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rs),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rs),
        "seconds": round(sum(r["seconds"] for r in rs), 3),
        "calls": len(rs),
    }
