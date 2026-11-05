"""Rerankers (plan section 4.6), through the CPU ``encoder`` container.

- X1 ``bge``: bge-reranker-v2-m3, pointwise cross-encoder, raw logits.
  Cached per (query, unit): a score doesn't depend on the other candidates.
- X2 ``jina``: jina-reranker-v3, listwise (query and all candidates in one
  context, <= 64 per call). Cached per (query, ordered candidate list): a
  score depends on the other documents in the call, so the gate-on runs need
  their own calls.

Both get the same candidates in the same order and the document text is the
unit's ``text`` (header line + body). Latency is the encoder time of the
original call.
"""

from __future__ import annotations

from ..enrich import text_sha
from ..runs import sha


def bge_scores(encoder, cache, query: str, cands: list[tuple[str, str]]) -> tuple[list[float], float]:
    """Scores for ``[(unit_id, text)]`` in input order, and their encoder seconds."""
    qh = sha(query)
    keys = [f"{qh}|{uid}|{text_sha(t)}" for uid, t in cands]
    miss = [(k, t) for k, (_, t) in zip(keys, cands) if k not in cache]
    if miss:
        scores, secs = encoder.rerank("bge", query, [t for _, t in miss])
        per = secs / len(miss)
        for (k, _), s in zip(miss, scores):
            cache.put(k, {"score": float(s), "seconds": per})
    vals = [cache.get(k) for k in keys]
    return [v["score"] for v in vals], sum(v["seconds"] for v in vals)


def jina_scores(encoder, cache, query: str, cands: list[tuple[str, str]]) -> tuple[list[float], float]:
    if len(cands) > 64:
        raise ValueError("jina-reranker-v3 takes at most 64 documents per call")
    key = f"{sha(query)}|{sha([(u, text_sha(t)) for u, t in cands])}"
    hit = cache.get(key)
    if hit is None:
        scores, secs = encoder.rerank("jina", query, [t for _, t in cands])
        hit = {"scores": [float(s) for s in scores], "seconds": secs, "unit_ids": [u for u, _ in cands]}
        cache.put(key, hit)
    return hit["scores"], hit["seconds"]
