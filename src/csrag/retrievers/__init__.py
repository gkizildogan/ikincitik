"""The experiment matrix (plan section 4): one ``Pipeline`` interprets any method config.

    res = Resources.connect(corpus)                       # Qdrant, encoder, caches, LLM
    cfg = method_config("X1", "units", gate=True)
    p = Pipeline(cfg, res)
    p.retrieve("q001", "What is ...?")                    # -> list[Hit]
    runs.execute(cfg, p.run, questions, res.meta(cfg["index"]))

Stages: query_transform -> first_stage(s) -> gate A -> fusion -> filter (inside each
search) -> ColBERT rerank / rerank -> gate B -> lookup -> expansion.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..corpus import UNIT_FILES, Corpus
from ..embed import BM25, EncoderClient, QueryEmbeddings
from ..enrich import corpus_version
from ..gate import GateParams, apply_gate
from ..runs import Caches
from .agentic import Agent
from .base import (ALL_METHODS, DESCRIPTIONS, EXPANSION_METHODS, PARAMS, RANKED_METHODS, Hit, get_params, grid,
                   method_config, revise)
from .fusion import qdrant_rrf, rrf
from .qdrant_search import ColbertIndex, Query, Searcher, engine_filter
from .query_llm import LLM, prompt_hashes, usage_of
from .rerank import bge_scores, jina_scores
from .structural import expand_fixed, id_lookup, lookup_then

__all__ = [
    "ALL_METHODS", "DESCRIPTIONS", "EXPANSION_METHODS", "PARAMS", "RANKED_METHODS", "Hit", "Pipeline", "Resources",
    "get_params", "grid", "method_config", "prompt_hashes", "revise", "rrf",
]


class Resources:
    """Shared handles for every pipeline: corpus, Qdrant, encoder, caches, query
    embeddings, the BM25 query encoder and the LLM client."""

    def __init__(self, corpus: Corpus, client, encoder: EncoderClient | None, caches: Caches, llm: LLM | None,
                 data_dir: str | Path = "data"):
        self.corpus = corpus
        self.client = client
        self.encoder = encoder
        self.caches = caches
        self.llm = llm
        self.data_dir = Path(data_dir)
        self.bm25 = BM25()  # query side only: weights are 1 per stem, avg_len does not matter
        self.qemb = {n: QueryEmbeddings(n, self.data_dir / "embeddings") for n in ("questions", "hyde", "agent")}
        self._meta: dict = {}
        self._colbert: dict[str, ColbertIndex] = {}

    def colbert(self, index: str) -> ColbertIndex:
        """Client-side ColBERT index of one collection, loaded from data/embeddings on first use."""
        if index not in self._colbert:
            from ..corpus import COLLECTIONS
            from ..embed import EmbeddingStore

            store = EmbeddingStore(COLLECTIONS[index], self.data_dir / "embeddings")
            self._colbert[index] = ColbertIndex(store, self.corpus.units(index))
        return self._colbert[index]

    @classmethod
    def connect(cls, corpus: Corpus, data_dir: str | Path = "data", with_llm: bool = True, with_encoder: bool = True):
        from ..index import connect

        caches = Caches(Path(data_dir) / "cache")
        llm = LLM(caches.llm) if with_llm else None
        return cls(corpus, connect(), EncoderClient() if with_encoder else None, caches, llm, data_dir)

    def meta(self, index: str) -> dict:
        """corpus_version and model revisions, stamped into every run's config.json."""
        if index not in self._meta:
            models: dict = {"llm": {"model": self.llm.model if self.llm else os.environ.get("VLLM_MODEL"),
                                    "serve_config": os.environ.get("VLLM_CONFIG", "serve.retrieval.yaml")}}
            if self.encoder is not None:
                try:
                    models["encoder"] = self.encoder.model_revisions()
                except Exception as e:  # encoder down: runs from cached lists still work
                    models["encoder"] = {"error": f"{type(e).__name__}: {e}"}
            try:
                models["qdrant"] = self.client.info().version
            except Exception:
                pass
            models["bm25"] = self.bm25.params() | {"note": "query side; avg_len lives in the stored document vectors"}
            self._meta[index] = {"corpus_version": corpus_version(self.data_dir / UNIT_FILES[index]), "models": models}
        return self._meta[index]


def _uniq_sections(units: dict, ids) -> list[str]:
    return list(dict.fromkeys(units[u]["section_id"] for u in ids))


class Pipeline:
    def __init__(self, cfg: dict, res: Resources, labels: dict[str, str] | None = None):
        self.cfg = cfg
        self.res = res
        self.index = cfg["index"]
        self.corpus = res.corpus
        self.units = res.corpus.units(self.index)
        self.k = cfg["k"]
        self.gate_a = GateParams(**cfg["gate_a"]) if cfg["gate_a"] else None
        self.gate_b = GateParams(**cfg["gate_b"]) if cfg["gate_b"] else None
        flt = cfg.get("filter") or {}
        # Only the oracle F0 reads a label; every other method never sees one.
        self.labels = labels if flt.get("type") == "oracle" else None
        if flt.get("type") == "oracle" and labels is None:
            raise ValueError("F0 (oracle filter) needs the engine_type labels")
        self.searcher = Searcher(res.client, res.caches.search, res.bm25, res.colbert, cfg.get("colbert_engine") or "client")

    # ------------------------------------------------------------ public interface

    def retrieve(self, question_id: str, text: str, k: int | None = None) -> list[Hit]:
        hits, _, _ = self.run(question_id, text)
        return hits[:k] if k else hits

    # ------------------------------------------------------------ helpers

    def _query(self, text: str, store: str = "questions") -> Query:
        e = self.res.qemb[store].get([text], self.res.encoder)[0]
        return Query("q:" + text, text, dense=e["dense"], bge_sparse=e["bge_sparse"], colbert=e["colbert"],
                     encoder_seconds=e["seconds"])

    def _first_stages(self, signals: list[str], q: Query, flt: dict | None, stats: dict) -> list[list[tuple[str, float]]]:
        lists = []
        for sig in signals:
            r = self.searcher.search(self.index, sig, q, flt, self.cfg["candidates"])
            stats["latency"]["qdrant"] += r["seconds"]
            raw = [(u, s) for u, s in r["hits"]]
            kept, tr = apply_gate(raw, self.gate_a)
            if self.gate_a is not None:
                ks = _uniq_sections(self.units, [u for u, _ in kept])
                stats["gates"].append({
                    "pos": "A", "signal": sig, **tr, "kept_sections": ks,
                    "cut_sections": [s for s in _uniq_sections(self.units, [u for u, _ in raw[len(kept):]]) if s not in ks],
                })
            lists.append(kept)
        return lists

    def h1_pool(self, q: Query, flt: dict | None, stats: dict) -> list[tuple[str, float]]:
        """H1: BM25 + dense, each top-50 -> gate A (if on) -> RRF. The whole fused list."""
        return rrf(self._first_stages(["bm25", "dense"], q, flt, stats), self.cfg["fusion"]["k"] if self.cfg["fusion"] else 60)

    def _gate_b(self, reranked: list[tuple[str, float]], stats: dict) -> list[tuple[str, float]]:
        kept, tr = apply_gate(reranked, self.gate_b)
        if self.gate_b is not None:
            ks = _uniq_sections(self.units, [u for u, _ in kept])
            stats["gates"].append({
                "pos": "B", "signal": self.cfg["method"], **tr, "kept_sections": ks,
                "cut_sections": [s for s in _uniq_sections(self.units, [u for u, _ in reranked[len(kept):]]) if s not in ks],
            })
        return kept[: self.k]

    def _hits(self, ranked: list[tuple[str, float]], stage: str) -> list[Hit]:
        return [Hit(u, self.units[u]["section_id"], s, stage, 0, {}, self.units[u]["n_tokens_bge"]) for u, s in ranked]

    # ------------------------------------------------------------ one question

    def run(self, question_id: str, text: str) -> tuple[list[Hit], dict, list[dict]]:
        cfg, res = self.cfg, self.res
        stats: dict = {"latency": {"qdrant": 0.0, "encoder": 0.0, "llm": 0.0, "rerank": 0.0}, "gates": []}
        llm_rs: list[dict] = []
        trace: list[dict] = []

        q = self._query(text)
        stats["latency"]["encoder"] += q.encoder_seconds

        # query transform (Q1, Q2)
        qt = cfg.get("query_transform")
        if qt and qt["type"] == "hyde":
            passage, r = res.llm.hyde(text)
            llm_rs.append(r)
            pe = res.qemb["hyde"].get([passage], res.encoder)[0]
            stats["latency"]["encoder"] += pe["seconds"]
            v = np.asarray(q.dense, np.float64) + np.asarray(pe["dense"], np.float64)
            v = (v / np.linalg.norm(v)).astype(np.float32)
            q = Query(f"hyde:{text}\x00{passage}", text, dense=v, bge_sparse=q.bge_sparse, colbert=q.colbert)
            stats["query_transform"] = {"type": "hyde", "passage": passage}
        elif qt and qt["type"] == "keywords":
            terms, r = res.llm.keywords(text)
            llm_rs.append(r)
            q.bm25_text = (text + " " + " ".join(terms)).strip()
            stats["query_transform"] = {"type": "keywords", "terms": terms}

        # filter (F0 oracle, F1 LLM): applied inside each signal's search, before gate A
        flt = None
        if cfg.get("filter"):
            if cfg["filter"]["type"] == "oracle":
                engine = self.labels.get(question_id)
            else:
                engine, r = res.llm.extract_filter(text)
                llm_rs.append(r)
            flt = engine_filter(engine)
            stats["filter"] = {"engine_type": engine, "qdrant_filter": flt}

        # first stage(s) -> gate A -> fusion
        lists = self._first_stages(cfg["first_stages"], q, flt, stats)
        if cfg.get("fusion"):
            pool, stage = rrf(lists, cfg["fusion"]["k"]), "fused"
        else:
            pool, stage = lists[0], ("gated" if self.gate_a else "first")

        if cfg.get("colbert_rerank"):  # M1
            ids = [u for u, _ in pool]
            r = self.searcher.search(self.index, "colbert", q, flt, ids=ids)
            stats["latency"]["qdrant"] += r["seconds"]
            hits = self._hits(self._gate_b([(u, s) for u, s in r["hits"]], stats), "reranked")
        elif cfg.get("rerank"):  # X1, X2
            cands = pool[: cfg["rerank"]["input"]]
            docs = [(u, self.units[u]["text"]) for u, _ in cands]
            fn = bge_scores if cfg["rerank"]["model"] == "bge" else jina_scores
            cache = res.caches.rerank_bge if cfg["rerank"]["model"] == "bge" else res.caches.rerank_jina
            scores, secs = fn(res.encoder, cache, text, docs)
            stats["latency"]["rerank"] += secs
            order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
            hits = self._hits(self._gate_b([(cands[i][0], scores[i]) for i in order], stats), "reranked")
        elif cfg.get("lookup"):  # T1
            ids = id_lookup(text, self.corpus)
            stats["lookup"] = ids
            hits = lookup_then(ids, pool, self.corpus, self.index, self.k)
        elif cfg.get("expansion"):  # T0-T4
            ex = cfg["expansion"]
            if ex["rule"] == "agent":
                def search_fn(query: str):
                    st = {"latency": {"qdrant": 0.0, "encoder": 0.0, "llm": 0.0, "rerank": 0.0}, "gates": []}
                    q2 = self._query(query, store="agent")
                    st["latency"]["encoder"] += q2.encoder_seconds
                    return self.h1_pool(q2, None, st), st

                agent = Agent(self.corpus, self.index, res.llm, ex["budget"], ex["step_limit"], ex["initial_hits"],
                              ex["search_top"], search_fn)
                out = agent.run(text, pool)
                hits = out["hits"]
                llm_rs += out["llm_results"]
                trace = out["trace"]
                for st in out["search_stats"]:
                    for kk, vv in st["latency"].items():
                        stats["latency"][kk] += vv
                stats["agent"] = {"path": out["path"], "forced_stop": out["forced_stop"],
                                  "selected_tokens": out["selected_tokens"], "searches": len(out["search_stats"])}
            else:
                ctx = expand_fixed(ex["rule"], pool, self.corpus, self.index, ex["budget"], ex.get("to", "section"))
                hits = ctx.hits()
        else:
            hits = self._hits(pool[: self.k], stage)

        u = usage_of(llm_rs)
        stats["latency"]["llm"] += u["seconds"]
        stats["latency"]["total"] = round(sum(stats["latency"].values()), 4)
        stats["llm"] = {k: u[k] for k in ("calls", "prompt_tokens", "completion_tokens")}
        in_ctx = [h.unit_id for h in hits if h.in_context]
        stats["context_tokens"] = self.corpus.context_tokens(in_ctx, self.index) if in_ctx else 0
        stats["n_hits"] = len(hits)
        return hits, stats, trace


def check_rrf(p: Pipeline, text: str, flt: dict | None = None) -> dict:
    """Client-side RRF on the gate-off lists vs Qdrant's server-side RRF with k = 61 (plan 8)."""
    q = p._query(text)
    raw = []
    for sig in ("bm25", "dense"):
        r = p.searcher.search(p.index, sig, q, flt, 50)
        raw.append([(u, s) for u, s in r["hits"]])
    mine = rrf(raw, 60)
    bm = p.res.bm25.embed_query(q.bm25_text)
    from qdrant_client import models as m

    server = qdrant_rrf(p.res.client, p.index,
                        [("bm25", m.SparseVector(indices=bm.indices.tolist(), values=bm.values.tolist())),
                         ("dense", np.asarray(q.dense, np.float32).tolist())], limit=50, k=61, out_limit=len(mine))
    same_scores = len(mine) == len(server) and all(abs(a[1] - b[1]) < 1e-6 for a, b in zip(sorted(mine), sorted(server)))
    # Qdrant computes in float32: equal ranks give equal scores within 1e-6, so compare the score sequences
    same_order = len(mine) == len(server) and all(abs(a[1] - b[1]) < 1e-6 for a, b in zip(mine, server))
    return {"n": len(mine), "same_set": {u for u, _ in mine} == {u for u, _ in server}, "same_scores": same_scores,
            "same_order_ignoring_ties": same_order}
