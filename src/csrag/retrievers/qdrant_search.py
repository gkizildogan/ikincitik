"""Qdrant scores, the notebook fuses (plan section 4).

Each signal's raw top-50 per (index, query_hash, signal, filter) is cached in
``data/cache/search.jsonl`` with its scores and the Qdrant latency. The gate,
fusion and the rerankers all work from these lists.

Signals: ``bm25`` (fastembed query, IDF in Qdrant), ``bge_sparse``, ``dense``,
``dense_exact`` (same query with ``exact=True``, the ANN check) and ``colbert``
(MaxSim brute force).

ColBERT engine. Qdrant caps a multivector at 1023 vectors of 1024 dimensions,
and 60 of the 252 base units are longer. With ``colbert_engine="client"`` (the
default) MaxSim is computed here, exactly, over each unit's full cached ColBERT
matrix (``ColbertIndex``): score = sum over query tokens of the max dot product
with the unit's tokens, the same as Qdrant's MAX_SIM with DOT. ``"qdrant"``
queries Qdrant instead (only valid where every point holds its vectors, e.g.
the windows index).
"""

from __future__ import annotations

import time

import numpy as np

from ..corpus import COLLECTIONS
from ..enrich import point_id
from ..runs import canonical, sha


class Query:
    """One query in every form a signal can use. ``key`` identifies its content
    (it is hashed into the cache key); vectors are filled lazily by ``Searcher``."""

    def __init__(self, key: str, text: str, bm25_text: str | None = None, dense: np.ndarray | None = None,
                 bge_sparse=None, colbert: np.ndarray | None = None, encoder_seconds: float = 0.0):
        self.key = key
        self.text = text
        self.bm25_text = bm25_text if bm25_text is not None else text
        self.dense = dense
        self.bge_sparse = bge_sparse
        self.colbert = colbert
        self.encoder_seconds = encoder_seconds

    @property
    def hash(self) -> str:
        return sha(self.key)


def engine_filter(engine: str | None) -> dict | None:
    """``engine_type in {engine, all}``; ``None`` / ``all`` means no filter."""
    if engine in (None, "all"):
        return None
    return {"engine_type": sorted([engine, "all"])}


def _qdrant_filter(flt: dict | None, ids: list[str] | None = None):
    from qdrant_client import models as m

    must = []
    if flt:
        for key, vals in flt.items():
            must.append(m.FieldCondition(key=key, match=m.MatchAny(any=list(vals))))
    if ids is not None:
        must.append(m.HasIdCondition(has_id=[point_id(u) for u in ids]))
    return m.Filter(must=must) if must else None


class ColbertIndex:
    """Exact client-side ColBERT MaxSim over one index's cached vectors (float16 in memory)."""

    def __init__(self, store, units: dict[str, dict], chunk_tokens: int = 32768):
        self.ids = [u for u in units if u in store.rows]
        self.missing = [u for u in units if u not in store.rows]  # must be empty after notebook 04
        mats = [store.rows[u]["colbert"] for u in self.ids]
        self.off = np.zeros(len(mats) + 1, np.int64)
        for i, mm in enumerate(mats):
            self.off[i + 1] = self.off[i] + len(mm)
        self.D = np.concatenate(mats).astype(np.float16)
        self.engine = {u: units[u]["engine_type"] for u in self.ids}
        self.pos = {u: i for i, u in enumerate(self.ids)}
        self.chunk_tokens = chunk_tokens

    def scores(self, q: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """MaxSim score of every unit (or of ``rows``) for query matrix ``q`` (n_q, 1024)."""
        q = np.asarray(q, np.float32)
        rows = np.arange(len(self.ids)) if rows is None else np.asarray(rows)
        out = np.empty(len(rows), np.float64)
        for j, r in enumerate(rows):
            a, b = self.off[r], self.off[r + 1]
            sims = q @ self.D[a:b].astype(np.float32).T
            out[j] = float(sims.max(axis=1).sum())
        return out

    def search(self, q: np.ndarray, flt: dict | None, limit: int, ids: list[str] | None = None) -> list[tuple[str, float]]:
        cand = [self.pos[u] for u in (ids if ids is not None else self.ids) if u in self.pos]
        if flt:
            allowed = set(flt.get("engine_type", []))
            cand = [r for r in cand if self.engine[self.ids[r]] in allowed]
        s = self.scores(q, np.asarray(cand, np.int64))
        order = sorted(range(len(cand)), key=lambda i: (-s[i], self.ids[cand[i]]))[:limit]
        return [(self.ids[cand[i]], float(s[i])) for i in order]


class Searcher:
    def __init__(self, client, cache, bm25, colbert_index=None, colbert_engine: str = "client"):
        self.client = client
        self.cache = cache  # JsonlCache (search.jsonl)
        self.bm25 = bm25  # embed.BM25, used for query vectors only
        self.colbert_index = colbert_index  # index -> ColbertIndex (client engine)
        self.colbert_engine = colbert_engine

    def _vector(self, q: Query, signal: str):
        from qdrant_client import models as m

        if signal == "bm25":
            s = self.bm25.embed_query(q.bm25_text)
            return m.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
        if signal == "bge_sparse":
            return m.SparseVector(indices=q.bge_sparse.indices.tolist(), values=q.bge_sparse.values.tolist())
        if signal in ("dense", "dense_exact"):
            return np.asarray(q.dense, np.float32).tolist()
        if signal.startswith("colbert"):
            return np.asarray(q.colbert, np.float32).tolist()
        raise ValueError(signal)

    def search(self, index: str, signal: str, q: Query, flt: dict | None = None, limit: int = 50,
               ids: list[str] | None = None) -> dict:
        """``{"hits": [[unit_id, score], ...], "seconds": s, "cached": bool}``, best first.

        With ``ids`` the search is restricted to those units (ColBERT rerank of M1).
        """
        from qdrant_client import models as m

        client_colbert = signal == "colbert" and self.colbert_engine == "client"
        sig_key = ("colbert[client]" if client_colbert else signal) + ("" if ids is None else f"@{sha(sorted(ids))}")
        query_hash = sha(q.bm25_text) if signal == "bm25" else q.hash
        key = canonical([index, query_hash, sig_key, flt, limit])
        hit = self.cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        if client_colbert:
            t0 = time.time()
            hits = self.colbert_index(index).search(q.colbert, flt, limit if ids is None else len(ids), ids)
            out = {"hits": [[u, s] for u, s in hits], "seconds": round(time.time() - t0, 4)}
            self.cache.put(key, out)
            return {**out, "cached": False}
        using = "dense" if signal == "dense_exact" else signal
        params = m.SearchParams(exact=True) if signal in ("dense_exact", "colbert") else None
        t0 = time.time()
        res = self.client.query_points(
            COLLECTIONS[index],
            query=self._vector(q, signal),
            using=using,
            query_filter=_qdrant_filter(flt, ids),
            search_params=params,
            limit=limit if ids is None else len(ids),
            with_payload=["unit_id"],
        ).points
        out = {"hits": [[p.payload["unit_id"], float(p.score)] for p in res], "seconds": round(time.time() - t0, 4)}
        self.cache.put(key, out)
        return {**out, "cached": False}
