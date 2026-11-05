"""Encoder client, BM25 and embedding caches (plan section 3.4).

- ``EncoderClient``: the CPU ``encoder`` container (``/embed``, ``/rerank``, ``/health``).
- ``BM25``: fastembed ``Qdrant/bm25`` in this process (no neural model). Qdrant
  applies the IDF per collection (``modifier=IDF``); ``avg_len`` is set per index.
- ``EmbeddingStore``: corpus vectors in ``data/embeddings/<collection>.{npz,jsonl}``,
  keyed by unit id + text hash, reused on every re-ingestion.
- ``QueryEmbeddings``: query vectors in ``data/embeddings/queries_<name>.npz``,
  keyed by text hash and shared by both indexes.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .enrich import text_sha

ENCODER_URL = os.environ.get("ENCODER_URL", "http://127.0.0.1:8081")
FASTEMBED_CACHE = os.environ.get("FASTEMBED_CACHE", ".cache/fastembed")
BM25_MODEL = "Qdrant/bm25"


# ---------------------------------------------------------------- encoder client


def _decode(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"]).copy()


@dataclass
class Sparse:
    indices: np.ndarray  # int64
    values: np.ndarray  # float32

    def as_dict(self) -> dict:
        return {"indices": self.indices.tolist(), "values": self.values.tolist()}


class EncoderClient:
    def __init__(self, base_url: str | None = None, timeout: float = 3600.0):
        import httpx

        self.base_url = (base_url or ENCODER_URL).rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict:
        return self.http.get("/health").raise_for_status().json()

    def wait_ready(self, timeout_s: float = 600) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                h = self.health()
                if h.get("status") == "ok":
                    return h
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError(f"encoder at {self.base_url} not ready after {timeout_s}s (docker compose logs encoder)")

    def model_revisions(self) -> dict:
        h = self.health()
        return {
            "models": {k: {"repo": v["repo"], "revision": v["revision"]} for k, v in h["models"].items()},
            "versions": h["versions"],
            "limits": h["limits"],
        }

    def embed(self, texts: list[str], dense: bool = True, sparse: bool = True, colbert: bool = True, batch_size: int = 4) -> dict:
        """bge-m3 outputs of one forward pass per batch: ``dense`` (n, 1024) float32,
        ``sparse`` [Sparse], ``colbert`` [(n_tok, 1024) float16], ``seconds``."""
        r = self.http.post(
            "/embed", json={"texts": texts, "dense": dense, "sparse": sparse, "colbert": colbert, "batch_size": batch_size}
        )
        r.raise_for_status()
        d = r.json()
        out: dict = {"seconds": d["seconds"]}
        if dense:
            out["dense"] = _decode(d["dense"])
        if sparse:
            out["sparse"] = [Sparse(np.asarray(s["indices"], np.int64), np.asarray(s["values"], np.float32)) for s in d["sparse"]]
        if colbert:
            out["colbert"] = [_decode(c) for c in d["colbert"]]
        return out

    def rerank(self, model: str, query: str, documents: list[str], batch_size: int = 4) -> tuple[list[float], float]:
        r = self.http.post("/rerank", json={"model": model, "query": query, "documents": documents, "batch_size": batch_size})
        r.raise_for_status()
        d = r.json()
        return d["scores"], d["seconds"]

    def tokens(self, model: str, query: str, documents: list[str]) -> dict:
        return self.http.post("/tokens", json={"model": model, "query": query, "documents": documents}).raise_for_status().json()


# ---------------------------------------------------------------- BM25 (fastembed)


class BM25:
    """fastembed ``Qdrant/bm25``: documents get BM25-saturated term frequencies,
    queries get weight 1 per distinct stem; Qdrant multiplies in the IDF."""

    def __init__(self, avg_len: float = 256.0, k1: float = 1.2, b: float = 0.75, cache_dir: str | None = None):
        from fastembed.sparse.bm25 import Bm25

        self.avg_len, self.k1, self.b = float(avg_len), float(k1), float(b)
        self.model = Bm25(BM25_MODEL, cache_dir=cache_dir or FASTEMBED_CACHE, k=self.k1, b=self.b, avg_len=self.avg_len)

    def params(self) -> dict:
        return {"model": BM25_MODEL, "k1": self.k1, "b": self.b, "avg_len": self.avg_len}

    def tokens(self, text: str) -> list[str]:
        """The stems BM25 counts (same steps as fastembed's ``raw_embed``)."""
        from fastembed.common.utils import remove_non_alphanumeric

        return self.model._stem(self.model.tokenizer.tokenize(remove_non_alphanumeric(text)))

    def doc_len(self, text: str) -> int:
        return len(self.tokens(text))

    def embed_docs(self, texts: list[str]) -> list[Sparse]:
        """Indices sorted ascending (Qdrant stores sparse vectors sorted, so they round-trip exactly)."""
        out = []
        for e in self.model.raw_embed(texts):
            order = np.argsort(e.indices)
            out.append(Sparse(np.asarray(e.indices, np.int64)[order], np.asarray(e.values, np.float32)[order]))
        return out

    def embed_query(self, text: str) -> Sparse:
        e = next(iter(self.model.query_embed(text)))
        order = np.argsort(e.indices)
        return Sparse(np.asarray(e.indices, np.int64)[order], np.asarray(e.values, np.float32)[order])


def bm25_avg_len(texts: Iterable[str], cache_dir: str | None = None) -> float:
    """Mean BM25 document length (in stems) of one index's unit texts."""
    m = BM25(cache_dir=cache_dir)
    lens = [m.doc_len(t) for t in texts]
    return float(np.mean(lens))


# ---------------------------------------------------------------- packing helpers


def _pack_sparse(items: list[Sparse]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = np.zeros(len(items) + 1, np.int64)
    for i, s in enumerate(items):
        off[i + 1] = off[i] + len(s.indices)
    idx = np.concatenate([s.indices for s in items]).astype(np.int64) if items else np.zeros(0, np.int64)
    val = np.concatenate([s.values for s in items]).astype(np.float32) if items else np.zeros(0, np.float32)
    return idx, val, off


def _unpack_sparse(idx: np.ndarray, val: np.ndarray, off: np.ndarray) -> list[Sparse]:
    return [Sparse(idx[off[i] : off[i + 1]], val[off[i] : off[i + 1]]) for i in range(len(off) - 1)]


def _pack_multi(items: list[np.ndarray], dim: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    off = np.zeros(len(items) + 1, np.int64)
    for i, m in enumerate(items):
        off[i + 1] = off[i] + len(m)
    data = np.concatenate(items).astype(np.float16) if items else np.zeros((0, dim), np.float16)
    return data, off


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def batches_by_tokens(items: list, n_tokens: Callable, max_tokens: int = 8192, max_items: int = 16) -> list[list]:
    """Length-sorted batches with at most ``max_tokens`` tokens (or one oversized item)."""
    items = sorted(items, key=n_tokens)
    out, cur, tot = [], [], 0
    for it in items:
        n = n_tokens(it)
        if cur and (tot + n > max_tokens or len(cur) >= max_items):
            out.append(cur)
            cur, tot = [], 0
        cur.append(it)
        tot += n
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- corpus embeddings


class EmbeddingStore:
    """Vectors of one collection's units: dense, bge sparse, ColBERT and BM25.

    ``data/embeddings/<collection>.npz`` holds the arrays; ``<collection>.jsonl``
    lists one row per unit (id, text hash, vector sizes) and opens anywhere.
    BM25 vectors depend on ``avg_len``; they are recomputed if it changes.
    """

    def __init__(self, collection: str, root: str | Path = "data/embeddings"):
        self.collection = collection
        self.root = Path(root)
        self.npz = self.root / f"{collection}.npz"
        self.jsonl = self.root / f"{collection}.jsonl"
        self.rows: dict[str, dict] = {}
        self.meta: dict = {}
        if self.npz.is_file():
            self._load()

    def _load(self) -> None:
        z = np.load(self.npz, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        ids = [str(x) for x in z["ids"]]
        shas = [str(x) for x in z["text_sha"]]
        sp = _unpack_sparse(z["sp_idx"], z["sp_val"], z["sp_off"])
        bm = _unpack_sparse(z["bm25_idx"], z["bm25_val"], z["bm25_off"])
        cb, cb_off = z["colbert"], z["colbert_off"]
        dense = z["dense"]
        for i, uid in enumerate(ids):
            self.rows[uid] = {
                "text_sha": shas[i],
                "dense": dense[i],
                "bge_sparse": sp[i],
                "colbert": cb[cb_off[i] : cb_off[i + 1]],
                "bm25": bm[i],
            }

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        ids = list(self.rows)
        rows = [self.rows[i] for i in ids]
        sp_idx, sp_val, sp_off = _pack_sparse([r["bge_sparse"] for r in rows])
        bm_idx, bm_val, bm_off = _pack_sparse([r["bm25"] for r in rows])
        cb, cb_off = _pack_multi([r["colbert"] for r in rows])
        _atomic_savez(
            self.npz,
            meta=np.array(json.dumps(self.meta)),
            ids=np.array(ids),
            text_sha=np.array([r["text_sha"] for r in rows]),
            dense=np.stack([r["dense"] for r in rows]).astype(np.float32) if rows else np.zeros((0, 1024), np.float32),
            sp_idx=sp_idx, sp_val=sp_val, sp_off=sp_off,
            bm25_idx=bm_idx, bm25_val=bm_val, bm25_off=bm_off,
            colbert=cb, colbert_off=cb_off,
        )
        with open(self.jsonl, "w", encoding="utf-8") as f:
            for uid, r in zip(ids, rows):
                f.write(
                    json.dumps(
                        {
                            "unit_id": uid,
                            "text_sha": r["text_sha"],
                            "n_colbert": int(len(r["colbert"])),
                            "n_bge_sparse": int(len(r["bge_sparse"].indices)),
                            "n_bm25": int(len(r["bm25"].indices)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def missing(self, units: Iterable[dict]) -> list[dict]:
        return [u for u in units if self.rows.get(u["unit_id"], {}).get("text_sha") != text_sha(u["text"])]

    def build(
        self,
        units: list[dict],
        encoder: EncoderClient,
        bm25: BM25,
        n_tokens: Callable[[str], int],
        max_tokens: int = 8192,
        progress: Callable | None = None,
        save_every: int = 10,
    ) -> dict:
        """Embed every unit whose text is new or changed; (re)compute BM25 for all if ``avg_len`` changed."""
        todo = self.missing(units)
        bm_params = bm25.params()
        stale_bm25 = self.meta.get("bm25") != bm_params
        self.meta.update({"collection": self.collection, "bm25": bm_params, "encoder": encoder.model_revisions()})
        batches = batches_by_tokens(todo, lambda u: n_tokens(u["text"]), max_tokens=max_tokens)
        it = progress(batches, desc=f"embed {self.collection}") if progress else batches
        seconds = 0.0
        for bi, batch in enumerate(it):
            out = encoder.embed([u["text"] for u in batch], batch_size=4)
            seconds += out["seconds"]
            bm = bm25.embed_docs([u["text"] for u in batch])
            for j, u in enumerate(batch):
                self.rows[u["unit_id"]] = {
                    "text_sha": text_sha(u["text"]),
                    "dense": out["dense"][j],
                    "bge_sparse": out["sparse"][j],
                    "colbert": out["colbert"][j],
                    "bm25": bm[j],
                }
            if (bi + 1) % save_every == 0:
                self.save()
        if stale_bm25:
            keep = {u["unit_id"] for u in units}
            ids = [u["unit_id"] for u in units if u["unit_id"] in self.rows]
            for uid, vec in zip(ids, bm25.embed_docs([u["text"] for u in units if u["unit_id"] in self.rows])):
                self.rows[uid]["bm25"] = vec
            self.rows = {k: v for k, v in self.rows.items() if k in keep}
        self.meta["encoder_seconds"] = self.meta.get("encoder_seconds", 0.0) + seconds
        self.save()
        return {"embedded": len(todo), "bm25_recomputed": stale_bm25, "encoder_seconds": round(seconds, 1)}

    def get(self, unit_id: str) -> dict:
        return self.rows[unit_id]


# ---------------------------------------------------------------- query embeddings


class QueryEmbeddings:
    """bge-m3 vectors of query texts (questions, HyDE passages, keyword-expanded
    queries, agent searches), keyed by text hash; shared by both indexes."""

    def __init__(self, name: str, root: str | Path = "data/embeddings"):
        self.path = Path(root) / f"queries_{name}.npz"
        self.rows: dict[str, dict] = {}
        if self.path.is_file():
            z = np.load(self.path, allow_pickle=False)
            sp = _unpack_sparse(z["sp_idx"], z["sp_val"], z["sp_off"])
            cb, off = z["colbert"], z["colbert_off"]
            for i, h in enumerate(str(x) for x in z["keys"]):
                self.rows[h] = {
                    "dense": z["dense"][i],
                    "bge_sparse": sp[i],
                    "colbert": cb[off[i] : off[i + 1]],
                    "seconds": float(z["seconds"][i]),
                }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.rows)
        rows = [self.rows[k] for k in keys]
        sp_idx, sp_val, sp_off = _pack_sparse([r["bge_sparse"] for r in rows])
        cb, off = _pack_multi([r["colbert"] for r in rows])
        _atomic_savez(
            self.path,
            keys=np.array(keys),
            dense=np.stack([r["dense"] for r in rows]).astype(np.float32) if rows else np.zeros((0, 1024), np.float32),
            sp_idx=sp_idx, sp_val=sp_val, sp_off=sp_off,
            colbert=cb, colbert_off=off,
            seconds=np.array([r["seconds"] for r in rows], np.float64),
        )

    def get(self, texts: list[str], encoder: EncoderClient | None, save: bool = True) -> list[dict]:
        """Vectors for ``texts``; misses are embedded in one request each batch.
        ``seconds`` is the encoder time of the original call (latency is reported from it)."""
        keys = [text_sha(t) for t in texts]
        miss = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self.rows))
        if miss:
            if encoder is None:
                raise KeyError(f"{len(miss)} query embeddings not cached and no encoder given")
            for i in range(0, len(miss), 16):
                chunk = miss[i : i + 16]
                out = encoder.embed(chunk, batch_size=8)
                per = out["seconds"] / len(chunk)
                for j, t in enumerate(chunk):
                    self.rows[text_sha(t)] = {
                        "dense": out["dense"][j],
                        "bge_sparse": out["sparse"][j],
                        "colbert": out["colbert"][j],
                        "seconds": per,
                    }
            if save:
                self.save()
        return [self.rows[k] for k in keys]
