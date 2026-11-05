"""CPU-only model server: bge-m3 (dense + sparse + ColBERT) and two rerankers.

Endpoints
  POST /embed   {texts, dense, sparse, colbert, batch_size, max_length}
                -> one bge-m3 forward pass per batch (BGEM3FlagModel)
  POST /rerank  {model: "bge" | "jina", query, documents}
                -> one score per document, in input order
  GET  /health  -> loaded models, their pinned revisions and library versions

Models are loaded from local snapshot paths (HF cache layout,
``.../models--org--name/snapshots/<revision>``) with HF_HUB_OFFLINE=1, so the
revision is the snapshot directory's name. Everything runs in fp32 on the CPU.
Model calls are serialised with one lock: concurrent requests would only
fight over the same cores.
"""

from __future__ import annotations

import base64
import importlib.metadata as md
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- config

THREADS = int(os.environ.get("TORCH_THREADS", "12"))
torch.set_num_threads(THREADS)
torch.set_grad_enabled(False)

PATHS = {
    "bge_m3": os.environ.get("BGE_M3_PATH", ""),
    "bge_reranker": os.environ.get("BGE_RERANKER_PATH", ""),
    "jina_reranker": os.environ.get("JINA_RERANKER_PATH", ""),
}
LOAD = [m.strip() for m in os.environ.get("LOAD_MODELS", "bge_m3,bge_reranker,jina_reranker").split(",") if m.strip()]

BGE_M3_MAX_LENGTH = 8192
BGE_RERANK_MAX_LENGTH = 8192  # FlagEmbedding's default of 512 would cut ~half the base units
JINA_MAX_DOCS = 64
JINA_MAX_DOC_TOKENS = 8192  # model default is 2048; no unit or window comes close to this
JINA_MAX_QUERY_TOKENS = 1024

LOCK = threading.Lock()
MODELS: dict[str, object] = {}
INFO: dict[str, dict] = {}


def _resolve(path: str) -> Path:
    """A snapshot directory: either the path itself, or the only/ref'd snapshot under a ``models--*`` dir."""
    p = Path(path)
    if (p / "snapshots").is_dir():
        ref = p / "refs" / "main"
        if ref.is_file():
            return p / "snapshots" / ref.read_text().strip()
        snaps = sorted((p / "snapshots").iterdir())
        if len(snaps) != 1:
            raise RuntimeError(f"{p}: expected exactly one snapshot, found {[s.name for s in snaps]}")
        return snaps[0]
    return p


def _load_bge_m3(path: Path):
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(str(path), use_fp16=False, devices="cpu", normalize_embeddings=True)


def _load_bge_reranker(path: Path):
    from FlagEmbedding import FlagReranker

    return FlagReranker(str(path), use_fp16=False, devices="cpu", max_length=BGE_RERANK_MAX_LENGTH)


def _load_jina(path: Path):
    from transformers import AutoModel

    m = AutoModel.from_pretrained(str(path), trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa")
    m.eval()
    return m


LOADERS = {"bge_m3": _load_bge_m3, "bge_reranker": _load_bge_reranker, "jina_reranker": _load_jina}


def load_models() -> None:
    for name in LOAD:
        if name not in LOADERS:
            raise RuntimeError(f"unknown model {name!r}; choose from {sorted(LOADERS)}")
        if not PATHS[name]:
            raise RuntimeError(f"{name}: path env var not set")
        snap = _resolve(PATHS[name])
        t0 = time.time()
        MODELS[name] = LOADERS[name](snap)
        INFO[name] = {
            "path": str(snap),
            "repo": snap.parent.parent.name.removeprefix("models--").replace("--", "/"),
            "revision": snap.name,
            "dtype": "float32",
            "load_seconds": round(time.time() - t0, 1),
        }


# ---------------------------------------------------------------- api

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    load_models()
    yield


app = FastAPI(title="ikincitik encoder", version="1", lifespan=_lifespan)


def _pkg(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if all(n in MODELS for n in LOAD) else "loading",
        "models": INFO,
        "torch_threads": torch.get_num_threads(),
        "versions": {p: _pkg(p) for p in ("torch", "transformers", "FlagEmbedding", "tokenizers", "fastapi")},
        "limits": {
            "bge_m3_max_length": BGE_M3_MAX_LENGTH,
            "bge_rerank_max_length": BGE_RERANK_MAX_LENGTH,
            "jina_max_docs": JINA_MAX_DOCS,
            "jina_max_doc_tokens": JINA_MAX_DOC_TOKENS,
            "jina_max_query_tokens": JINA_MAX_QUERY_TOKENS,
        },
    }


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    dense: bool = True
    sparse: bool = True
    colbert: bool = True
    batch_size: int = 4
    max_length: int = BGE_M3_MAX_LENGTH


def _b64(a: np.ndarray, dtype: str) -> dict:
    a = np.ascontiguousarray(a, dtype=dtype)
    return {"shape": list(a.shape), "dtype": dtype, "b64": base64.b64encode(a.tobytes()).decode("ascii")}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    """bge-m3 outputs from one forward pass.

    - dense: float32, L2-normalised, base64 of shape (n, 1024)
    - sparse: per text {indices, values} (token id -> lexical weight; special tokens dropped)
    - colbert: per text base64 float16 of shape (n_tokens, 1024), L2-normalised rows
      (float16 is what Qdrant stores, so the values round-trip exactly)
    """
    if "bge_m3" not in MODELS:
        raise HTTPException(503, "bge_m3 not loaded")
    if req.max_length > BGE_M3_MAX_LENGTH:
        raise HTTPException(400, f"max_length > {BGE_M3_MAX_LENGTH}")
    m = MODELS["bge_m3"]
    t0 = time.time()
    with LOCK:
        out = m.encode(
            req.texts,
            batch_size=req.batch_size,
            max_length=req.max_length,
            return_dense=req.dense,
            return_sparse=req.sparse,
            return_colbert_vecs=req.colbert,
        )
    res: dict = {"n": len(req.texts)}
    if req.dense:
        res["dense"] = _b64(np.asarray(out["dense_vecs"]), "float32")
    if req.sparse:
        sp = []
        for w in out["lexical_weights"]:
            items = sorted((int(k), float(v)) for k, v in w.items())
            sp.append({"indices": [k for k, _ in items], "values": [v for _, v in items]})
        res["sparse"] = sp
    if req.colbert:
        res["colbert"] = [_b64(np.asarray(v), "float16") for v in out["colbert_vecs"]]
    res["seconds"] = round(time.time() - t0, 3)
    return res


class RerankRequest(BaseModel):
    model: Literal["bge", "jina"]
    query: str
    documents: list[str] = Field(min_length=1)
    batch_size: int = 4


@app.post("/rerank")
def rerank(req: RerankRequest) -> dict:
    """One score per document, in input order.

    - bge: raw logits (no sigmoid), pointwise, max_length 8192
    - jina: listwise cosine scores; all documents in one context (<= 64 per call)
    """
    t0 = time.time()
    if req.model == "bge":
        if "bge_reranker" not in MODELS:
            raise HTTPException(503, "bge_reranker not loaded")
        pairs = [[req.query, d] for d in req.documents]
        with LOCK:
            s = MODELS["bge_reranker"].compute_score(
                pairs, batch_size=req.batch_size, max_length=BGE_RERANK_MAX_LENGTH, normalize=False
            )
        scores = [float(x) for x in (s if isinstance(s, list) else [s])]
    else:
        if "jina_reranker" not in MODELS:
            raise HTTPException(503, "jina_reranker not loaded")
        if len(req.documents) > JINA_MAX_DOCS:
            raise HTTPException(400, f"jina takes at most {JINA_MAX_DOCS} documents per call")
        with LOCK:
            ranked = MODELS["jina_reranker"].rerank(
                req.query,
                req.documents,
                max_doc_length=JINA_MAX_DOC_TOKENS,
                max_query_length=JINA_MAX_QUERY_TOKENS,
            )
        scores = [0.0] * len(req.documents)
        for r in ranked:
            scores[int(r["index"])] = float(r["relevance_score"])
    return {"model": req.model, "scores": scores, "seconds": round(time.time() - t0, 3)}


@app.post("/tokens")
def tokens(req: RerankRequest) -> dict:
    """Token counts of query and documents under the chosen reranker's tokenizer (truncation check)."""
    if req.model == "bge":
        tok = MODELS["bge_reranker"].tokenizer
        q = len(tok(req.query)["input_ids"])
        d = [len(tok(req.query, x)["input_ids"]) for x in req.documents]
        return {"query": q, "pairs": d, "max_length": BGE_RERANK_MAX_LENGTH}
    m = MODELS["jina_reranker"]
    m._ensure_tokenizer()
    tok = m._tokenizer
    return {
        "query": len(tok(req.query)["input_ids"]),
        "documents": [len(tok(x)["input_ids"]) for x in req.documents],
        "max_doc_length": JINA_MAX_DOC_TOKENS,
    }
