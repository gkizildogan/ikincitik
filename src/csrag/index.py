"""Qdrant collections ``cse_units_v1`` and ``cse_win500_v1`` (plan section 3.5).

Both collections have the same schema:

| named vector | type                     | config                                              |
|--------------|--------------------------|-----------------------------------------------------|
| dense        | dense, 1024              | Cosine, HNSW default                                |
| colbert      | multivector, 1024/token  | MAX_SIM, float16, hnsw m=0 (rerank / brute force)   |
| bge_sparse   | sparse                   | bge-m3 lexical weights, no modifier                 |
| bm25         | sparse                   | modifier=IDF                                        |

ColBERT vectors are L2-normalised by bge-m3, so the DOT product equals the
cosine; using DOT means Qdrant stores the float16 values as given and they
round-trip exactly.

Qdrant limit: a multivector holds at most 1,048,575 numbers, i.e. 1023 ColBERT
vectors of 1024 dimensions. Every window fits, but 60 of the 252 base units are
longer. Those points are stored without the ``colbert`` vector, and C1/M1 score
ColBERT MaxSim client-side over the cached full-length vectors for both indexes
(``retrievers.qdrant_search.ColbertIndex``; the same math as Qdrant's MAX_SIM
with DOT). ``check_colbert_engines`` confirms both engines agree where Qdrant
holds the vectors.
"""

from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np

from .corpus import COLLECTIONS
from .embed import BM25, EmbeddingStore
from .enrich import point_id

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")

PAYLOAD_FIELDS = [
    "unit_id", "index", "section_id", "group_id", "kind", "subpart", "engine_type", "page_start", "page_end",
    "part_index", "part_count", "char_start", "char_end", "n_tokens_bge", "breadcrumb", "cross_refs",
]
MAX_MULTIVECTOR = 1023  # Qdrant: total multivector size must be < 1,048,576 numbers (1024 x 1024)
KEYWORD_INDEXES = ["section_id", "group_id", "kind", "subpart", "engine_type", "cross_refs", "unit_id"]
INTEGER_INDEXES = ["page_start", "page_end"]


def connect(url: str | None = None, prefer_grpc: bool = True, timeout: int = 600):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url or QDRANT_URL, prefer_grpc=prefer_grpc, timeout=timeout)


def create_collection(client, index: str, recreate: bool = False) -> str:
    from qdrant_client import models as m

    name = COLLECTIONS[index]
    if client.collection_exists(name):
        if not recreate:
            return name
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={
            "dense": m.VectorParams(size=1024, distance=m.Distance.COSINE),
            "colbert": m.VectorParams(
                size=1024,
                distance=m.Distance.DOT,
                multivector_config=m.MultiVectorConfig(comparator=m.MultiVectorComparator.MAX_SIM),
                datatype=m.Datatype.FLOAT16,
                hnsw_config=m.HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            "bge_sparse": m.SparseVectorParams(),
            "bm25": m.SparseVectorParams(modifier=m.Modifier.IDF),
        },
    )
    for f in KEYWORD_INDEXES:
        client.create_payload_index(name, field_name=f, field_schema=m.PayloadSchemaType.KEYWORD)
    for f in INTEGER_INDEXES:
        client.create_payload_index(name, field_name=f, field_schema=m.PayloadSchemaType.INTEGER)
    return name


def payload(unit: dict, corpus_version: str) -> dict:
    p = {k: unit[k] for k in PAYLOAD_FIELDS}
    p["corpus_version"] = corpus_version
    p["text"] = unit["text"]
    return p


def _point(unit: dict, vec: dict, corpus_version: str):
    from qdrant_client import models as m

    vectors = {
        "dense": vec["dense"].astype(np.float32).tolist(),
        "bge_sparse": m.SparseVector(indices=vec["bge_sparse"].indices.tolist(), values=vec["bge_sparse"].values.tolist()),
        "bm25": m.SparseVector(indices=vec["bm25"].indices.tolist(), values=vec["bm25"].values.tolist()),
    }
    if len(vec["colbert"]) <= MAX_MULTIVECTOR:
        vectors["colbert"] = vec["colbert"].astype(np.float32).tolist()
    p = payload(unit, corpus_version)
    p["colbert_in_qdrant"] = len(vec["colbert"]) <= MAX_MULTIVECTOR
    p["n_colbert"] = int(len(vec["colbert"]))
    return m.PointStruct(id=point_id(unit["unit_id"]), vector=vectors, payload=p)


def upsert(client, index: str, units: list[dict], store: EmbeddingStore, corpus_version: str,
           max_colbert_tokens: int = 4096, progress: Callable | None = None) -> int:
    """Upsert every unit (idempotent: point id = uuid5(unit_id)). Batches are
    bounded by ColBERT tokens to keep each gRPC message small."""
    name = COLLECTIONS[index]
    batches, cur, tot = [], [], 0
    for u in units:
        n = min(len(store.get(u["unit_id"])["colbert"]), MAX_MULTIVECTOR + 1)
        if cur and tot + n > max_colbert_tokens:
            batches.append(cur)
            cur, tot = [], 0
        cur.append(u)
        tot += n
    if cur:
        batches.append(cur)
    it = progress(batches, desc=f"upsert {name}") if progress else batches
    for b in it:
        client.upsert(name, points=[_point(u, store.get(u["unit_id"]), corpus_version) for u in b], wait=True)
    return sum(len(b) for b in batches)


# ---------------------------------------------------------------- checks


def _all_payloads(client, name: str) -> dict[str, dict]:
    out, offset = {}, None
    while True:
        pts, offset = client.scroll(name, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            out[p.payload["unit_id"]] = {"point_id": str(p.id), **p.payload}
        if offset is None:
            return out


def verify(client, index: str, units: list[dict], store: EmbeddingStore, bm25: BM25,
           n_roundtrip: int = 10, seed: int = 0, progress: Callable | None = None) -> dict:
    """Ingestion checks: point count, ids, exact vector round-trip, self-retrieval
    (dense and BM25 at rank 1; for windows an overlapping neighbour also counts)."""
    from qdrant_client import models as m

    name = COLLECTIONS[index]
    res: dict = {"collection": name}
    info = client.get_collection(name)
    res["points"] = info.points_count
    res["count_ok"] = info.points_count == len(units)
    stored = _all_payloads(client, name)
    ids = {u["unit_id"] for u in units}
    res["missing_in_qdrant"] = sorted(ids - set(stored))
    res["extra_in_qdrant"] = sorted(set(stored) - ids)
    res["point_id_mismatch"] = [u for u in ids & set(stored) if stored[u]["point_id"] != point_id(u)]

    rng = random.Random(seed)
    sample = rng.sample(sorted(ids), min(n_roundtrip, len(ids)))
    bad = []
    for uid in sample:
        p = client.retrieve(name, ids=[point_id(uid)], with_vectors=True)[0]
        v, e = p.vector, store.get(uid)
        ok_dense = np.allclose(np.asarray(v["dense"], np.float32), e["dense"], atol=1e-6)
        if len(e["colbert"]) <= MAX_MULTIVECTOR:
            ok_colbert = "colbert" in v and np.array_equal(np.asarray(v["colbert"], np.float32), e["colbert"].astype(np.float32))
        else:
            ok_colbert = "colbert" not in v  # too long for Qdrant: scored client-side
        def same_sparse(got, exp) -> bool:
            a, b = np.argsort(got.indices), np.argsort(exp.indices)
            return (np.array_equal(np.asarray(got.indices)[a], exp.indices[b])
                    and np.array_equal(np.asarray(got.values, np.float32)[a], exp.values[b]))

        ok_sparse = all(same_sparse(v[k], e[k]) for k in ("bge_sparse", "bm25"))
        if not (ok_dense and ok_colbert and ok_sparse):
            bad.append({"unit_id": uid, "dense": ok_dense, "colbert": ok_colbert, "sparse": ok_sparse})
    res["roundtrip_checked"] = len(sample)
    res["colbert_in_qdrant"] = sum(1 for u in units if len(store.get(u["unit_id"])["colbert"]) <= MAX_MULTIVECTOR)
    res["roundtrip_failed"] = bad

    by_id = {u["unit_id"]: u for u in units}

    def same_or_neighbour(a: str, b: str) -> bool:
        if a == b:
            return True
        if index != "win500":
            return False
        ua, ub = by_id[a], by_id[b]
        return ua["section_id"] == ub["section_id"] and ua["char_start"] < ub["char_end"] and ub["char_start"] < ua["char_end"]

    fails = {"dense": [], "bm25": []}
    exact = {"dense": 0, "bm25": 0}
    it = progress(units, desc=f"self-retrieval {name}") if progress else units
    for u in it:
        uid = u["unit_id"]
        for sig in ("dense", "bm25"):
            if sig == "dense":
                q = store.get(uid)["dense"].tolist()
            else:
                s = bm25.embed_query(u["text"])
                q = m.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
            top = client.query_points(name, query=q, using=sig, limit=1, with_payload=["unit_id"]).points
            got = top[0].payload["unit_id"] if top else None
            exact[sig] += got == uid
            if got is None or not same_or_neighbour(uid, got):
                fails[sig].append((uid, got))
    n = len(units)
    res["self_rank1"] = {s: round(1 - len(f) / n, 4) for s, f in fails.items()}
    res["self_rank1_exact_id"] = {s: round(c / n, 4) for s, c in exact.items()}
    res["self_fail"] = fails
    res["ok"] = (
        res["count_ok"] and not res["missing_in_qdrant"] and not res["extra_in_qdrant"] and not bad
        and all(v >= 0.99 for v in res["self_rank1"].values())
    )
    return res


def check_colbert_engines(client, index: str, colbert_index, queries: list[np.ndarray], tol: float = 1e-3) -> dict:
    """Qdrant MaxSim vs client MaxSim on the points that hold a ColBERT vector in Qdrant."""
    from qdrant_client import models as m

    name = COLLECTIONS[index]
    flt = m.Filter(must=[m.FieldCondition(key="colbert_in_qdrant", match=m.MatchValue(value=True))])
    worst = 0.0
    for q in queries:
        res = client.query_points(name, query=np.asarray(q, np.float32).tolist(), using="colbert", query_filter=flt,
                                  limit=50, with_payload=["unit_id"], search_params=m.SearchParams(exact=True)).points
        mine = dict(colbert_index.search(q, None, 50, ids=[p.payload["unit_id"] for p in res]))
        worst = max([worst] + [abs(mine[p.payload["unit_id"]] - p.score) for p in res])
    return {"queries": len(queries), "max_abs_diff": worst, "agree": worst < tol}
