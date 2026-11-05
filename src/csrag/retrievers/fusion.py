"""Client-side reciprocal rank fusion (Cormack et al.), and its check against Qdrant.

score(d) = sum over lists of 1 / (k + rank_d), rank from 1, k = 60.

Qdrant's own RRF counts ranks from 0 and defaults to k = 2, so ``k = 61``
there reproduces these scores (plan sections 4.0 and 8).
"""

from __future__ import annotations

from ..corpus import COLLECTIONS


def rrf(lists: list[list], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of ``(unit_id, score)`` into ``[(unit_id, rrf_score)]``, best first.
    Ties break on the best single rank, then on the unit id (deterministic)."""
    score: dict[str, float] = {}
    best: dict[str, int] = {}
    for lst in lists:
        for rank, (uid, _) in enumerate(lst, 1):
            score[uid] = score.get(uid, 0.0) + 1.0 / (k + rank)
            best[uid] = min(best.get(uid, rank), rank)
    return sorted(score.items(), key=lambda x: (-x[1], best[x[0]], x[0]))


def qdrant_rrf(client, index: str, prefetch: list[tuple[str, object]], limit: int = 50, k: int = 61,
               query_filter=None, out_limit: int = 100) -> list[tuple[str, float]]:
    """Server-side RRF of the given ``(using, query_vector)`` prefetches (the check only)."""
    from qdrant_client import models as m

    res = client.query_points(
        COLLECTIONS[index],
        prefetch=[m.Prefetch(query=v, using=u, limit=limit, filter=query_filter) for u, v in prefetch],
        query=m.RrfQuery(rrf=m.Rrf(k=k)),
        limit=out_limit,
        with_payload=["unit_id"],
    ).points
    return [(p.payload["unit_id"], float(p.score)) for p in res]
