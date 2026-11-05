"""Score gate (plan section 4.1).

For one list with scores s1 >= s2 >= ... >= sn:

1. background b = the median score of the list;
2. relative strength r_i = (s_i - b) / (s_1 - b)   (best hit = 1, typical unrelated hit ~ 0);
3. cut before the first rank i where r_i < tau ("a few hits dominate")
   or r_{i-1} - r_i >= delta ("the score falls off a cliff");
4. guards: keep at least ``min_keep`` and at most ``max_keep`` hits; if the list
   has fewer than 2 * min_keep hits, or s1 = b, nothing is cut.

One rule for every scorer: BM25 sums, sparse dot products, cosines, ColBERT
MaxSim sums and reranker logits are all measured against their own list.
The gate always returns a prefix of its input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GateParams:
    tau: float
    delta: float
    min_keep: int
    max_keep: int

    def as_dict(self) -> dict:
        return asdict(self)


def gate_keep(scores: list[float], p: GateParams) -> int:
    """Number of leading hits the gate keeps."""
    n = len(scores)
    if n == 0:
        return 0
    s = np.asarray(scores, dtype=np.float64)
    if np.any(np.diff(s) > 1e-12):
        raise ValueError("gate expects scores sorted in descending order")
    if n < 2 * p.min_keep:
        return min(n, p.max_keep)
    b = float(np.median(s))
    top = float(s[0])
    if top <= b:
        return min(n, p.max_keep)
    r = (s - b) / (top - b)
    keep = n
    for i in range(1, n):
        if r[i] < p.tau or r[i - 1] - r[i] >= p.delta:
            keep = i
            break
    return int(min(max(keep, p.min_keep), p.max_keep, n))


def apply_gate(hits: list, p: GateParams | None, score=lambda h: h[1]) -> tuple[list, dict]:
    """(kept prefix, trace). ``p=None`` means gate off (the list is returned whole)."""
    if p is None:
        return list(hits), {"n_in": len(hits), "n_kept": len(hits)}
    k = gate_keep([score(h) for h in hits], p)
    return list(hits[:k]), {"n_in": len(hits), "n_kept": k}
