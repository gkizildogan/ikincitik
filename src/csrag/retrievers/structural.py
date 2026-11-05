"""Structural methods: T1 id lookup (plan 4.4) and the fixed expansion rules (4.7).

Fixed rules, all on H1's fused list and under one token budget (bge-m3 tokens
of the merged passages that would be handed over):

- T0     no expansion: H1's units/windows in rank order, overlapping windows merged
- T2     parent expansion (small-to-big): each hit -> its whole section (or group);
         if the parent does not fit the remaining budget, the hit itself is added
- T3     cross-references: each hit, then the sections it references (in order
         of mention), plus the AMCs of a retrieved CS; one hop
- T2+T3  both: each hit -> its parent section, then the hit's references

Items that do not fit are skipped and filling continues with the next ones.

One hop, no chains: search results and their expansions are hop 0; what a
reference brings in is hop 1, and references are followed only from hop-0
content. A unit that is both a search result and a reference target counts as
hop 0 (a search result).
"""

from __future__ import annotations

from ..enrich import find_section_ids
from .base import Hit


def id_lookup(text: str, corpus) -> list[str]:
    """Section ids named in the question (e.g. "CS-E 740", "AMC E 20(f)"), in order of mention."""
    return find_section_ids(text, corpus.section_ids)


class ContextBuilder:
    """An ordered evidence set under a token budget."""

    def __init__(self, corpus, index: str, budget: int):
        self.corpus = corpus
        self.index = index
        self.budget = budget
        self.items: list[dict] = []  # unit_id, section_id, hop, stage, score, source
        self.pos: dict[str, int] = {}
        self.tokens = 0

    def __contains__(self, unit_id: str) -> bool:
        return unit_id in self.pos

    def hop_of(self, unit_id: str) -> int:
        return self.items[self.pos[unit_id]]["hop"]

    def cost_with(self, unit_ids: list[str]) -> int:
        ids = [i["unit_id"] for i in self.items] + [u for u in unit_ids if u not in self.pos]
        return self.corpus.context_tokens(ids, self.index)

    def add(self, units: list[dict], hop: int, stage: str, score: float = 0.0, source: str | None = None) -> dict:
        """Add ``units`` together if they fit: ``{"added": [...], "tokens_added": n, "refused": bool, "needed": n}``."""
        new = [u for u in units if u["unit_id"] not in self.pos]
        if hop == 0:
            for u in units:  # a search result already present as a reference target becomes hop 0
                if u["unit_id"] in self.pos and self.items[self.pos[u["unit_id"]]]["hop"] == 1:
                    self.items[self.pos[u["unit_id"]]].update(hop=0, source=None)
        if not new:
            return {"added": [], "tokens_added": 0, "refused": False, "needed": 0}
        t = self.cost_with([u["unit_id"] for u in new])
        if t > self.budget:
            return {"added": [], "tokens_added": 0, "refused": True, "needed": t - self.tokens}
        for u in new:
            self.pos[u["unit_id"]] = len(self.items)
            self.items.append(
                {"unit_id": u["unit_id"], "section_id": u["section_id"], "hop": hop, "stage": stage, "score": score, "source": source}
            )
        added = t - self.tokens
        self.tokens = t
        return {"added": [u["unit_id"] for u in new], "tokens_added": added, "refused": False, "needed": added}

    @property
    def remaining(self) -> int:
        return self.budget - self.tokens

    def hits(self) -> list[Hit]:
        units = self.corpus.units(self.index)
        return [
            Hit(i["unit_id"], i["section_id"], i["score"], i["stage"], i["hop"],
                {"source": i["source"]} if i["source"] else {}, units[i["unit_id"]]["n_tokens_bge"], True)
            for i in self.items
        ]


def ref_targets(corpus, unit: dict) -> list[str]:
    """Sections reached by one hop from a unit: its references in order of mention,
    then (for a CS) the AMCs of its group."""
    out = list(unit["cross_refs"])
    for a in corpus.amcs_of(unit["section_id"]):
        if a not in out and a != unit["section_id"]:
            out.append(a)
    return out


def expand_fixed(rule: str, pool: list[tuple[str, float]], corpus, index: str, budget: int, to: str = "section") -> ContextBuilder:
    """T0 / T2 / T3 / T2+T3 over H1's fused list ``pool`` = ``[(unit_id, score)]``."""
    units = corpus.units(index)
    ctx = ContextBuilder(corpus, index, budget)
    for uid, score in pool:
        if ctx.remaining <= 0:
            break
        u = units[uid]
        if rule in ("parent", "parent+refs"):
            parent = corpus.units_of(u["section_id"], index) if to == "section" else corpus.units_of_group(u["group_id"], index)
            r = ctx.add(parent, 0, "expanded", score, None)
            if r["refused"]:
                ctx.add([u], 0, "fused", score, None)
            elif uid in ctx:
                ctx.items[ctx.pos[uid]].update(stage="fused", score=score)
        else:
            ctx.add([u], 0, "fused", score, None)
        if rule in ("refs", "parent+refs") and uid in ctx and ctx.hop_of(uid) == 0:
            for dst in ref_targets(corpus, u):
                ctx.add(corpus.units_of(dst, index), 1, "expanded", 0.0, uid)
    return ctx


def lookup_then(ids: list[str], pool: list[tuple[str, float]], corpus, index: str, k: int) -> list[Hit]:
    """T1: the units/windows of the named sections first (in order), then the pool, deduplicated."""
    units = corpus.units(index)
    out: list[Hit] = []
    seen: set[str] = set()
    for sid in ids:
        for u in corpus.units_of(sid, index):
            if u["unit_id"] not in seen:
                seen.add(u["unit_id"])
                # lookup hits rank above every fused hit (RRF scores are < 1)
                out.append(Hit(u["unit_id"], sid, 1000.0 - len(out), "first", 0, {"lookup": True}, u["n_tokens_bge"]))
    for uid, score in pool:
        if uid not in seen:
            seen.add(uid)
            out.append(Hit(uid, units[uid]["section_id"], score, "fused", 0, {}, units[uid]["n_tokens_bge"]))
    return out[:k]
