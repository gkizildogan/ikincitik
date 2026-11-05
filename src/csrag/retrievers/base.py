"""Common types, the fixed parameters (plan section 4.0) and the method configs.

Every retriever implements one interface. It receives only the question id
(for caching) and the question text, never the labels (the oracle F0 is the
one exception and gets its labels at construction):

    retrieve(question_id: str, text: str, k: int = 20) -> list[Hit]

A configuration is a plain dict (stages: query_transform -> first_stages ->
gate A -> fusion -> filter -> colbert_rerank / rerank -> gate B -> lookup ->
expansion). It is hashed into ``config_hash``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..gate import GateParams

STAGES = ("first", "gated", "fused", "reranked", "expanded")


@dataclass
class Hit:
    unit_id: str
    section_id: str
    score: float
    stage: str
    hop: int = 0
    debug: dict = field(default_factory=dict)
    n_tokens: int = 0
    in_context: bool = True

    def row(self, question_id: str, rank: int) -> dict:
        return {
            "question_id": question_id,
            "rank": rank,
            "unit_id": self.unit_id,
            "section_id": self.section_id,
            "score": float(self.score),
            "stage": self.stage,
            "hop": int(self.hop),
            "n_tokens": int(self.n_tokens),
            "in_context": bool(self.in_context),
            "source": self.debug.get("source"),
        }


# ---------------------------------------------------------------- fixed parameters

PARAMS: dict[str, dict] = {
    "v1": {
        "candidates": 50,  # top-50 per signal from Qdrant
        "k": 20,  # final list length (shorter when gate B cuts)
        "rrf_k": 60,  # score = sum 1 / (60 + rank), rank from 1
        "gate_a": GateParams(tau=0.4, delta=0.3, min_keep=5, max_keep=50),
        "gate_b": GateParams(tau=0.2, delta=0.5, min_keep=5, max_keep=20),
        "bm25": {"k1": 1.2, "b": 0.75},  # avg_len = the index's mean length (set at ingestion)
        "hyde_passages": 1,
        "rerank_input": 50,  # H1's fused list, at most 50
        "budgets": [4096, 8192, 16384],  # expansion token budgets (bge-m3 tokens)
        "hops": 1,
        "t2_to": "section",  # small-to-big target of T2
        "agent_step_limit": 5,
        "agent_initial_hits": 5,  # H1 hits in the agent's starting evidence
        "agent_search_top": 5,  # new results a search adds
        "llm": {"temperature": 0.0, "thinking": False},
        # Qdrant holds at most 1023 ColBERT vectors per point; 60 base units are longer,
        # so MaxSim is computed client-side, exactly, for both indexes (plan deviation).
        "colbert_engine": "client",
    },
}

# v2 is written here by hand, once, after reading all v1 metrics (plan section 4.8):
#   PARAMS["v2"] = revise("v1", gate_a=GateParams(...), ...)


def revise(base: str, **changes) -> dict:
    p = copy.deepcopy(PARAMS[base])
    p.update(changes)
    return p


def get_params(version: str) -> dict:
    if version not in PARAMS:
        raise KeyError(f"params_version {version!r} is not defined; v2 is set by hand after reading v1 (plan 4.8)")
    return PARAMS[version]


# ---------------------------------------------------------------- method configs

SINGLE = {"L1": "bm25", "S1": "bge_sparse", "D1": "dense", "C1": "colbert"}
RANKED_METHODS = ["L1", "S1", "D1", "C1", "H1", "H2", "M1", "F0", "F1", "T1", "Q1", "Q2", "X1", "X2"]
EXPANSION_METHODS = ["T0", "T2", "T3", "T2+T3", "T4"]
ALL_METHODS = RANKED_METHODS + EXPANSION_METHODS

DESCRIPTIONS = {
    "L1": "BM25",
    "S1": "bge-m3 learned sparse",
    "D1": "bge-m3 dense",
    "C1": "ColBERT MaxSim over all units",
    "H1": "BM25 + dense, RRF",
    "H2": "BM25 + bge-sparse + dense, RRF",
    "M1": "dense -> ColBERT rerank",
    "F0": "H1 + oracle engine-type filter",
    "F1": "H1 + LLM-extracted engine-type filter",
    "T1": "section-id lookup, then H1",
    "Q1": "HyDE (dense)",
    "Q2": "LLM keyword expansion (BM25)",
    "X1": "H1 -> bge-reranker-v2-m3",
    "X2": "H1 -> jina-reranker-v3",
    "T0": "H1, no expansion (budgeted)",
    "T2": "H1 -> parent section (small-to-big)",
    "T3": "H1 -> follow cross-references (1 hop)",
    "T2+T3": "H1 -> parent section + cross-references",
    "T4": "H1 -> agentic small-to-big",
}


def method_config(method: str, index: str, gate: bool, params_version: str = "v1",
                  budget: int | None = None, params: dict | None = None, prompts: dict | None = None) -> dict:
    """The full, readable config of one run. ``prompts`` maps prompt name -> hash
    (from ``query_llm.prompt_hashes()``), so editing a prompt changes ``config_hash``."""
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    P = params or get_params(params_version)
    if method == "T4" and not gate:
        raise ValueError("T4 runs with the gate on only")
    if method in EXPANSION_METHODS and budget is None:
        raise ValueError(f"{method} needs a token budget")
    ga = P["gate_a"].as_dict() if gate else None
    gb = P["gate_b"].as_dict() if gate else None
    llm = {**P["llm"], "prompts": prompts or {}}
    cfg: dict = {
        "method": method,
        "description": DESCRIPTIONS[method],
        "index": index,
        "gate": bool(gate),
        "params_version": params_version,
        "k": P["k"],
        "candidates": P["candidates"],
        "query_transform": None,
        "first_stages": [],
        "gate_a": ga,
        "fusion": None,
        "filter": None,
        "colbert_rerank": None,
        "rerank": None,
        "gate_b": None,
        "lookup": None,
        "expansion": None,
        "bm25": dict(P["bm25"]),
        "llm": None,
    }
    h1 = {"first_stages": ["bm25", "dense"], "fusion": {"type": "rrf", "k": P["rrf_k"]}}
    if method in ("C1", "M1"):
        cfg["colbert_engine"] = P["colbert_engine"]
    if method in SINGLE:
        cfg["first_stages"] = [SINGLE[method]]
    elif method == "H1":
        cfg.update(h1)
    elif method == "H2":
        cfg.update(first_stages=["bm25", "bge_sparse", "dense"], fusion={"type": "rrf", "k": P["rrf_k"]})
    elif method == "M1":
        cfg.update(first_stages=["dense"], colbert_rerank={"input": "dense top-50 after gate A", "top": P["k"]}, gate_b=gb)
    elif method == "F0":
        cfg.update(h1, filter={"type": "oracle", "field": "engine_type", "label": "engine_type"})
    elif method == "F1":
        cfg.update(h1, filter={"type": "llm", "field": "engine_type"}, llm=llm)
    elif method == "T1":
        cfg.update(h1, lookup={"type": "section_id_regex", "then": "H1"})
    elif method == "Q1":
        cfg.update(first_stages=["dense"], query_transform={"type": "hyde", "passages": P["hyde_passages"],
                                                            "combine": "normalised mean of question and passage"}, llm=llm)
    elif method == "Q2":
        cfg.update(first_stages=["bm25"], query_transform={"type": "keywords", "combine": "question + terms"}, llm=llm)
    elif method in ("X1", "X2"):
        model = "bge" if method == "X1" else "jina"
        cfg.update(h1, rerank={"model": model, "input": P["rerank_input"], "top": P["k"]}, gate_b=gb)
    elif method in ("T0", "T2", "T3", "T2+T3"):
        rule = {"T0": "none", "T2": "parent", "T3": "refs", "T2+T3": "parent+refs"}[method]
        cfg.update(h1, expansion={"rule": rule, "budget": int(budget), "hops": P["hops"], "to": P["t2_to"],
                                  "pool": "H1 fused list"})
    elif method == "T4":
        cfg.update(h1, llm=llm, expansion={
            "rule": "agent", "budget": int(budget), "hops": P["hops"], "step_limit": P["agent_step_limit"],
            "initial_hits": P["agent_initial_hits"], "search_top": P["agent_search_top"],
            "tools": ["expand", "follow_refs", "search", "done"],
        })
    return cfg


def grid(params_version: str = "v1", indexes=("units", "win500"), prompts: dict | None = None,
         params: dict | None = None, methods=None) -> list[dict]:
    """Every configuration of the run plan (plan 4.8): every method on both
    indexes, gate off and on (T4 gate on only), expansion methods at every budget."""
    P = params or get_params(params_version)
    out = []
    for m in methods or ALL_METHODS:
        for index in indexes:
            for gate in ((True,) if m == "T4" else (False, True)):
                if m in EXPANSION_METHODS:
                    for b in P["budgets"]:
                        out.append(method_config(m, index, gate, params_version, b, P, prompts))
                else:
                    out.append(method_config(m, index, gate, params_version, None, P, prompts))
    return out
