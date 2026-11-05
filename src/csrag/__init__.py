"""Embedding, ingestion and retrieval experiments on CS-E Amendment 9 (see plan.md).

Modules: corpus (in-memory corpus for both indexes), windows (500/50 splitter),
enrich (engine_type, spans, cross-refs, ids), embed (encoder client, BM25,
embedding caches), index (Qdrant collections), runs (run records and caches),
gate (score gate), retrievers/ (all methods), evaluate (qrels, metrics, tests),
report (metrics files and Mermaid diagrams).
"""

from .corpus import COLLECTIONS, INDEXES, Corpus

__all__ = ["COLLECTIONS", "INDEXES", "Corpus"]
