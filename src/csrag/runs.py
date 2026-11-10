"""Run records and caches (plan section 3.2). Files only, no database.

Caches (``data/cache/``, reused across runs; append-only JSONL, last write wins):
  llm.jsonl          hash of the full LLM request -> output (+ usage, seconds)
  search.jsonl       (index, query_hash, signal, filter) -> raw top-50 with scores
  rerank_bge.jsonl   (query_hash, unit_id) -> score            (pointwise)
  rerank_jina.jsonl  (query_hash, hash of ordered candidates) -> scores (listwise)

Run records (``results/<experiment>/runs/<run_id>/``; the folder comes from the
active experiment, see ``experiment.py``):
  config.json     retriever id, index, gate, full config, config_hash, params_version,
                  corpus_version, model revisions from /health, start time, notes,
                  n_questions and questions_hash (the question set the run was made with)
  hits.parquet    question_id, rank, unit_id, section_id, score, stage, hop, n_tokens, in_context,
                  source (the hop-0 unit a hop-1 item was reached from)
  stats.jsonl     per question: latency by component, LLM tokens, context tokens, gate trace
  trace.jsonl     T4 only: one line per agent step
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable, Iterable

from . import experiment

CACHE_DIR = Path(os.environ.get("CSRAG_CACHE_DIR", "data/cache"))

HIT_COLUMNS = ["question_id", "rank", "unit_id", "section_id", "score", "stage", "hop", "n_tokens", "in_context", "source"]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def sha(obj, n: int = 16) -> str:
    s = obj if isinstance(obj, str) else canonical(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


class JsonlCache:
    """Append-only key -> value store; the whole file is loaded into a dict."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, object] = {}
        self._lock = threading.Lock()
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        r = json.loads(ln)
                        self.data[r["key"]] = r["value"]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def put(self, key: str, value) -> None:
        with self._lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class Caches:
    def __init__(self, root: str | Path | None = None):
        root = Path(root) if root else CACHE_DIR
        self.llm = JsonlCache(root / "llm.jsonl")
        self.search = JsonlCache(root / "search.jsonl")
        self.rerank_bge = JsonlCache(root / "rerank_bge.jsonl")
        self.rerank_jina = JsonlCache(root / "rerank_jina.jsonl")


# ---------------------------------------------------------------- run records


def config_hash(cfg: dict) -> str:
    return sha({k: v for k, v in cfg.items() if k not in ("notes",)}, 12)


def run_id_of(cfg: dict) -> str:
    """Readable and unique: ``<method>[-b<budget>]-<index>-gate<on|off>-<params>-<hash8>``."""
    parts = [cfg["method"]]
    if cfg.get("expansion") and cfg["expansion"].get("budget"):
        parts.append(f"b{cfg['expansion']['budget'] // 1000}k")
    parts += [cfg["index"], "gateon" if cfg["gate"] else "gateoff", cfg["params_version"], config_hash(cfg)[:8]]
    return "-".join(parts).replace("+", "")


def runs_root(root: str | Path | None = None) -> Path:
    """``root`` if given, else the active experiment's ``results/<name>/runs``."""
    return Path(root) if root else experiment.active().runs_dir


def run_dir(run_id: str, root: str | Path | None = None) -> Path:
    return runs_root(root) / run_id


def is_complete(run_id: str, root: str | Path | None = None) -> bool:
    d = run_dir(run_id, root)
    return (d / "hits.parquet").is_file() and (d / "config.json").is_file()


def questions_hash(questions: Iterable[dict]) -> str:
    """Identity of a question set: ids and texts, in order."""
    return sha([[q["question_id"], q["question"]] for q in questions], 12)


def same_questions(run_id: str, questions: list[dict], root: str | Path | None = None) -> bool:
    """Was this completed run made with exactly these questions?

    Run ids do not depend on the questions, so this is what keeps one question
    set's results from being read as another's.
    """
    c = load_config(run_id, root)
    if "questions_hash" in c:
        return c["questions_hash"] == questions_hash(questions)
    return c.get("n_questions") == len(questions)  # records written before questions_hash existed


def execute(
    cfg: dict,
    retrieve: Callable[[str, str], tuple[list, dict, list]],
    questions: Iterable[dict],
    meta: dict,
    root: str | Path | None = None,
    progress: Callable | None = None,
    force: bool = False,
) -> str:
    """Run one configuration over the questions and write its record.

    ``retrieve(question_id, text) -> (hits, stats, trace)``. Questions already in
    ``partial.jsonl`` are skipped, so an interrupted run resumes; the caches make
    a full re-run cheap as well. A completed run is returned as it is, unless it
    was made with a different question set: that is an error, not a skip.
    """
    import pandas as pd

    rid = run_id_of(cfg)
    d = run_dir(rid, root)
    qs = list(questions)
    if is_complete(rid, root) and not force:
        if not same_questions(rid, qs, root):
            raise RuntimeError(
                f"{d} holds a completed run made with a different question set "
                f"({load_config(rid, root).get('n_questions')} questions, now {len(qs)}). Run ids do not depend on the "
                "questions: give the new question set its own experiment name in experiments.toml, or pass force=True "
                "to overwrite this run.")
        return rid
    d.mkdir(parents=True, exist_ok=True)
    partial = d / "partial.jsonl"
    done: dict[str, dict] = {}
    if partial.is_file() and not force:
        with open(partial, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    done[r["question_id"]] = r
    elif partial.is_file():
        partial.unlink()
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    it = progress(qs, desc=rid) if progress else qs
    for q in it:
        qid = q["question_id"]
        if qid in done:
            continue
        hits, stats, trace = retrieve(qid, q["question"])
        rec = {
            "question_id": qid,
            "hits": [h.row(qid, rank) for rank, h in enumerate(hits, 1)],
            "stats": {"question_id": qid, **stats},
            "trace": [{"question_id": qid, **t} for t in trace],
        }
        with open(partial, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        done[qid] = rec
    order = [q["question_id"] for q in qs]
    rows = [h for qid in order for h in done[qid]["hits"]]
    df = pd.DataFrame(rows, columns=HIT_COLUMNS)
    df.to_parquet(d / "hits.parquet", index=False)
    with open(d / "stats.jsonl", "w", encoding="utf-8") as f:
        for qid in order:
            f.write(json.dumps(done[qid]["stats"], ensure_ascii=False, default=float) + "\n")
    traces = [t for qid in order for t in done[qid]["trace"]]
    if traces:
        with open(d / "trace.jsonl", "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False, default=float) + "\n")
    config = {
        "run_id": rid,
        "retriever": cfg["method"],
        "index": cfg["index"],
        "gate": cfg["gate"],
        "params_version": cfg["params_version"],
        "config_hash": config_hash(cfg),
        "config": cfg,
        "corpus_version": meta.get("corpus_version"),
        "models": meta.get("models"),
        "started": started,
        "finished": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_questions": len(order),
        "questions_hash": questions_hash(qs),
        "notes": cfg.get("notes"),
    }
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    partial.unlink(missing_ok=True)
    return rid


def load_config(run_id: str, root: str | Path | None = None) -> dict:
    with open(run_dir(run_id, root) / "config.json", encoding="utf-8") as f:
        return json.load(f)


def load_hits(run_id: str, root: str | Path | None = None):
    import pandas as pd

    return pd.read_parquet(run_dir(run_id, root) / "hits.parquet")


def load_stats(run_id: str, root: str | Path | None = None) -> list[dict]:
    with open(run_dir(run_id, root) / "stats.jsonl", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_trace(run_id: str, root: str | Path | None = None) -> list[dict]:
    p = run_dir(run_id, root) / "trace.jsonl"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def list_runs(root: str | Path | None = None, params_version: str | None = None) -> list[str]:
    base = runs_root(root)
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if (d / "config.json").is_file() and (d / "hits.parquet").is_file():
            if params_version is None or load_config(d.name, root)["params_version"] == params_version:
                out.append(d.name)
    return out
