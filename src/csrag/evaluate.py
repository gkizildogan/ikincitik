"""Evaluation (plan sections 2 and 6).

- Scored questions: the 92 answerable ones only. The 8 unanswerable questions run
  through every retriever but are never scored (they go to the ``unanswerable`` sheet).
- Levels: ``unit`` (the index's own unit: base unit or window; per index only),
  ``section`` (hits deduplicated by section_id, first rank kept = MaxP) and
  ``group``. Section/group metrics and evidence recall compare across indexes.
- Qrels come from character spans (``data/eval/qrels.jsonl``):
  unit level   grade 2 = the unit contains the whole quote span; grade 1 = another
               unit of a gold section. If no unit of the gold section contains the
               whole span (a quote cut by a base-unit split), the units overlapping
               it get grade 2 (``partial: true``).
  section      grade 2 = a gold section; grade 1 = a sibling in the same group.
  group        grade 2 = a gold group.
- Metrics: Hit@k, Recall@k (k = 1, 3, 5, 10, 20), MRR@10, nDCG@10 (graded,
  linear gain), R-precision; evidence recall@k (the gold quote's span is covered
  by the retrieved units' spans together) and on the whole context
  (``evidence_recall@ctx``). Gated lists are scored as returned, no padding.
- Multi-section metrics for ``requires_all_gold`` questions: supporting recall@k
  and all-gold@k, at unit and section level, reported with their n.
- Statistics: bootstrap 95% CIs (one resample matrix for all runs, so they are
  paired), paired randomisation (sign-flip) tests, Holm correction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .enrich import normalize_ws
from .runs import load_config, load_hits, load_stats

KS = (1, 3, 5, 10, 20)
LEVELS = ("unit", "section", "group")
N_BOOT = 10_000
SEED = 20261105

# ---------------------------------------------------------------- questions and qrels


def load_questions(path: str | Path, corpus) -> tuple[list[dict], dict]:
    """Questions with validated gold quotes and their normalised spans (``_gold``)."""
    qs = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    problems: dict[str, list] = {"duplicate_ids": [], "unknown_section": [], "quote_not_verbatim": [],
                                 "quote_not_in_normalised_body": [], "quote_ambiguous": [],
                                 "answerable_without_gold": [], "unanswerable_with_gold": []}
    seen = set()
    for q in qs:
        if q["question_id"] in seen:
            problems["duplicate_ids"].append(q["question_id"])
        seen.add(q["question_id"])
        if q["answerable"] and not q["gold"]:
            problems["answerable_without_gold"].append(q["question_id"])
        if not q["answerable"] and q["gold"]:
            problems["unanswerable_with_gold"].append(q["question_id"])
        q["_gold"] = []
        for g in q["gold"]:
            sid = g["section_id"]
            if sid not in corpus.sections:
                problems["unknown_section"].append((q["question_id"], sid))
                continue
            body = corpus.sections[sid]["body"]
            if g["quote"] not in body:
                problems["quote_not_verbatim"].append((q["question_id"], sid))
            nb, _ = corpus.section_norm(sid)
            nq = normalize_ws(g["quote"])
            i = nb.find(nq)
            if i < 0:
                problems["quote_not_in_normalised_body"].append((q["question_id"], sid))
                continue
            if nb.count(nq) > 1:
                problems["quote_ambiguous"].append((q["question_id"], sid, nb.count(nq)))
            q["_gold"].append({"section_id": sid, "group_id": corpus.sections[sid]["group_id"], "role": g["role"],
                               "char_start": i, "char_end": i + len(nq)})
    return qs, problems


def build_qrels(questions: list[dict], corpus) -> list[dict]:
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        qid = q["question_id"]
        for index in corpus.indexes():
            grades: dict[str, tuple[int, bool]] = {}
            for g in q["_gold"]:
                us = corpus.units_of(g["section_id"], index)
                full = [u for u in us if u["char_start"] <= g["char_start"] and u["char_end"] >= g["char_end"]]
                partial = False
                if not full:
                    full = [u for u in us if u["char_start"] < g["char_end"] and u["char_end"] > g["char_start"]]
                    partial = True
                for u in us:
                    gr = 2 if u in full else 1
                    old = grades.get(u["unit_id"], (0, False))
                    if gr > old[0]:
                        grades[u["unit_id"]] = (gr, partial and gr == 2)
            for uid, (gr, partial) in grades.items():
                rows.append({"question_id": qid, "level": "unit", "index": index, "id": uid, "grade": gr,
                             **({"partial": True} if partial else {})})
        gold_secs = list(dict.fromkeys(g["section_id"] for g in q["_gold"]))
        for sid in gold_secs:
            role = next(g["role"] for g in q["_gold"] if g["section_id"] == sid)
            rows.append({"question_id": qid, "level": "section", "index": None, "id": sid, "grade": 2, "role": role})
        for sid in gold_secs:
            for sib in corpus.groups[corpus.sections[sid]["group_id"]]["section_ids"]:
                if sib not in gold_secs and not any(r["question_id"] == qid and r["level"] == "section" and r["id"] == sib for r in rows):
                    rows.append({"question_id": qid, "level": "section", "index": None, "id": sib, "grade": 1})
        for gid in dict.fromkeys(g["group_id"] for g in q["_gold"]):
            rows.append({"question_id": qid, "level": "group", "index": None, "id": gid, "grade": 2})
    return rows


def write_qrels(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_qrels(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def qrels_lookup(rows: list[dict]) -> dict:
    """``{(level, index): {question_id: {id: grade}}}``; section/group levels use index ``None``."""
    out: dict = {}
    for r in rows:
        out.setdefault((r["level"], r["index"]), {}).setdefault(r["question_id"], {})[r["id"]] = r["grade"]
    return out


def check_qrels(rows: list[dict], questions: list[dict], corpus) -> dict:
    """No orphans: every id exists, every answerable question has grade-2 qrels at every level and index."""
    qids = {q["question_id"] for q in questions if q["answerable"]}
    probs = {"orphan_ids": [], "missing": [], "partial_unit_qrels": []}
    for r in rows:
        if r["level"] == "unit" and r["id"] not in corpus.units(r["index"]):
            probs["orphan_ids"].append(r)
        if r["level"] == "section" and r["id"] not in corpus.sections:
            probs["orphan_ids"].append(r)
        if r.get("partial"):
            probs["partial_unit_qrels"].append((r["question_id"], r["index"], r["id"]))
    lk = qrels_lookup(rows)
    keys = [("unit", i) for i in corpus.indexes()] + [("section", None), ("group", None)]
    for key in keys:
        for qid in qids:
            if not any(g >= 2 for g in lk.get(key, {}).get(qid, {}).values()):
                probs["missing"].append((key, qid))
    return probs


# ---------------------------------------------------------------- metrics of one ranked list


def dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def rank_metrics(ranked: list[str], qrel: dict[str, int], ks=KS) -> dict:
    rel = {i for i, g in qrel.items() if g >= 2}
    R = len(rel)
    out: dict = {}
    for k in ks:
        top = ranked[:k]
        n_rel = sum(1 for x in top if x in rel)
        out[f"hit@{k}"] = float(n_rel > 0)
        out[f"recall@{k}"] = n_rel / R if R else float("nan")
    first = next((i for i, x in enumerate(ranked, 1) if x in rel), None)
    out["mrr@10"] = 1.0 / first if first and first <= 10 else 0.0
    gains = [qrel.get(x, 0) for x in ranked[:10]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(qrel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    out["ndcg@10"] = dcg / idcg if idcg else float("nan")
    out["rprec"] = sum(1 for x in ranked[:R] if x in rel) / R if R else float("nan")
    out["first_gold_rank"] = float(first) if first else float("nan")
    return out


def _covered(spans: list[tuple[int, int]], a: int, b: int) -> bool:
    """Is [a, b) inside the union of ``spans``?"""
    pos = a
    for s, e in sorted(spans):
        if s > pos:
            break
        pos = max(pos, e)
        if pos >= b:
            return True
    return pos >= b


def evidence_recall(unit_ids: list[str], gold: list[dict], units: dict) -> float:
    """Share of gold quotes whose span is covered by the given units' spans together."""
    if not gold:
        return float("nan")
    by_sec: dict[str, list[tuple[int, int]]] = {}
    for uid in unit_ids:
        u = units[uid]
        by_sec.setdefault(u["section_id"], []).append((u["char_start"], u["char_end"]))
    hit = sum(_covered(by_sec.get(g["section_id"], []), g["char_start"], g["char_end"]) for g in gold)
    return hit / len(gold)


# ---------------------------------------------------------------- one run


def question_metrics(run_id: str, questions: list[dict], qrels: list[dict], corpus, root=None):
    """Per-question metrics of one run (answerable questions only), long format:
    run_id, index, question_id, level, metric, value."""
    import pandas as pd

    cfg = load_config(run_id, root)
    index = cfg["index"]
    units = corpus.units(index)
    hits = load_hits(run_id, root).sort_values(["question_id", "rank"])
    lk = qrels_lookup(qrels)
    by_q = {qid: df for qid, df in hits.groupby("question_id", sort=False)}
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        qid = q["question_id"]
        df = by_q.get(qid)
        unit_ids = list(df["unit_id"]) if df is not None else []
        in_ctx = list(df.loc[df["in_context"], "unit_id"]) if df is not None else []
        lists = {
            "unit": dedupe(unit_ids),
            "section": dedupe(units[u]["section_id"] for u in unit_ids),
            "group": dedupe(units[u]["group_id"] for u in unit_ids),
        }
        for level in LEVELS:
            qrel = lk.get((level, index if level == "unit" else None), {}).get(qid, {})
            for m, v in rank_metrics(lists[level], qrel).items():
                rows.append((run_id, index, qid, level, m, v))
        gold = q["_gold"]
        for k in KS:
            rows.append((run_id, index, qid, "evidence", f"evidence_recall@{k}", evidence_recall(lists["unit"][:k], gold, units)))
        rows.append((run_id, index, qid, "evidence", "evidence_recall@ctx", evidence_recall(in_ctx, gold, units)))
        if q.get("requires_all_gold"):
            primary = {g["section_id"] for g in gold if g["role"] == "primary"}
            support = {g["section_id"] for g in gold if g["role"] == "supporting"}
            sup_spans = [g for g in gold if g["role"] == "supporting"]
            for k in KS:
                top = set(lists["section"][:k])
                rows.append((run_id, index, qid, "multi_section", f"supporting_recall@{k}", float(support <= top) if support else float("nan")))
                rows.append((run_id, index, qid, "multi_section", f"all_gold@{k}", float((primary | support) <= top)))
                rows.append((run_id, index, qid, "multi_unit", f"supporting_recall@{k}",
                             evidence_recall(lists["unit"][:k], sup_spans, units) if sup_spans else float("nan")))
                rows.append((run_id, index, qid, "multi_unit", f"all_gold@{k}",
                             float(evidence_recall(lists["unit"][:k], gold, units) == 1.0)))
    return pd.DataFrame(rows, columns=["run_id", "index", "question_id", "level", "metric", "value"])


def run_costs(run_id: str, questions: list[dict], root=None) -> dict:
    """Mean cost per question (all questions, answerable or not, ran the same way)."""
    st = load_stats(run_id, root)
    lat = {k: float(np.mean([s["latency"].get(k, 0.0) for s in st])) for k in ("qdrant", "encoder", "llm", "rerank", "total")}
    return {
        **{f"latency_{k}_s": v for k, v in lat.items()},
        "llm_prompt_tokens": float(np.mean([s["llm"]["prompt_tokens"] for s in st])),
        "llm_completion_tokens": float(np.mean([s["llm"]["completion_tokens"] for s in st])),
        "llm_calls": float(np.mean([s["llm"]["calls"] for s in st])),
        "context_tokens": float(np.mean([s["context_tokens"] for s in st])),
        "n_hits": float(np.mean([s["n_hits"] for s in st])),
    }


def gate_diagnostics(run_id: str, questions: list[dict], root=None) -> list[dict]:
    """Per gate position (gate A per signal, gate B): kept length and gate loss.

    Gate loss = share of answerable questions whose gold section was in that
    position's gate-off list (the raw list) but was cut by the gate.
    """
    st = {s["question_id"]: s for s in load_stats(run_id, root)}
    gold = {q["question_id"]: {g["section_id"] for g in q["_gold"]} for q in questions if q["answerable"]}
    pos: dict[str, dict] = {}
    for qid, s in st.items():
        for g in s.get("gates", []):
            key = f"{g['pos']}:{g['signal']}"
            d = pos.setdefault(key, {"kept": [], "n_in": [], "loss": []})
            d["kept"].append(g["n_kept"])
            d["n_in"].append(g["n_in"])
            if qid in gold:
                kept, cut = set(g["kept_sections"]), set(g["cut_sections"])
                d["loss"].append(float(bool(gold[qid] & cut) and not (gold[qid] & kept)))
    return [
        {"run_id": run_id, "position": k, "kept_median": float(np.median(v["kept"])), "kept_mean": float(np.mean(v["kept"])),
         "in_mean": float(np.mean(v["n_in"])), "gate_loss": float(np.mean(v["loss"])) if v["loss"] else float("nan"),
         "n": len(v["loss"])}
        for k, v in sorted(pos.items())
    ]


# ---------------------------------------------------------------- statistics


def boot_index(n: int, b: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """One resample matrix for every run: the bootstrap CIs are paired across runs."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def bootstrap_ci(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    import warnings

    v = np.asarray(values, float)
    if np.isnan(v).all():
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # resamples that draw only NaNs (e.g. first_gold_rank)
        means = np.nanmean(v[idx], axis=1)
        return float(np.nanpercentile(means, 100 * alpha / 2)), float(np.nanpercentile(means, 100 * (1 - alpha / 2)))


_SIGNS: dict[tuple[int, int, int], np.ndarray] = {}


def _signs(n: int, n_perm: int, seed: int) -> np.ndarray:
    key = (n, n_perm, seed)
    if key not in _SIGNS:
        _SIGNS[key] = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_perm, n))
    return _SIGNS[key]


def paired_randomization(a: np.ndarray, b: np.ndarray, n_perm: int = N_BOOT, seed: int = SEED) -> float:
    """Two-sided paired sign-flip test of mean(a - b) = 0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    signs = _signs(len(d), n_perm, seed)
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# ---------------------------------------------------------------- pre-declared comparisons

COMPARISONS = [
    ("RQ1", "S1", "L1"), ("RQ1", "D1", "L1"), ("RQ1", "C1", "L1"),
    ("RQ2", "H1", "L1"), ("RQ2", "H1", "D1"), ("RQ2", "H2", "H1"), ("RQ2", "M1", "D1"),
    ("RQ3", "X1", "H1"), ("RQ3", "X2", "H1"), ("RQ3", "X1", "X2"),
    ("RQ4", "Q1", "D1"), ("RQ4", "Q2", "L1"),
    ("RQ5", "F1", "H1"), ("RQ5", "T1", "H1"),
    ("RQ6", "T2", "T0"), ("RQ6", "T3", "T0"), ("RQ6", "T2+T3", "T0"), ("RQ6", "T4", "T2+T3"),
]
RQ7_METHODS = ["L1", "D1", "C1", "H1", "X1"]
RQ7_LEVELS = ("section", "group", "evidence")


def run_table(run_ids: list[str], root=None):
    """One row per run: run_id, method, index, gate, params_version, budget, config_hash."""
    import pandas as pd

    rows = []
    for rid in run_ids:
        c = load_config(rid, root)
        ex = c["config"].get("expansion") or {}
        rows.append({"run_id": rid, "method": c["retriever"], "index": c["index"], "gate": c["gate"],
                     "params_version": c["params_version"], "budget": ex.get("budget"), "config_hash": c["config_hash"]})
    return pd.DataFrame(rows)


def significance(pq, runs, qids: list[str], index: str | None = None, rq7: bool = False):
    """Paired randomisation tests for the pre-declared comparisons on every metric,
    Holm-corrected within each RQ, metric (level x metric) and index (and params_version).

    RQ1-RQ6 pair runs within one index, with the same gate, params_version and
    budget. RQ7 pairs windows with units (gate off) on section/group level and
    evidence recall.
    """
    import pandas as pd

    pqi = pq.set_index(["run_id", "level", "metric", "question_id"])["value"].sort_index()

    def vec(rid, level, metric):
        try:
            s = pqi.loc[(rid, level, metric)]
        except KeyError:
            return np.full(len(qids), np.nan)
        return np.array([s.get(q, np.nan) for q in qids], float)

    metrics = pq[["level", "metric"]].drop_duplicates()
    metrics = metrics[~metrics["metric"].isin(["first_gold_rank"]) & ~metrics["level"].str.startswith("multi")]
    rows = []
    idx = boot_index(len(qids))
    if not rq7:
        r = runs[runs["index"] == index]
        for rq, a, b in COMPARISONS:
            ra, rb = r[r["method"] == a], r[r["method"] == b]
            for _, x in ra.iterrows():
                y = rb[(rb["gate"] == x["gate"]) & (rb["params_version"] == x["params_version"])
                       & ((rb["budget"] == x["budget"]) | (rb["budget"].isna() & pd.isna(x["budget"])))]
                if y.empty:
                    continue
                y = y.iloc[0]
                for _, mt in metrics.iterrows():
                    va, vb = vec(x["run_id"], mt["level"], mt["metric"]), vec(y["run_id"], mt["level"], mt["metric"])
                    rows.append(_test_row(rq, a, b, index, x, mt, va, vb, idx, x["run_id"], y["run_id"]))
    else:
        r = runs[~runs["gate"]]
        for m in RQ7_METHODS:
            for pv in sorted(r["params_version"].unique()):
                w = r[(r["method"] == m) & (r["index"] == "win500") & (r["params_version"] == pv)]
                u = r[(r["method"] == m) & (r["index"] == "units") & (r["params_version"] == pv)]
                if w.empty or u.empty:
                    continue
                x, y = w.iloc[0], u.iloc[0]
                for _, mt in metrics[metrics["level"].isin(RQ7_LEVELS)].iterrows():
                    va, vb = vec(x["run_id"], mt["level"], mt["metric"]), vec(y["run_id"], mt["level"], mt["metric"])
                    rows.append(_test_row("RQ7", f"{m}@win500", f"{m}@units", "both", x, mt, va, vb, idx, x["run_id"], y["run_id"]))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p_holm"] = np.nan
    for _, g in df.groupby(["rq", "level", "metric", "index", "params_version"]):
        ok = g["p"].notna()
        if ok.any():
            df.loc[g.index[ok], "p_holm"] = holm(list(g.loc[ok, "p"]))
    return df


def _test_row(rq, a, b, index, x, mt, va, vb, idx, rid_a, rid_b) -> dict:
    d = va - vb
    ok = ~np.isnan(d)
    lo, hi = bootstrap_ci(d, idx) if ok.any() else (np.nan, np.nan)
    return {
        "rq": rq, "a": a, "b": b, "index": index, "gate": bool(x["gate"]), "params_version": x["params_version"],
        "budget": x["budget"], "level": mt["level"], "metric": mt["metric"], "n": int(ok.sum()),
        "mean_a": float(np.nanmean(va)) if ok.any() else np.nan, "mean_b": float(np.nanmean(vb)) if ok.any() else np.nan,
        "diff": float(np.nanmean(d)) if ok.any() else np.nan, "diff_ci_lo": lo, "diff_ci_hi": hi,
        "p": paired_randomization(va[ok], vb[ok]) if ok.any() else np.nan, "run_a": rid_a, "run_b": rid_b,
    }


# ---------------------------------------------------------------- everything


def evaluate_all(run_ids: list[str], questions: list[dict], qrels: list[dict], corpus, root=None, progress=None):
    """(per-question metrics, run_metrics, gate diagnostics, costs) for all runs.

    run_metrics columns: run_id, index, level, metric, value, subset, n, ci_lo, ci_hi.
    Subsets: ``all`` and ``mentions_section_id=true|false``; every row states its n
    (92 answerable for ``all``; the 8 unanswerable questions are never scored).
    """
    import pandas as pd

    it = progress(run_ids, desc="evaluate") if progress else run_ids
    pq = pd.concat([question_metrics(r, questions, qrels, corpus, root) for r in it], ignore_index=True)
    answerable = [q for q in questions if q["answerable"]]
    qids = [q["question_id"] for q in answerable]
    subsets = {
        "all": qids,
        "mentions_section_id=true": [q["question_id"] for q in answerable if q["mentions_section_id"]],
        "mentions_section_id=false": [q["question_id"] for q in answerable if not q["mentions_section_id"]],
    }
    idx_all = boot_index(len(qids))
    rows = []
    for (rid, index, level, metric), g in pq.groupby(["run_id", "index", "level", "metric"], sort=False):
        s = g.set_index("question_id")["value"]
        for name, ids in subsets.items():
            v = np.array([s.get(q, np.nan) for q in ids], float)
            n = int((~np.isnan(v)).sum())
            val = float(np.nanmean(v)) if n else float("nan")
            lo, hi = bootstrap_ci(v, idx_all) if name == "all" and n else (float("nan"), float("nan"))
            rows.append((rid, index, level, metric, val, name, n, lo, hi))
    rm = pd.DataFrame(rows, columns=["run_id", "index", "level", "metric", "value", "subset", "n", "ci_lo", "ci_hi"])
    gates = pd.DataFrame([g for r in run_ids for g in gate_diagnostics(r, questions, root)])
    costs = pd.DataFrame([{"run_id": r, **run_costs(r, questions, root)} for r in run_ids])
    # cost and gate rows also go to run_metrics (level "cost" / "gate")
    extra = []
    for _, c in costs.iterrows():
        index = load_config(c["run_id"], root)["index"]
        for k, v in c.items():
            if k != "run_id":
                extra.append((c["run_id"], index, "cost", k, float(v), "all", len(questions), np.nan, np.nan))
    for _, g in gates.iterrows() if not gates.empty else []:
        index = load_config(g["run_id"], root)["index"]
        for k in ("kept_median", "kept_mean", "gate_loss"):
            extra.append((g["run_id"], index, "gate", f"{k}:{g['position']}", float(g[k]), "all", int(g["n"]), np.nan, np.nan))
    rm = pd.concat([rm, pd.DataFrame(extra, columns=rm.columns)], ignore_index=True)
    return pq, rm, gates, costs


def gate_loss_vs_gate_off(pq, runs):
    """Overall gate loss of each gate-on run against the gate-off run of the same
    configuration: share of questions whose gold section is in the gate-off
    run's final list but not in the gate-on run's (section level, whole list)."""
    import pandas as pd

    sec = pq[(pq["level"] == "section") & (pq["metric"] == "first_gold_rank")]
    rank = {(r, q): v for r, q, v in zip(sec["run_id"], sec["question_id"], sec["value"])}
    qids = sorted(sec["question_id"].unique())
    rows = []
    for _, on in runs[runs["gate"]].iterrows():
        off = runs[(~runs["gate"]) & (runs["method"] == on["method"]) & (runs["index"] == on["index"])
                   & (runs["params_version"] == on["params_version"])
                   & ((runs["budget"] == on["budget"]) | (runs["budget"].isna() & pd.isna(on["budget"])))]
        if off.empty:
            continue
        roff = off.iloc[0]["run_id"]
        lost = [float(not np.isnan(rank.get((roff, q), np.nan)) and np.isnan(rank.get((on["run_id"], q), np.nan))) for q in qids]
        rows.append({"run_id": on["run_id"], "gate_off_run": roff, "gate_loss_overall": float(np.mean(lost)) if lost else np.nan})
    return pd.DataFrame(rows)
