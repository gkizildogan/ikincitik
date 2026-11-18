"""The final pipeline X3 (x2_to_x3_plan.md): the logic behind notebook 10.

Kept apart from ``evaluate.py`` and ``report.py`` so that they keep reproducing the run 1
archive. Nothing here calls a model or a service: every function reads run records,
caches and the corpus.

- ``find_run``, ``latency``, ``ablation_config``: the stored baselines and their cost; the optional X3 variants.
- ``replay_cut``: apply a cut-off to a stored run's list (X2: its stored hits; X3: ``stats["reranked"]``).
- ``metrics_of``, ``compare``, ``cut_sweep``, ``other_type_at_rank1``, ``evidence_losses``: the X2 vs X3 tables.
- ``write_metric_files``: the files notebook 09 writes, for the given runs only (X3 is measured like every other run).
- ``keyword_report``, ``keyword_summary``: the keyword rule against the labels (labels are used for diagnosis only).
- ``check_cut``, ``check_filter``, ``check_targets``, ``check_metric_files``, ``check_contexts``: notebook 10's checks;
  each returns a list of problems.
- ``write_contexts``, ``read_contexts``: the hand-off file for the generation notebooks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import report as rp
from . import runs
from .engine_rule import detect
from .gate import score_cut
from .retrievers.base import method_config, revise
from .retrievers.fusion import rrf
from .retrievers.qdrant_search import engine_filter

METRICS = ["hit@1", "hit@3", "mrr@10", "ndcg@10", "recall@10", "evidence_ctx"]  # section level; evidence on the whole context
CI_METRICS = ["hit@1", "hit@3", "mrr@10", "evidence_ctx"]
UNCUT = {"min_score": -math.inf, "min_keep": 1, "max_keep": 10**6}  # replay_cut(...) keeps the whole reranked list
ABLATION_LABELS = {"no_filter": "X3, no keyword filter", "rerank_input_50": "X3, 50 candidates to jina"}  # optional real runs


# ---------------------------------------------------------------- runs and replays


def find_run(method: str, index: str = "win500", gate: bool = False, params_version: str = "v1", root=None) -> str:
    """The one stored run of ``method`` with this index, gate and parameter version (no expansion budget)."""
    rt = ev.run_table(runs.list_runs(root, params_version=params_version), root)
    if rt.empty:
        raise LookupError(f"no {params_version} runs in {runs.runs_root(root)}")
    sel = rt[(rt["method"] == method) & (rt["index"] == index) & (rt["gate"] == gate) & rt["budget"].isna()]
    if len(sel) != 1:
        raise LookupError(f"{len(sel)} runs of {method} / {index} / gate {'on' if gate else 'off'} / {params_version} in {runs.runs_root(root)}, expected 1")
    return str(sel["run_id"].iloc[0])


def ablation_config(name: str, index: str, params_version: str, engine_rules_hash: str) -> dict:
    """An X3 variant that is run for real, as its own run record (a different config hash, so a different run id):
    ``no_filter`` (the keyword filter off) or ``rerank_input_50`` (50 candidates to jina, as in X2)."""
    if name == "no_filter":
        cfg = method_config("X3", index, False, params_version, engine_rules_hash=engine_rules_hash)
        cfg["filter"] = None
        cfg["notes"] = "ablation: X3 without the engine-type filter"
    elif name == "rerank_input_50":
        cfg = method_config("X3", index, False, params_version, params=revise(params_version, rerank_input=50),
                            engine_rules_hash=engine_rules_hash)
        cfg["notes"] = "ablation: X3 with 50 candidates to jina"
    else:
        raise ValueError(f"unknown ablation {name!r}; defined: {sorted(ABLATION_LABELS)}")
    return cfg


def latency(run_id: str, root=None) -> dict:
    """Mean seconds per question that the run recorded: the reranker and the total."""
    c = ev.run_costs(run_id, [], root)
    return {"rerank_s": c["latency_rerank_s"], "total_s": c["latency_total_s"]}


def replay_cut(run_id: str, corpus, min_score: float = 0.0, min_keep: int = 1, max_keep: int = 8, root=None) -> pd.DataFrame:
    """A cut-off applied to a stored run, as a hits frame (``runs.HIT_COLUMNS``, all ``in_context``).

    The source list is ``stats["reranked"]`` when the run has one (X3: the whole jina ordering of its
    candidates), else the run's stored hits (X2: its top 20). ``UNCUT`` keeps the source list whole;
    ``min_score=-inf, min_keep=max_keep=8`` is a plain top 8."""
    cfg = runs.load_config(run_id, root)
    units = corpus.units(cfg["index"])
    hits = runs.load_hits(run_id, root).sort_values(["question_id", "rank"])
    by_q = {qid: df for qid, df in hits.groupby("question_id", sort=False)}
    rows = []
    for st in runs.load_stats(run_id, root):
        qid = st["question_id"]
        if "reranked" in st:
            lst = [(u, float(s)) for u, s in st["reranked"]]
        else:
            df = by_q.get(qid)
            lst = [] if df is None else [(u, float(s)) for u, s in zip(df["unit_id"], df["score"])]
        n = score_cut([s for _, s in lst], min_score, min_keep, max_keep)
        for rank, (u, s) in enumerate(lst[:n], 1):
            rows.append({"question_id": qid, "rank": rank, "unit_id": u, "section_id": units[u]["section_id"], "score": s,
                         "stage": "reranked", "hop": 0, "n_tokens": int(units[u]["n_tokens_bge"]), "in_context": True, "source": None})
    return pd.DataFrame(rows, columns=runs.HIT_COLUMNS)


# ---------------------------------------------------------------- per-question metrics and tables


def metrics_of(hits: pd.DataFrame, questions: list[dict], qrels: list[dict], corpus, index: str = "win500") -> pd.DataFrame:
    """One row per answerable question: section-level Hit@1, Hit@3, MRR@10, nDCG@10, Recall@10 (``ev.rank_metrics``),
    ``evidence_ctx`` (``ev.evidence_recall`` over the in-context windows), ``windows``, ``tokens`` (after merging
    overlaps) and ``precision`` (share of the returned windows that lie in a gold section; NaN without windows)."""
    units = corpus.units(index)
    section_qrels = ev.qrels_lookup(qrels).get(("section", None), {})
    by_q = {qid: df.sort_values("rank") for qid, df in hits.groupby("question_id", sort=False)}
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        qid = q["question_id"]
        df = by_q.get(qid)
        ids = list(df["unit_id"]) if df is not None else []
        ctx = list(df.loc[df["in_context"], "unit_id"]) if df is not None else []
        m = ev.rank_metrics(ev.dedupe(units[u]["section_id"] for u in ids), section_qrels.get(qid, {}))
        gold = {g["section_id"] for g in q["_gold"]}
        rows.append({
            "question_id": qid, "family": ev.family_of(q),
            "hit@1": m["hit@1"], "hit@3": m["hit@3"], "mrr@10": m["mrr@10"], "ndcg@10": m["ndcg@10"], "recall@10": m["recall@10"],
            "evidence_ctx": ev.evidence_recall(ctx, q["_gold"], units),
            "windows": len(ctx), "tokens": corpus.context_tokens(ctx, index) if ctx else 0,
            "precision": sum(units[u]["section_id"] in gold for u in ctx) / len(ctx) if ctx else float("nan"),
        })
    return pd.DataFrame(rows)


def compare(rows: dict[str, pd.DataFrame], pairs: list[tuple[str, str]], lat: dict[str, dict] | None = None):
    """The X2 vs X3 tables from per-question frames (``metrics_of``) of the same questions.

    ``rows`` maps a row label to its frame; ``pairs`` are (a, b) labels for the paired differences a - b;
    ``lat`` maps a label to ``latency(...)``. Returns ``(table, by_family, paired)``:

    - ``table``: one row per label: the metrics, 95% bootstrap CIs for Hit@1, Hit@3, MRR@10 and evidence in
      context, windows (mean, median), context tokens, context precision, reranker and total seconds;
    - ``by_family``: Hit@1, Hit@3, evidence in context and windows per label and family;
    - ``paired``: a - b per metric with a bootstrap CI, the paired randomisation p-value and its Holm
      correction within each pair (all rows share one resample matrix, as in ``evaluate_all``).
    """
    labels = list(rows)
    qids = list(rows[labels[0]]["question_id"])
    for lb in labels:
        if list(rows[lb]["question_id"]) != qids:
            raise ValueError(f"row {lb!r} does not cover the same questions in the same order")
    idx = ev.boot_index(len(qids))
    table, fam = [], []
    for lb, df in rows.items():
        r: dict = {"row": lb, "n": len(df)}
        for m in METRICS:
            r[m] = float(df[m].mean())
        for m in CI_METRICS:
            r[f"{m}_ci_lo"], r[f"{m}_ci_hi"] = ev.bootstrap_ci(df[m].to_numpy(float), idx)
        r.update(windows_mean=float(df["windows"].mean()), windows_median=float(df["windows"].median()),
                 tokens_mean=float(df["tokens"].mean()), precision=float(df["precision"].mean()), **(lat or {}).get(lb, {}))
        table.append(r)
        for f, g in df.groupby("family", sort=False):
            fam.append({"row": lb, "family": f, "n": len(g), "hit@1": g["hit@1"].mean(), "hit@3": g["hit@3"].mean(),
                        "evidence_ctx": g["evidence_ctx"].mean(), "windows_mean": g["windows"].mean()})
    paired = []
    for a, b in pairs:
        part = []
        for m in METRICS:
            va, vb = rows[a][m].to_numpy(float), rows[b][m].to_numpy(float)
            lo, hi = ev.bootstrap_ci(va - vb, idx)
            part.append({"a": a, "b": b, "metric": m, "mean_a": float(np.nanmean(va)), "mean_b": float(np.nanmean(vb)),
                         "diff": float(np.nanmean(va - vb)), "diff_ci_lo": lo, "diff_ci_hi": hi, "p": ev.paired_randomization(va, vb)})
        for r, adj in zip(part, ev.holm([r["p"] for r in part])):
            r["p_holm"] = adj
        paired += part
    return pd.DataFrame(table), pd.DataFrame(fam), pd.DataFrame(paired)


def cut_sweep(run_id: str, questions: list[dict], qrels: list[dict], corpus, min_scores, max_keeps, min_keep: int = 1,
              root=None) -> pd.DataFrame:
    """Other cut-offs on the run's own ``stats["reranked"]`` (no reranker call): per ``min_score`` and
    ``max_keep``, the metrics of ``metrics_of`` and the cut loss = share of questions whose evidence in
    context is lower than that of the whole uncut ordering."""
    index = runs.load_config(run_id, root)["index"]
    whole = metrics_of(replay_cut(run_id, corpus, root=root, **UNCUT), questions, qrels, corpus, index)
    out = []
    for ms in min_scores:
        for mk in max_keeps:
            m = metrics_of(replay_cut(run_id, corpus, ms, min_keep, mk, root), questions, qrels, corpus, index)
            lost = (m["evidence_ctx"].to_numpy(float) < whole["evidence_ctx"].to_numpy(float) - 1e-9).astype(float)
            for fam in ["all"] + list(dict.fromkeys(m["family"])):
                sel = np.ones(len(m), bool) if fam == "all" else (m["family"] == fam).to_numpy()
                g = m[sel]
                out.append({"min_score": ms, "max_keep": mk, "family": fam, "n": int(sel.sum()), "hit@1": g["hit@1"].mean(),
                            "hit@3": g["hit@3"].mean(), "evidence_ctx": g["evidence_ctx"].mean(), "windows_mean": g["windows"].mean(),
                            "tokens_mean": g["tokens"].mean(), "cut_loss": lost[sel].mean()})
    return pd.DataFrame(out)


def other_type_at_rank1(hits: pd.DataFrame, questions: list[dict], corpus, index: str = "win500", family: str = "twin_topic") -> dict:
    """Among the answerable questions of ``family`` with a piston or turbine label, how many have a hit of the
    other engine type at rank 1 (Subpart A, ``all``, does not count)? Labels are used here for diagnosis only."""
    units = corpus.units(index)
    top = hits[hits["rank"] == 1].set_index("question_id")["unit_id"]
    n = other = 0
    for q in questions:
        if not q["answerable"] or ev.family_of(q) != family or q["engine_type"] not in ("piston", "turbine") or q["question_id"] not in top.index:
            continue
        n += 1
        other += units[top[q["question_id"]]]["engine_type"] not in ("all", q["engine_type"])
    return {"family": family, "n": n, "other_type_at_rank1": other, "share": other / n if n else float("nan")}


def evidence_losses(x3_id: str, x2_hits: pd.DataFrame, questions: list[dict], qrels: list[dict], corpus, root=None) -> pd.DataFrame:
    """Answerable questions where X3's context has less evidence than the X2 hits frame, each with its cause:

    - ``filter``: the keyword rule removed a gold section (a detected type that is not the section's type);
    - ``candidates``: the 30 candidates sent to jina do not hold the evidence (30 instead of 50, or the fused order);
    - ``cut``: the candidates hold it, and the cut-off dropped it.
    """
    cfg = runs.load_config(x3_id, root)
    index = cfg["index"]
    units = corpus.units(index)
    p3 = metrics_of(runs.load_hits(x3_id, root), questions, qrels, corpus, index).set_index("question_id")
    p2 = metrics_of(x2_hits, questions, qrels, corpus, index).set_index("question_id")
    st = {s["question_id"]: s for s in runs.load_stats(x3_id, root)}
    rows = []
    for q in questions:
        qid = q["question_id"]
        if not q["answerable"] or p3.loc[qid, "evidence_ctx"] >= p2.loc[qid, "evidence_ctx"] - 1e-9:
            continue
        engine = (st[qid].get("filter") or {}).get("engine_type")
        gold_types = {corpus.units_of(g["section_id"], index)[0]["engine_type"] for g in q["_gold"]}
        pool = ev.evidence_recall([u for u, _ in st[qid]["reranked"]], q["_gold"], units)
        if engine and any(t not in (engine, "all") for t in gold_types):
            cause = "filter"
        elif pool < p2.loc[qid, "evidence_ctx"] - 1e-9:
            cause = "candidates"
        else:
            cause = "cut"
        rows.append({"question_id": qid, "family": ev.family_of(q), "evidence_x2": p2.loc[qid, "evidence_ctx"],
                     "evidence_x3": p3.loc[qid, "evidence_ctx"], "evidence_in_candidates": pool, "detected": engine, "cause": cause,
                     "question": q["question"]})
    return pd.DataFrame(rows, columns=["question_id", "family", "evidence_x2", "evidence_x3", "evidence_in_candidates", "detected",
                                       "cause", "question"])


def unanswerable_view(run_id: str, questions: list[dict], corpus, root=None) -> pd.DataFrame:
    """The unanswerable questions: detected engine type, windows kept, top jina score and the kept sections (not scored)."""
    index = runs.load_config(run_id, root)["index"]
    units = corpus.units(index)
    hits = runs.load_hits(run_id, root).sort_values(["question_id", "rank"])
    st = {s["question_id"]: s for s in runs.load_stats(run_id, root)}
    rows = []
    for q in questions:
        if q["answerable"]:
            continue
        h = hits[hits["question_id"] == q["question_id"]]
        rows.append({"question_id": q["question_id"], "question": q["question"],
                     "detected": (st[q["question_id"]].get("filter") or {}).get("engine_type"), "windows": len(h),
                     "top1_score": float(h["score"].iloc[0]) if len(h) else float("nan"),
                     "sections": " | ".join(dict.fromkeys(units[u]["section_id"] for u in h["unit_id"]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the standard metric files


def write_metric_files(run_ids: list[str], questions: list[dict], qrels: list[dict], corpus, out_dir: str | Path,
                       index: str = "win500", root=None, progress=None) -> dict:
    """The files notebook 09 writes, for these runs only, into ``out_dir``: ``run_metrics.parquet``,
    ``question_metrics.parquet`` and ``<index>/metrics.{csv,xlsx}`` (no significance table). Same code as
    notebook 09 (``ev.evaluate_all``, ``rp.write_index_files``), so X3 is measured exactly like every other run."""
    out = Path(out_dir)
    rt = ev.run_table(run_ids, root)
    other = rt[rt["index"] != index]
    if len(other):
        raise ValueError(f"write_metric_files({index!r}): runs of another index: {list(other['run_id'])}")
    stale = [r for r in run_ids if not runs.same_questions(r, questions, root)]
    if stale:
        raise ValueError(f"runs made with another question set: {stale}")
    pq, rm, gates, costs = ev.evaluate_all(run_ids, questions, qrels, corpus, root=root, progress=progress)
    gate_overall = ev.gate_loss_vs_gate_off(pq, rt)
    out.mkdir(parents=True, exist_ok=True)
    rm.to_parquet(out / "run_metrics.parquet", index=False)
    pq.to_parquet(out / "question_metrics.parquet", index=False)
    n_answerable = sum(bool(q["answerable"]) for q in questions)
    scored = rm[(rm["subset"] == "all") & rm["level"].isin(["unit", "section", "group", "evidence"]) & (rm["metric"] != "first_gold_rank")]
    if not (scored["n"] == n_answerable).all():
        raise AssertionError("every scored metric row must cover every answerable question")
    files = rp.write_index_files(index, rm, pq, rt, gates, costs, gate_overall, None, questions, corpus, out_root=out, root=root)
    return {"run_table": rt, "question_metrics": pq, "run_metrics": rm, "gates": gates, "costs": costs, "files": files}


# ---------------------------------------------------------------- the keyword rule against the labels


def keyword_report(questions: list[dict], rules: dict, corpus, index: str = "win500") -> pd.DataFrame:
    """One row per question: the label, what the rule detected (and why), and the gold sections that the
    detected filter would remove (``excluded_gold``). Labels and gold are used here for diagnosis only."""
    rows = []
    for q in questions:
        d = detect(q["question"], rules)
        types = {g["section_id"]: corpus.units_of(g["section_id"], index)[0]["engine_type"] for g in q.get("_gold", [])}
        gone = [s for s, t in types.items() if d.engine_type and t not in (d.engine_type, "all")]
        rows.append({"question_id": q["question_id"], "family": ev.family_of(q), "answerable": bool(q["answerable"]),
                     "label": q["engine_type"], "detected": d.engine_type or "none", "reason": d.reason,
                     "matched": "; ".join(f"{t}: {', '.join(k)}" for t, k in d.matched.items()),
                     "gold_engine_types": ", ".join(sorted(set(types.values()))), "excluded_gold": "; ".join(gone),
                     "n_excluded_gold": len(gone), "question": q["question"]})
    return pd.DataFrame(rows)


def keyword_summary(report: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Label against detected type (counts), and the counts of ``one_type`` / ``no_cue`` / ``both_types``."""
    return pd.crosstab(report["label"], report["detected"], margins=True), report["reason"].value_counts()


# ---------------------------------------------------------------- checks (each returns a list of problems)


def check_cut(run_id: str, root=None) -> list[str]:
    """Every list has between ``min_keep`` and ``max_keep`` windows, scores descend, the kept list is a prefix of
    ``stats["reranked"]``, every kept window beyond ``min_keep`` has score >= ``min_score``, and the cut is no
    shorter than the rule allows (the next window scores below ``min_score``, or the list ended)."""
    c = runs.load_config(run_id, root)["config"]["cut"]
    hits = runs.load_hits(run_id, root).sort_values(["question_id", "rank"])
    by_q = {qid: df for qid, df in hits.groupby("question_id", sort=False)}
    bad = []
    for st in runs.load_stats(run_id, root):
        qid, rr = st["question_id"], st["reranked"]
        df = by_q.get(qid)
        kept = [] if df is None else list(zip(df["unit_id"], df["score"]))
        n = len(kept)
        if not (min(c["min_keep"], len(rr)) <= n <= min(c["max_keep"], len(rr))):
            bad.append(f"{qid}: {n} windows kept (min_keep {c['min_keep']}, max_keep {c['max_keep']}, {len(rr)} reranked)")
        if df is not None and not df["in_context"].all():
            bad.append(f"{qid}: a kept window is not in_context")
        if [u for u, _ in kept] != [u for u, _ in rr[:n]] or not np.allclose([s for _, s in kept], [s for _, s in rr[:n]], atol=1e-9):
            bad.append(f"{qid}: the kept list is not a prefix of stats['reranked']")
        if any(a < b - 1e-12 for (_, a), (_, b) in zip(kept, kept[1:])):
            bad.append(f"{qid}: scores do not descend")
        if any(s < c["min_score"] for _, s in kept[c["min_keep"]:]):
            bad.append(f"{qid}: a window beyond min_keep scores below min_score")
        if n < c["max_keep"] and n < len(rr) and rr[n][1] >= c["min_score"]:
            bad.append(f"{qid}: the cut is shorter than the rule allows")
        if st["cut"]["n_kept"] != n or st["n_hits"] != n:
            bad.append(f"{qid}: stats disagree with hits.parquet")
    return bad


def check_filter(run_id: str, questions: list[dict], corpus, lists_for, root=None) -> list[str]:
    """When a type was detected, every candidate has ``engine_type`` in {type, all}; the Qdrant filter recorded is the one
    for that type (none when no type was detected). The candidates are exactly the top ``rerank_input`` of the RRF of the
    cached first-stage lists for that filter, so with no type detected they come from the cached unfiltered lists.

    ``lists_for(text, filter) -> [bm25 list, dense list]`` returns the cached lists as ``[(unit_id, score), ...]``."""
    c = runs.load_config(run_id, root)["config"]
    units = corpus.units(c["index"])
    text = {q["question_id"]: q["question"] for q in questions}
    bad = []
    for st in runs.load_stats(run_id, root):
        qid = st["question_id"]
        engine = st["filter"]["engine_type"]
        flt = engine_filter(engine)
        ids = [u for u, _ in st["reranked"]]
        if st["filter"]["qdrant_filter"] != flt:
            bad.append(f"{qid}: recorded filter {st['filter']['qdrant_filter']} != {flt}")
        if engine and any(units[u]["engine_type"] not in (engine, "all") for u in ids):
            bad.append(f"{qid}: a candidate lies outside {engine} / all")
        fused = [u for u, _ in rrf(lists_for(text[qid], flt), c["fusion"]["k"])[: c["rerank"]["input"]]]
        if sorted(fused) != sorted(ids):
            bad.append(f"{qid}: the candidates differ from the fused top {c['rerank']['input']} of the cached lists")
    return bad


def check_targets(rows: dict[str, pd.DataFrame], x3: str, hit3_reference: str, hit1: float = 0.90, evidence: float = 0.95) -> list[str]:
    """Overall Hit@1 >= ``hit1``, evidence in context >= ``evidence``, Hit@3 not below the reference row. A failure
    names the families, with their values."""
    df = rows[x3]
    bad = []
    for metric, floor in (("hit@1", hit1), ("evidence_ctx", evidence)):
        if df[metric].mean() < floor:
            by = df.groupby("family", sort=False)[metric].mean().round(3).to_dict()
            bad.append(f"{metric} {df[metric].mean():.3f} < {floor} (by family: {by})")
    ref = rows[hit3_reference]["hit@3"].mean()
    if df["hit@3"].mean() < ref - 1e-12:
        by = {f: (round(float(g["hit@3"].mean()), 3), round(float(rows[hit3_reference][rows[hit3_reference]["family"] == f]["hit@3"].mean()), 3))
              for f, g in df.groupby("family", sort=False)}
        bad.append(f"hit@3 {df['hit@3'].mean():.3f} < {hit3_reference} {ref:.3f} (family: X3, reference: {by})")
    return bad


def check_metric_files(out_dir: str | Path, run_ids: list[str], baseline_ids: list[str], main_dir: str | Path | None = None,
                       index: str = "win500") -> list[str]:
    """``<out_dir>/<index>/metrics.csv`` has one row per evaluated run; when the main folder has a ``metrics.csv`` too
    (notebook 09 ran), the baseline rows equal its rows (same code and seeds, so identical)."""
    bad = []
    t = pd.read_csv(Path(out_dir) / index / "metrics.csv", encoding="utf-8")
    if sorted(t["run_id"]) != sorted(run_ids):
        bad.append(f"metrics.csv has runs {sorted(t['run_id'])}, expected {sorted(run_ids)}")
    main = Path(main_dir) / index / "metrics.csv" if main_dir else None
    if main is not None and main.is_file():
        m = pd.read_csv(main, encoding="utf-8")
        cols = [c for c in t.columns if c in m.columns]
        for rid in baseline_ids:
            a, b = t[t["run_id"] == rid], m[m["run_id"] == rid]
            if len(a) != 1 or len(b) != 1:
                bad.append(f"{rid}: not found exactly once in both tables")
                continue
            for c in cols:
                x, y = a[c].iloc[0], b[c].iloc[0]
                same = (np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True) if isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                        else (x == y or (pd.isna(x) and pd.isna(y))))
                if not same:
                    bad.append(f"{rid}: column {c}: {x} != {y}")
    return bad


# ---------------------------------------------------------------- the hand-off file


def write_contexts(run_id: str, questions: list[dict], corpus, path: str | Path, root=None) -> Path:
    """``contexts.jsonl``: one line per question, for the generation notebooks.

    ``hits`` is what X3 hands over (the windows with their text); ``reranked`` is every candidate with its jina
    score, best first, without text (the text of a window is ``corpus.units(index)[unit_id]["text"]``). The first
    ``len(hits)`` entries of ``reranked`` are ``hits``. Gold answers stay in the question files."""
    if not runs.same_questions(run_id, questions, root):
        raise ValueError(f"{run_id} was made with another question set")
    index = runs.load_config(run_id, root)["index"]
    units = corpus.units(index)
    hits = runs.load_hits(run_id, root).sort_values(["question_id", "rank"])
    by_q = {qid: df for qid, df in hits.groupby("question_id", sort=False)}
    st = {s["question_id"]: s for s in runs.load_stats(run_id, root)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            qid = q["question_id"]
            df = by_q.get(qid)
            rec = {
                "question_id": qid, "question": q["question"], "family": ev.family_of(q), "answerable": bool(q["answerable"]),
                "run_id": run_id, "engine_type_detected": (st[qid].get("filter") or {}).get("engine_type"),
                "context_tokens": int(st[qid]["context_tokens"]),
                "hits": [] if df is None else [
                    {"rank": int(r.rank), "unit_id": r.unit_id, "section_id": r.section_id, "score": float(r.score),
                     "n_tokens": int(r.n_tokens), "breadcrumb": units[r.unit_id]["breadcrumb"],
                     "page_start": units[r.unit_id]["page_start"], "page_end": units[r.unit_id]["page_end"],
                     "text": units[r.unit_id]["text"]} for r in df.itertuples()],
                "reranked": [[u, float(s)] for u, s in st[qid]["reranked"]],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def read_contexts(path: str | Path, question_ids: list[str] | None = None) -> list[dict]:
    """The records of ``contexts.jsonl``; with ``question_ids``, only those, in that order (KeyError for an unknown id)."""
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    if question_ids is None:
        return recs
    by_id = {r["question_id"]: r for r in recs}
    missing = [i for i in question_ids if i not in by_id]
    if missing:
        raise KeyError(f"question ids not in {path}: {missing}")
    return [by_id[i] for i in question_ids]


def check_contexts(path: str | Path, run_id: str, questions: list[dict], corpus, root=None) -> list[str]:
    """One line per question; each ``hits`` list matches the run's ``hits.parquet`` (ids, order, scores) and the corpus texts;
    ``reranked`` equals ``stats["reranked"]`` and starts with the unit ids of ``hits``."""
    index = runs.load_config(run_id, root)["index"]
    units = corpus.units(index)
    hits = runs.load_hits(run_id, root).sort_values(["question_id", "rank"])
    by_q = {qid: df for qid, df in hits.groupby("question_id", sort=False)}
    st = {s["question_id"]: s for s in runs.load_stats(run_id, root)}
    recs = read_contexts(path)
    bad = []
    if [r["question_id"] for r in recs] != [q["question_id"] for q in questions]:
        bad.append("the questions of the file are not the experiment's questions, in order")
    for r in recs:
        qid = r["question_id"]
        df = by_q.get(qid)
        ids = [] if df is None else list(df["unit_id"])
        if [h["unit_id"] for h in r["hits"]] != ids:
            bad.append(f"{qid}: hits differ from hits.parquet")
        if any(h["text"] != units[h["unit_id"]]["text"] for h in r["hits"]):
            bad.append(f"{qid}: a text differs from the corpus")
        if r["reranked"] != [[u, float(s)] for u, s in st[qid]["reranked"]]:
            bad.append(f"{qid}: reranked differs from stats.jsonl")
        if [u for u, _ in r["reranked"]][: len(ids)] != ids:
            bad.append(f"{qid}: reranked does not start with hits")
    return bad
