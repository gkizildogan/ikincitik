"""Outputs (plan section 6.1) and pipeline diagrams (6.2).

- ``results/<index>/metrics.csv``: one row per run (UTF-8, one header row)
- ``results/<index>/metrics.xlsx``: the same table + breakdowns, significance,
  gate, per_question and unanswerable sheets
- ``results/granularity.csv``: the RQ7 comparisons (windows vs units)
- ``results/diagrams/<run_id>.md``: a Mermaid flowchart generated from the run's
  stored config, with its metric panel

Tables are sorted by section-level nDCG@10 for readability only; no metric is
used to pick a winner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .corpus import COLLECTIONS
from .runs import load_config, load_hits, load_stats

SORT_KEY = "section_ndcg@10"
KEY_METRICS = ["section_ndcg@10", "section_recall@10", "section_mrr@10", "section_hit@1", "evidence_recall@10",
               "evidence_recall@ctx", "unit_ndcg@10", "group_ndcg@10"]


# ---------------------------------------------------------------- readable config


def _gate_txt(g: dict | None, name: str) -> str:
    return f"{name} τ{g['tau']} δ{g['delta']} keep {g['min_keep']}–{g['max_keep']}" if g else ""


def readable_stages(cfg: dict) -> dict:
    qt = cfg.get("query_transform")
    flt = cfg.get("filter")
    ex = cfg.get("expansion")
    if cfg.get("colbert_rerank"):
        rerank = f"ColBERT MaxSim (dense candidates, {cfg.get('colbert_engine', 'client')})"
    elif cfg.get("rerank"):
        rerank = {"bge": "bge-reranker-v2-m3", "jina": "jina-reranker-v3"}[cfg["rerank"]["model"]] + f" (top {cfg['rerank']['input']})"
    else:
        rerank = "-"
    if ex:
        rule = {"none": "none", "parent": f"parent {ex.get('to', 'section')}", "refs": "cross-refs (1 hop)",
                "parent+refs": f"parent {ex.get('to', 'section')} + cross-refs", "agent": f"agent (≤{ex.get('step_limit')} steps)"}[ex["rule"]]
        expansion = f"{rule}, budget {ex['budget']}"
    else:
        expansion = "-"
    gates = [x for x in (_gate_txt(cfg.get("gate_a"), "A"), _gate_txt(cfg.get("gate_b"), "B")) if x]
    return {
        "query_transform": {None: "-", "hyde": "HyDE (1 passage)", "keywords": "LLM keywords"}[qt["type"] if qt else None],
        "first_stages": " + ".join(cfg["first_stages"]),
        "gate_config": "; ".join(gates) if gates else "off",
        "fusion": f"RRF k={cfg['fusion']['k']}" if cfg.get("fusion") else "-",
        "filter": f"engine_type ({flt['type']})" if flt else "-",
        "rerank": rerank,
        "lookup": "section-id regex" if cfg.get("lookup") else "-",
        "expansion": expansion,
    }


# ---------------------------------------------------------------- tables


def metrics_table(rm, runs, gates, costs, gate_overall, index: str):
    """One row per run of one index, sorted by section nDCG@10."""
    import pandas as pd

    r = runs[runs["index"] == index].copy()
    allm = rm[(rm["subset"] == "all") & rm["level"].isin(["unit", "section", "group", "evidence"])].copy()
    allm["col"] = np.where(allm["level"] == "evidence", allm["metric"], allm["level"] + "_" + allm["metric"])
    allm = allm[~allm["metric"].eq("first_gold_rank")]
    wide = allm.pivot_table(index="run_id", columns="col", values="value", aggfunc="first")
    ci = rm[(rm["subset"] == "all") & (rm["level"] == "section") & (rm["metric"] == "ndcg@10")].set_index("run_id")
    wide["section_ndcg@10_ci_lo"] = ci["ci_lo"]
    wide["section_ndcg@10_ci_hi"] = ci["ci_hi"]
    n = rm[(rm["subset"] == "all") & (rm["level"] == "section") & (rm["metric"] == "ndcg@10")].set_index("run_id")["n"]
    wide["n_scored"] = n
    # multi-section metrics with their n
    multi = rm[(rm["subset"] == "all") & rm["level"].str.startswith("multi")].copy()
    if not multi.empty:
        multi["col"] = multi["level"] + "_" + multi["metric"]
        wide = wide.join(multi.pivot_table(index="run_id", columns="col", values="value", aggfunc="first"))
        wide = wide.join(multi.groupby("run_id")["n"].max().rename("multi_n"))
    else:
        wide["multi_n"] = 0
    # gate diagnostics, summarised per gate position type
    if not gates.empty:
        g = gates.copy()
        g["pos"] = g["position"].str.split(":").str[0]
        gs = g.groupby(["run_id", "pos"]).agg(kept_mean=("kept_mean", "mean"), kept_median=("kept_median", "median"),
                                              gate_loss=("gate_loss", "max")).unstack("pos")
        gs.columns = [f"gate{p}_{m}" for m, p in gs.columns]
        wide = wide.join(gs)
    if gate_overall is not None and not gate_overall.empty:
        wide = wide.join(gate_overall.set_index("run_id")[["gate_loss_overall"]])
    wide = wide.join(costs.set_index("run_id"))
    stages = pd.DataFrame([{"run_id": rid, **readable_stages(load_config(rid)["config"])} for rid in r["run_id"]]).set_index("run_id")
    out = r.set_index("run_id")[["config_hash", "params_version", "index", "gate", "method", "budget"]].join(stages).join(wide)
    out = out.reset_index().sort_values(SORT_KEY, ascending=False, na_position="last")
    out.insert(0, "n_note", "n = 92 answerable; 8 unanswerable excluded")
    return out


def _question_frame(questions, corpus):
    import pandas as pd

    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        prim = next((g for g in q["_gold"] if g["role"] == "primary"), q["_gold"][0])
        n = corpus.n_tokens(corpus.sections[prim["section_id"]]["body"])
        rows.append({
            "question_id": q["question_id"], "question_type": q["question_type"], "engine_type": q["engine_type"],
            "subpart": q.get("subpart"), "kind": q.get("gold_kind") or corpus.sections[prim["section_id"]]["kind"],
            "mentions_section_id": bool(q["mentions_section_id"]),
            "length_bucket": "≤500" if n <= 500 else ("501–1500" if n <= 1500 else ">1500"),
        })
    return pd.DataFrame(rows)


def breakdowns(pq, questions, corpus, runs, index: str) -> dict:
    """``by_question_type``, ``by_engine_type``, ``by_kind``, ``mentions_section_id``, ``length_bucket``."""
    qf = _question_frame(questions, corpus)
    p = pq[pq["index"] == index].copy()
    p["col"] = np.where(p["level"] == "evidence", p["metric"], p["level"] + "_" + p["metric"])
    p = p[p["col"].isin(KEY_METRICS)]
    wide = p.pivot_table(index=["run_id", "question_id"], columns="col", values="value", aggfunc="first").reset_index()
    wide = wide.merge(qf, on="question_id").merge(runs[["run_id", "method", "gate", "budget", "params_version"]], on="run_id")
    out = {}
    for sheet, by in [("by_question_type", "question_type"), ("by_engine_type", "engine_type"), ("by_kind", "kind"),
                      ("mentions_section_id", "mentions_section_id"), ("length_bucket", "length_bucket")]:
        cols = [c for c in KEY_METRICS if c in wide.columns]
        g = wide.groupby(["run_id", "method", "gate", "budget", "params_version", by], dropna=False)
        t = g[cols].mean()
        t.insert(0, "n", g.size())
        out[sheet] = t.reset_index().sort_values(["method", "gate", "budget", by])
    return out


def per_question_ranks(pq, runs, index: str):
    """Rank of the first gold hit (section level), one row per question, one column per run."""
    p = pq[(pq["index"] == index) & (pq["level"] == "section") & (pq["metric"] == "first_gold_rank")]
    return p.pivot_table(index="question_id", columns="run_id", values="value", aggfunc="first").reset_index()


def unanswerable_sheet(run_ids: list[str], questions: list[dict], corpus, root=None):
    """Retrieved sections for the 8 unanswerable questions, per run (inspection only, not scored)."""
    import pandas as pd

    un = [q for q in questions if not q["answerable"]]
    rows = []
    for rid in run_ids:
        cfg = load_config(rid, root)
        units = corpus.units(cfg["index"])
        hits = load_hits(rid, root)
        st = {s["question_id"]: s for s in load_stats(rid, root)}
        for q in un:
            h = hits[hits["question_id"] == q["question_id"]].sort_values("rank")
            secs = list(dict.fromkeys(units[u]["section_id"] for u in h["unit_id"]))[:5]
            dis = set(q.get("distractor_section_ids") or [])
            s = st.get(q["question_id"], {})
            rows.append({
                "run_id": rid, "method": cfg["retriever"], "index": cfg["index"], "gate": cfg["gate"],
                "question_id": q["question_id"], "question": q["question"],
                "top5_sections": " | ".join(secs), "top5_distractors": " | ".join(x for x in secs if x in dis),
                "n_distractors_in_top5": sum(x in dis for x in secs),
                "top1_score": float(h["score"].iloc[0]) if len(h) else np.nan, "n_hits": len(h),
                "gate_kept": "; ".join(f"{g['pos']}:{g['signal']}={g['n_kept']}/{g['n_in']}" for g in s.get("gates", [])),
            })
    return pd.DataFrame(rows)


def write_index_files(index: str, rm, pq, runs, gates, costs, gate_overall, sig, questions, corpus,
                      out_root: str | Path = "results", root=None) -> dict:
    """``results/<index>/metrics.csv`` and ``metrics.xlsx``."""
    import pandas as pd

    d = Path(out_root) / index
    d.mkdir(parents=True, exist_ok=True)
    table = metrics_table(rm, runs, gates, costs, gate_overall, index)
    table.to_csv(d / "metrics.csv", index=False, encoding="utf-8")
    run_ids = list(runs.loc[runs["index"] == index, "run_id"])
    sheets = {"metrics": table, **breakdowns(pq, questions, corpus, runs, index)}
    s = sig[sig["index"] == index] if sig is not None and not sig.empty else pd.DataFrame()
    sheets["significance"] = s.sort_values(["rq", "level", "metric", "a", "b"]) if not s.empty else s
    sheets["gate"] = gates[gates["run_id"].isin(run_ids)] if not gates.empty else gates
    sheets["per_question"] = per_question_ranks(pq, runs, index)
    sheets["unanswerable"] = unanswerable_sheet(run_ids, questions, corpus, root)
    with pd.ExcelWriter(d / "metrics.xlsx", engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return {"csv": d / "metrics.csv", "xlsx": d / "metrics.xlsx", "runs": len(table)}


def write_granularity(sig_rq7, path: str | Path = "results/granularity.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sig_rq7.to_csv(path, index=False, encoding="utf-8")
    return Path(path)


# ---------------------------------------------------------------- Mermaid diagrams


def _esc(s: str) -> str:
    return str(s).replace('"', "'")


def pipeline_diagram(run_id: str, rm=None, sig=None, gates=None, costs=None, root=None) -> str:
    """Mermaid flowchart of one run, generated from its stored config (never hand-drawn)."""
    c = load_config(run_id, root)
    cfg = c["config"]
    lines = ["flowchart LR", '  Q(["question"])']
    prev = ["Q"]

    def link(srcs, dst):
        for s in srcs:
            lines.append(f"  {s} --> {dst}")

    qt = cfg.get("query_transform")
    if qt:
        label = "Q1 HyDE<br/>Qwen, 1 passage,<br/>mean with question" if qt["type"] == "hyde" else "Q2 keywords<br/>Qwen terms<br/>+ question"
        lines.append(f'  QT["{label}"]')
        link(prev, "QT")
        prev = ["QT"]
    flt = cfg.get("filter")
    if flt:
        src = "label (oracle)" if flt["type"] == "oracle" else "Qwen JSON"
        lines.append(f'  FL{{"filter<br/>engine_type ∈ {{x, all}}<br/>{src}"}}')
        link(prev, "FL")
        prev = ["FL"]
    lines.append(f'  IX[("{COLLECTIONS[cfg["index"]]}<br/>{"base units" if cfg["index"] == "units" else "500-token windows"}")]')
    link(prev, "IX")
    names = {"bm25": "BM25", "bge_sparse": "bge-m3 sparse", "dense": "bge-m3 dense",
             "colbert": f"ColBERT MaxSim<br/>{cfg.get('colbert_engine', 'client')}, exact"}
    outs = []
    for i, s in enumerate(cfg["first_stages"]):
        lines.append(f'  S{i}["{names[s]}<br/>top-{cfg["candidates"]}"]')
        lines.append(f"  IX --> S{i}")
        node = f"S{i}"
        if cfg.get("gate_a"):
            g = cfg["gate_a"]
            lines.append(f'  GA{i}{{"gate A<br/>τ {g["tau"]} δ {g["delta"]}<br/>keep {g["min_keep"]}–{g["max_keep"]}"}}')
            lines.append(f"  S{i} --> GA{i}")
            node = f"GA{i}"
        outs.append(node)
    prev = outs
    if cfg.get("fusion"):
        lines.append(f'  FU["RRF<br/>k = {cfg["fusion"]["k"]}"]')
        link(prev, "FU")
        prev = ["FU"]
    if cfg.get("colbert_rerank") or cfg.get("rerank"):
        label = f"ColBERT MaxSim rerank<br/>{cfg.get('colbert_engine', 'client')}, exact" if cfg.get("colbert_rerank") else (
            ("X1 bge-reranker-v2-m3" if cfg["rerank"]["model"] == "bge" else "X2 jina-reranker-v3") + f"<br/>top {cfg['rerank']['input']} in")
        lines.append(f'  RR["{label}"]')
        link(prev, "RR")
        prev = ["RR"]
        if cfg.get("gate_b"):
            g = cfg["gate_b"]
            lines.append(f'  GB{{"gate B<br/>τ {g["tau"]} δ {g["delta"]}<br/>keep {g["min_keep"]}–{g["max_keep"]}"}}')
            link(prev, "GB")
            prev = ["GB"]
    if cfg.get("lookup"):
        lines.append('  LK["T1 section-id lookup<br/>named sections first"]')
        link(prev, "LK")
        prev = ["LK"]
    ex = cfg.get("expansion")
    if ex:
        if ex["rule"] == "agent":
            label = f"T4 agent<br/>≤{ex['step_limit']} steps, budget {ex['budget']}<br/>expand · follow_refs · search · done"
        else:
            label = {"none": "T0 no expansion", "parent": "T2 parent section", "refs": "T3 cross-refs, 1 hop",
                     "parent+refs": "T2+T3 parent + cross-refs"}[ex["rule"]] + f"<br/>budget {ex['budget']}"
        lines.append(f'  EX["{label}"]')
        link(prev, "EX")
        prev = ["EX"]
    lines.append(f'  CTX(["final context<br/>{"budgeted" if ex else "k ≤ " + str(cfg["k"])}"])')
    link(prev, "CTX")

    panel = _panel(run_id, rm, gates, costs)
    if panel:
        lines.append(f'  subgraph M["metrics"]\n    MP["{_esc(panel)}"]\n  end')
        lines.append("  CTX -.- MP")
    md = [f"# {run_id}", "", f"{cfg['method']}: {cfg.get('description', '')} · index `{cfg['index']}` · gate "
          f"{'on' if cfg['gate'] else 'off'} · params `{cfg['params_version']}` · config `{c['config_hash']}`", "",
          "```mermaid", *lines, "```", ""]
    if sig is not None and not sig.empty:
        s = sig[(sig["run_a"] == run_id) | (sig["run_b"] == run_id)]
        s = s[(s["level"] == "section") & (s["metric"] == "ndcg@10")]
        if not s.empty:
            md += ["| comparison | Δ section nDCG@10 | 95% CI | p (Holm) |", "|---|---|---|---|"]
            for _, r in s.iterrows():
                md.append(f"| {r['rq']} {r['a']} vs {r['b']} | {r['diff']:+.3f} | [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}] | {r['p_holm']:.4f} |")
            md.append("")
    return "\n".join(md)


def _panel(run_id: str, rm, gates, costs) -> str:
    if rm is None:
        return ""
    r = rm[(rm["run_id"] == run_id) & (rm["subset"] == "all")]

    def val(level, metric):
        x = r[(r["level"] == level) & (r["metric"] == metric)]
        return x.iloc[0] if len(x) else None

    out = []
    nd = val("section", "ndcg@10")
    if nd is not None:
        out.append(f"section nDCG@10 {nd['value']:.3f} [{nd['ci_lo']:.3f}, {nd['ci_hi']:.3f}]")
    for level, metric, name in [("section", "recall@10", "Recall@10"), ("evidence", "evidence_recall@10", "evidence recall@10")]:
        x = val(level, metric)
        if x is not None:
            out.append(f"{name} {x['value']:.3f}")
    if gates is not None and not gates.empty:
        g = gates[gates["run_id"] == run_id]
        if not g.empty:
            out.append(f"kept length {g['kept_mean'].mean():.1f} · gate loss {g['gate_loss'].max():.3f}")
    if costs is not None and not costs.empty:
        cc = costs[costs["run_id"] == run_id]
        if not cc.empty:
            out.append(f"latency {cc['latency_total_s'].iloc[0]:.2f} s/question")
    out.append("n = 92 answerable")
    return "<br/>".join(out)


def write_diagram(run_id: str, out_dir: str | Path = "results/diagrams", **kw) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.md"
    p.write_text(pipeline_diagram(run_id, **kw), encoding="utf-8")
    return p


def diagram_matches_config(run_id: str, root=None) -> bool:
    """Plan 8: each diagram's stage list matches its run's config.json."""
    cfg = load_config(run_id, root)["config"]
    text = pipeline_diagram(run_id, root=root)
    checks = [
        all(s in text for s in [{"bm25": "BM25", "bge_sparse": "bge-m3 sparse", "dense": "bge-m3 dense",
                                  "colbert": "ColBERT MaxSim"}[x] for x in cfg["first_stages"]]),
        ("gate A" in text) == bool(cfg.get("gate_a")),
        ("gate B" in text) == bool(cfg.get("gate_b")),
        ("RRF" in text) == bool(cfg.get("fusion")),
        ("filter" in text) == bool(cfg.get("filter")),
        ("lookup" in text) == bool(cfg.get("lookup")),
        ("budget" in text) == bool(cfg.get("expansion")),
    ]
    return all(checks)
