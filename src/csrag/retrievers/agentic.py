"""T4: agentic small-to-big (plan sections 1 and 4.7).

Single question, step limit, token budget. The agent starts from H1's hits
(not from an empty context) and decides per question whether they are already
enough (it calls ``done`` straight away), or whether to

- ``expand(unit_id, to="section"|"group")``  bring in the parent section or CS+AMC group,
- ``follow_refs(unit_id)``                   bring in the sections a unit references (one hop:
                                             refused on hop-1 content, so no chains),
- ``search(query)``                          run H1 again (with the run's gate) on a new query,
- ``done(evidence_ids)``                     stop, listing the evidence that answers the question.

Every step is one LLM call (Qwen, thinking off, temperature 0, JSON-schema output),
re-sent with the full current state rather than a growing transcript, so the
prompt stays bounded by the budget and every call is cacheable. Each step is
logged with its decision, tool, arguments and the tokens it added.

Ranked list for evaluation: the ``done`` evidence first, then everything else
the agent retrieved, in order of first appearance. Evidence recall is also
measured on the whole context within the budget.
"""

from __future__ import annotations

import json
from typing import Callable

from .base import Hit
from .structural import ContextBuilder

AGENT_SYSTEM = """You are a retrieval agent for EASA CS-E Amendment 9: the certification specifications for aircraft engines (CS-E sections) and their acceptable means of compliance (AMC sections). Each CS-E section and its AMCs form a group.

You do not answer the question. You decide whether the evidence collected so far is enough to answer it completely, and if it is not, you collect more with one tool per step:
- expand(unit_id, to): add the whole parent "section" of an evidence item, or its whole "group" (the CS-E section with all its AMCs). Use it when an item is cut off or the answer continues elsewhere in the same section or group.
- follow_refs(unit_id): add the sections that an evidence item explicitly references (for example "see CS-E 510" or "AMC E 740(c)(4)"). One hop only: not allowed on items that were themselves added by follow_refs (hop 1).
- search(query): run the search engine again with a new query, for evidence that is missing and not reachable from the current items. The top new results are added.
- done(evidence_ids): stop. List the ids of the evidence items that are needed to answer the question, most important first.

Rules:
- If the evidence already answers the question, call done immediately.
- Every addition must fit the remaining token budget; additions that do not fit are refused.
- You have a limited number of steps; when they run out, all evidence is kept.

Reply with one JSON object with the fields: reason (one short sentence), tool, unit_id, to, query, evidence_ids. Set the fields the tool does not use to null (or [] for evidence_ids)."""

AGENT_USER = """Question: {question}

Step {step} of {step_limit}. Token budget: {used} of {budget} used, {remaining} left.

Evidence so far (hop 0 = found by search or expanded from a search result; hop 1 = brought in by a cross-reference):

{evidence}

Actions so far:
{history}

Choose the next action."""


def _schema(ids: list[str]) -> dict:
    uid = {"anyOf": [{"type": "string", "enum": ids}, {"type": "null"}]} if ids else {"type": "null"}
    ev = {"type": "array", "items": {"type": "string", "enum": ids}} if ids else {"type": "array", "maxItems": 0}
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "tool": {"type": "string", "enum": ["expand", "follow_refs", "search", "done"]},
            "unit_id": uid,
            "to": {"anyOf": [{"type": "string", "enum": ["section", "group"]}, {"type": "null"}]},
            "query": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "evidence_ids": ev,
        },
        "required": ["reason", "tool", "unit_id", "to", "query", "evidence_ids"],
        "additionalProperties": False,
    }


class Agent:
    def __init__(self, corpus, index: str, llm, budget: int, step_limit: int = 5, initial_hits: int = 5,
                 search_top: int = 5, search_fn: Callable[[str], tuple[list[tuple[str, float]], dict]] | None = None):
        self.corpus = corpus
        self.index = index
        self.llm = llm
        self.budget = budget
        self.step_limit = step_limit
        self.initial_hits = initial_hits
        self.search_top = search_top
        self.search_fn = search_fn  # query -> (H1 fused list, stats)

    # ------------------------------------------------------------ prompt

    def _evidence_text(self, ctx: ContextBuilder) -> str:
        units = self.corpus.units(self.index)
        if not ctx.items:
            return "(none)"
        blocks = []
        for p in self.corpus.merge_spans([i["unit_id"] for i in ctx.items], self.index):
            hops = sorted({ctx.hop_of(u) for u in p["unit_ids"]})
            refs = []
            for u in p["unit_ids"]:
                refs += [r for r in units[u]["cross_refs"] if r not in refs]
            head = (
                f"### ids: {', '.join(p['unit_ids'])} | section {p['section_id']} | group {units[p['unit_ids'][0]]['group_id']}"
                f" | hop {'/'.join(map(str, hops))} | {p['n_tokens']} tokens"
                + (f" | references: {', '.join(refs)}" if refs else "")
            )
            blocks.append(f"{head}\n{p['text']}")
        return "\n\n".join(blocks)

    def _messages(self, question: str, ctx: ContextBuilder, step: int, history: list[str]) -> list[dict]:
        user = AGENT_USER.format(
            question=question,
            step=step,
            step_limit=self.step_limit,
            used=ctx.tokens,
            budget=self.budget,
            remaining=ctx.remaining,
            evidence=self._evidence_text(ctx),
            history="\n".join(history) if history else "(none)",
        )
        return [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user}]

    # ------------------------------------------------------------ tools

    def _add_sections(self, ctx: ContextBuilder, section_ids: list[str], hop: int, source: str | None) -> tuple[list[str], int]:
        notes, total = [], 0
        for sid in section_ids:
            r = ctx.add(self.corpus.units_of(sid, self.index), hop, "expanded", 0.0, source)
            if r["refused"]:
                notes.append(f"{sid}: refused, needs {r['needed']} tokens, {ctx.remaining} left")
            elif r["added"]:
                notes.append(f"{sid}: added {r['tokens_added']} tokens")
                total += r["tokens_added"]
            else:
                notes.append(f"{sid}: already in the evidence")
        return notes, total

    def _apply(self, a: dict, ctx: ContextBuilder, appeared: list[str], search_stats: list[dict]) -> tuple[str, int, bool]:
        """(observation, tokens added, valid)."""
        units = self.corpus.units(self.index)
        tool, uid = a.get("tool"), a.get("unit_id")
        if tool in ("expand", "follow_refs") and (uid is None or uid not in ctx):
            return f"invalid: {tool} needs the id of an evidence item", 0, False
        if tool == "expand":
            to = a.get("to") or "section"
            u = units[uid]
            target = self.corpus.units_of(u["section_id"], self.index) if to == "section" else self.corpus.units_of_group(u["group_id"], self.index)
            r = ctx.add(target, ctx.hop_of(uid), "expanded", 0.0, ctx.items[ctx.pos[uid]]["source"])
            for x in target:
                if x["unit_id"] not in appeared:
                    appeared.append(x["unit_id"])
            if r["refused"]:
                return f"refused: the {to} needs {r['needed']} more tokens, {ctx.remaining} left", 0, True
            if not r["added"]:
                return f"nothing new: the {to} is already in the evidence", 0, True
            return f"added the {to} of {uid}: {len(r['added'])} units, {r['tokens_added']} tokens", r["tokens_added"], True
        if tool == "follow_refs":
            if ctx.hop_of(uid) != 0:
                return f"refused: {uid} was brought in by a reference (hop 1); references are followed one hop only", 0, True
            refs = units[uid]["cross_refs"]
            if not refs:
                return f"{uid} references no other CS-E or AMC section", 0, True
            for sid in refs:
                for x in self.corpus.units_of(sid, self.index):
                    if x["unit_id"] not in appeared:
                        appeared.append(x["unit_id"])
            notes, total = self._add_sections(ctx, refs, 1, uid)
            return "; ".join(notes), total, True
        if tool == "search":
            q = (a.get("query") or "").strip()
            if not q or self.search_fn is None:
                return "invalid: search needs a query", 0, False
            pool, st = self.search_fn(q)
            search_stats.append(st)
            new = [(u, s) for u, s in pool if u not in ctx][: self.search_top]
            for u, _ in pool:
                if u not in appeared:
                    appeared.append(u)
            notes, total = [], 0
            for u, s in new:
                r = ctx.add([units[u]], 0, "fused", s, None)
                if r["added"]:
                    notes.append(f"{u} (+{r['tokens_added']})")
                    total += r["tokens_added"]
            if not notes:
                return "no new results fit the budget", 0, True
            return "added " + ", ".join(notes), total, True
        return f"invalid tool {tool!r}", 0, False

    # ------------------------------------------------------------ loop

    def run(self, question: str, pool: list[tuple[str, float]]) -> dict:
        units = self.corpus.units(self.index)
        ctx = ContextBuilder(self.corpus, self.index, self.budget)
        appeared = [u for u, _ in pool]
        for u, s in pool[: self.initial_hits]:
            ctx.add([units[u]], 0, "fused", s, None)
        history: list[str] = []
        trace: list[dict] = []
        llm_results: list[dict] = []
        search_stats: list[dict] = []
        done_ids: list[str] | None = None
        for step in range(1, self.step_limit + 1):
            ids = [i["unit_id"] for i in ctx.items]
            r = self.llm.chat(self._messages(question, ctx, step, history), max_tokens=400, schema=_schema(ids), name="agent_step")
            llm_results.append(r)
            try:
                a = json.loads(r["content"])
            except json.JSONDecodeError:
                a = {"tool": None, "reason": "unparseable output"}
            tool = a.get("tool")
            args = {k: a.get(k) for k in ("unit_id", "to", "query", "evidence_ids") if a.get(k) not in (None, [])}
            if tool == "done":
                done_ids = [e for e in dict.fromkeys(a.get("evidence_ids") or []) if e in ctx]
                obs, added, valid = f"done with {len(done_ids)} evidence items", 0, True
            else:
                obs, added, valid = self._apply(a, ctx, appeared, search_stats)
            trace.append(
                {
                    "step": step,
                    "decision": "stop" if tool == "done" else (tool or "invalid"),
                    "tool": tool,
                    "args": args,
                    "reason": a.get("reason"),
                    "valid": valid,
                    "observation": obs,
                    "tokens_added": added,
                    "context_tokens": ctx.tokens,
                    "llm_seconds": r["seconds"],
                    "prompt_tokens": r["usage"]["prompt_tokens"],
                    "completion_tokens": r["usage"]["completion_tokens"],
                }
            )
            history.append(f"{step}. {tool}({json.dumps(args, ensure_ascii=False)}) -> {obs}")
            if done_ids is not None:
                break
        forced = done_ids is None
        if forced:
            done_ids = [i["unit_id"] for i in ctx.items]
            trace.append({"step": len(trace) + 1, "decision": "limit", "tool": None, "args": {}, "reason": "step limit reached",
                          "valid": True, "observation": "all evidence kept", "tokens_added": 0, "context_tokens": ctx.tokens,
                          "llm_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0})
        # ranked list: done evidence first, then everything else in order of first appearance
        order = list(done_ids) + [u for u in [i["unit_id"] for i in ctx.items] + appeared if u not in done_ids]
        order = list(dict.fromkeys(order))
        score = dict(pool)
        hits = []
        for u in order:
            if u in ctx:
                it = ctx.items[ctx.pos[u]]
                hits.append(Hit(u, it["section_id"], it["score"], it["stage"], it["hop"],
                                {"source": it["source"], "selected": u in done_ids} if it["source"] else {"selected": u in done_ids},
                                units[u]["n_tokens_bge"], True))
            else:
                hits.append(Hit(u, units[u]["section_id"], score.get(u, 0.0), "fused", 0, {"selected": False},
                                units[u]["n_tokens_bge"], False))
        return {
            "hits": hits,
            "trace": trace,
            "llm_results": llm_results,
            "search_stats": search_stats,
            "context_tokens": ctx.tokens,
            "selected_tokens": self.corpus.context_tokens(done_ids, self.index) if done_ids else 0,
            "path": [t["decision"] for t in trace],
            "forced_stop": forced,
        }
