"""Question set 2: checks and helpers for the generation specs in ``data/eval/specs/``.

Set 2 is a second question set, all answerable, written family by family:
``paraphrase``, ``section_id``, ``twin_topic`` and ``multi_section``. Each family is
one file, ``data/eval/set2/<family>.jsonl``, in the record format of
``data/eval/questions.jsonl`` plus the fields in ``data/eval/specs/README.md``.

Nothing here runs a retriever or reads ``results/``. The pool for the
single-source check is plain BM25 over whole sections plus structural
neighbours (group siblings, twin sections, cross-references), listed in
document order so that the judge sees no ranking.

The bge-m3 tokenizer is not needed: ``LightCorpus`` has the sections, groups and
window counts, which is all the quote and span checks use.

CLI, from the repo root:

    uv run python -m csrag.qgen fill    data/eval/set2/paraphrase.jsonl    # derived fields, in place
    uv run python -m csrag.qgen check   data/eval/set2/paraphrase.jsonl    # errors -> exit code 1
    uv run python -m csrag.qgen check   --final data/eval/set2/*.jsonl     # also: every check passed
    uv run python -m csrag.qgen pool    data/eval/set2/paraphrase.jsonl    # candidates for the judge
    uv run python -m csrag.qgen stats   data/eval/set2/*.jsonl
    uv run python -m csrag.qgen overlap "question text" "CS-E 740" ["AMC E 740(c)(4)" ...]
    uv run python -m csrag.qgen refs    [--section "CS-E 740"]              # cross-group references
    uv run python -m csrag.qgen twins                                       # the twin table
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from csparse.merge import Corpus as _BaseCorpus
from csparse.merge import read_jsonl

from . import evaluate as ev
from .enrich import REF_RE, engine_type, file_sha256, find_section_ids, normalize_map, normalize_ref

DATA = Path("data")
V1_QUESTIONS = DATA / "eval" / "questions.jsonl"
SET2 = DATA / "eval" / "set2"

FAMILIES = ("paraphrase", "section_id", "twin_topic", "multi_section")
ID_RE = {
    "paraphrase": re.compile(r"pa(\d{3})([hu])"),  # pa<v1 number><h = heavy | u = user_voice>
    "section_id": re.compile(r"id\d{3}"),
    "twin_topic": re.compile(r"tw(\d{2})([pt])"),  # tw<pair number><p = piston | t = turbine>
    "multi_section": re.compile(r"ms\d{3}"),
}

V1_FIELDS = (
    "question_id", "question", "question_type", "answer", "answer_aliases", "answer_references", "answerable",
    "gold", "gold_kind", "requires_all_gold", "mentions_section_id", "engine_type", "subpart", "difficulty",
    "derived_from", "distractor_section_ids", "source", "generator", "corpus_version", "review",
)
COMMON_FIELDS = ("family", "asks_about_engine_type", "lexical_overlap", "verification")
FAMILY_FIELDS = {
    "paraphrase": ("variant_of", "paraphrase_level"),
    "section_id": ("id_form", "id_intent"),
    "twin_topic": ("twin_key", "twin_of", "engine_cue"),
    "multi_section": ("hop_type",),
}
OPTIONAL_FIELDS = {"multi_section": ("twin_key",)}

QUESTION_TYPES = {"factual", "numeric", "list", "procedural", "definition", "rationale", "summary"}
DIFFICULTY = {"easy", "medium", "hard"}
ASKS_ABOUT = {"piston", "turbine", "both", "unspecified"}
REVIEW_STATUS = {"unreviewed", "ok", "fixed", "rejected"}
CHECK_STATUS = {"pass", "fail", "pending", "n/a"}
VERIFICATION_KEYS = ("answer_from_gold", "single_source", "same_meaning", "each_alone_fails", "second_judge")
# family-specific checks; "n/a" for the other families
FAMILY_CHECKS = {"same_meaning": "paraphrase", "each_alone_fails": "multi_section"}

PARAPHRASE_LEVELS = {"heavy", "user_voice"}
# Content-word overlap with the gold section(s), see ``lexical_overlap``. Under this measure the
# 92 answerable v1 questions have mean 0.84, median 0.88, 10th percentile 0.63 and minimum 0.47.
OVERLAP_CAP = {"heavy": 0.50, "user_voice": 0.65}
OVERLAP_MIN_DROP = 0.20  # a paraphrase must be at least this much below its original

ID_FORMS = {"canonical", "paragraph", "loose"}
ID_INTENTS = {"objective", "summary", "lookup", "redirect"}
SUMMARY_MIN_QUOTES = 3
ENGINE_CUES = {"explicit", "implied"}
HOP_TYPES = {"cs_amc", "reference", "comparison"}

TYPE_WORD_RE = re.compile(r"\b(piston|turbine|reciprocating)s?\b", re.I)
# Section ids as people type them: "CS-E 740", "cs-e 740", "CSE740", "CS E 740 (c)(4)", "AMC E510", "AMC to CS-E 50(l)".
# Other documents ("CS-25", "AMC-20", "CS 23.1093") do not match.
LOOSE_ID_RE = re.compile(r"\b(CS|AMC)[\s-]*(?:(to)\s+CS[\s-]*)?E[\s-]*(\d{2,4})((?:\s*\([a-z0-9]+\))*)", re.I)

# Piston/turbine twin topics, checked against the text; data/eval/specs/twin_topic.md says
# what each side covers. The section lists are the only sections a twin question's gold may
# come from, per side. "pairs" is the number of question pairs to write for the key.
#   cross_subpart   the piston side is in Subparts B/C, the turbine side in D/E (CS-E 10(c), (d))
#   common_section  one Subpart A section (applies to both) with a piston and a turbine paragraph
#   loose           related topics, not parallel texts: optional, extra care
_AMC740 = ["AMC E 740(c)(2)(i)", "AMC E 740(c)(3)", "AMC E 740(c)(4)", "AMC E 740(f)(1)", "AMC E 740(f)(5)",
           "AMC E 740(g)(1)", "AMC E 740(i)(2)"]
TWINS: dict[str, dict] = {
    "failure_analysis": {"class": "cross_subpart", "pairs": 2,
                         "piston": ["CS-E 210", "AMC E 210"], "turbine": ["CS-E 510", "AMC E 510"]},
    "fuel_system": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 250"], "turbine": ["CS-E 560", "AMC E 560"]},
    "oil_system": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 270"], "turbine": ["CS-E 570", "AMC E 570"]},
    "relight_in_flight": {"class": "cross_subpart", "pairs": 1, "piston": ["CS-E 240"], "turbine": ["CS-E 910", "AMC E 910"]},
    "performance_correction": {"class": "cross_subpart", "pairs": 1,
                               "piston": ["CS-E 320", "AMC E 320"], "turbine": ["CS-E 620", "AMC E 620"]},
    "vibration": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 340", "AMC E 340"], "turbine": ["CS-E 650", "AMC E 650"]},
    "calibration": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 350", "AMC E 350"], "turbine": ["CS-E 730", "AMC E 730"]},
    "starting": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 370"], "turbine": ["CS-E 750", "AMC E 750(b)"]},
    "low_temperature_starting": {"class": "cross_subpart", "pairs": 2,
                                 "piston": ["CS-E 380", "AMC E 380"], "turbine": ["CS-E 770", "AMC E 770"]},
    "acceleration": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 390"], "turbine": ["CS-E 745", "AMC E 745"]},
    "over_speed": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 400"], "turbine": ["CS-E 830", "AMC E 830(c)"]},
    "endurance": {"class": "cross_subpart", "pairs": 2, "piston": ["CS-E 440", "AMC E 440(b)(4)"], "turbine": ["CS-E 740", *_AMC740]},
    "contaminated_fuel": {"class": "cross_subpart", "pairs": 2,
                          "piston": ["CS-E 470", "AMC E 470"], "turbine": ["CS-E 670", "AMC E 670"]},
    "ratings": {"class": "common_section", "pairs": 1, "piston": ["CS-E 40"], "turbine": ["CS-E 40"]},
    "operating_limitations": {"class": "common_section", "pairs": 1, "piston": ["AMC E 40(d)"], "turbine": ["AMC E 40(d)"]},
    "propeller_pitch_change_cycles": {"class": "common_section", "pairs": 1, "piston": ["CS-E 180"], "turbine": ["CS-E 180"]},
    "installation_assumptions": {"class": "common_section", "pairs": 1, "piston": ["AMC E 30"], "turbine": ["AMC E 30"]},
    "icing": {"class": "loose", "pairs": 1, "piston": ["CS-E 230"], "turbine": ["CS-E 780", "AMC E 780"]},
    "water_ingestion": {"class": "loose", "pairs": 1, "piston": ["CS-E 430"],
                        "turbine": ["CS-E 790", "AMC E 790", "AMC E 790(a)(1)", "AMC E 790(a)(2)", "CS-E 790 Appendix A"]},
}

STOPWORDS = frozenset(
    """a an the and or nor but if then than so of to in on at by for from with without into onto over under
    about above below between during before after through throughout within per via as is are was were be
    been being am do does did done has have had having it its this that these those there here which what
    who whom whose when where why how whether any all each every some such no not only own same other
    another more most less least many much few several can could may might must shall should will would
    also please tell me my we our us you your they their them i engine engines cs amc e""".split()
)
BM25_K1, BM25_B, POOL_BM25_K, POOL_REF_K = 1.2, 0.75, 6, 4


# ---------------------------------------------------------------- corpus without the tokenizer


class LightCorpus(_BaseCorpus):
    """``csparse.merge.Corpus`` plus what ``evaluate.load_questions`` needs, without bge-m3."""

    def __init__(self, data_dir: str | Path = DATA):
        super().__init__(data_dir)
        d = Path(data_dir)
        self.data_dir = d
        self._norm: dict[str, tuple[str, list[int]]] = {}
        self.section_ids = list(self.sections)
        self.order = {sid: i for i, sid in enumerate(self.section_ids)}
        self.n_windows = Counter(r["section_id"] for r in read_jsonl(d / "windows_500.jsonl"))
        self.refs: set[tuple[str, str]] = {
            (r["src_section_id"], r["dst_section_id"]) for r in read_jsonl(d / "cross_refs.jsonl") if r["index"] == "units"
        }

    def section_norm(self, section_id: str) -> tuple[str, list[int]]:
        if section_id not in self._norm:
            self._norm[section_id] = normalize_map(self.sections[section_id]["body"])
        return self._norm[section_id]

    def corpus_version(self) -> str:
        return "groups.jsonl sha256:" + file_sha256(self.data_dir / "groups.jsonl")


@lru_cache(maxsize=1)
def light_corpus(data_dir: str = str(DATA)) -> LightCorpus:
    return LightCorpus(data_dir)


# ---------------------------------------------------------------- words, overlap, ids


@lru_cache(maxsize=1)
def _stemmer():
    from py_rust_stemmers import SnowballStemmer  # installed with fastembed (the BM25 stemmer)

    return SnowballStemmer("english")


def words(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; "CS-E 740(c)" -> cs, e, 740, c."""
    return re.findall(r"[a-z0-9]+", text.lower())


def content_stems(text: str) -> list[str]:
    """Stems of the content words: stopwords and one-letter tokens dropped, numbers kept."""
    st = _stemmer()
    return [st.stem_word(w) for w in words(text) if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


def lexical_overlap(question: str, section_ids: list[str], corpus: LightCorpus | None = None) -> tuple[float, list[str], list[str]]:
    """Share of the question's distinct content stems that occur in the gold sections' text.

    Returns ``(overlap, shared, not_shared)``. This is the measure behind the specs' caps;
    it is close to, but not the same as, the one in ``project_v1_output.md`` section 6.
    """
    corpus = corpus or light_corpus()
    q = list(dict.fromkeys(content_stems(question)))
    if not q:
        return float("nan"), [], []
    st = _stemmer()
    gold = {st.stem_word(w) for sid in section_ids for w in words(corpus.sections[sid]["text"])}
    shared = [w for w in q if w in gold]
    return round(len(shared) / len(q), 4), shared, [w for w in q if w not in gold]


def named_ids(text: str, corpus: LightCorpus | None = None) -> tuple[list[str], list[str]]:
    """``(t1_ids, any_ids)``: section ids T1's regex finds (``find_section_ids``), and the ids
    named in any spelling (``LOOSE_ID_RE``), both resolved to existing section ids."""
    corpus = corpus or light_corpus()
    t1 = find_section_ids(text, corpus.section_ids)
    anyf: list[str] = []
    for m in LOOSE_ID_RE.finditer(text):
        fam = "CS-E" if m.group(1).upper() == "CS" else "AMC E"
        parens = re.sub(r"\s+", "", m.group(4) or "").lower()
        for sid in normalize_ref(fam, m.group(3), parens, corpus.section_ids):
            if sid not in anyf:
                anyf.append(sid)
    return t1, anyf


def has_paragraph_ref(text: str, corpus: LightCorpus | None = None) -> bool:
    """Does the text name a paragraph below a section id, e.g. "CS-E 740(c)(4)"? A section id that
    has parentheses of its own ("AMC E 20(f)", "AMC to CS-E 50(l)") is not a paragraph reference."""
    corpus = corpus or light_corpus()
    return any(m.group(3) and m.group(0) not in corpus.sections for m in REF_RE.finditer(text))


def gold_engine_type(section_ids: list[str], corpus: LightCorpus) -> str:
    """v1 semantics: the gold sections' engine type; ``all`` if Subpart A only or both types."""
    types = {engine_type(corpus.sections[s]["subpart"]) for s in section_ids} - {"all"}
    return types.pop() if len(types) == 1 else "all"


# ---------------------------------------------------------------- files


def read(path: str | Path) -> list[dict]:
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def write(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ordered(rec: dict) -> dict:
    fam = rec.get("family")
    keys = [*V1_FIELDS, *COMMON_FIELDS, *FAMILY_FIELDS.get(fam, ()), *OPTIONAL_FIELDS.get(fam, ())]
    out = {k: rec[k] for k in keys if k in rec}
    out.update({k: v for k, v in rec.items() if k not in out})
    return out


def fill(path: str | Path, generator: str | None = None) -> int:
    """Write the derived fields in place: engine_type, subpart, gold_kind, lexical_overlap,
    corpus_version; defaults for the empty v1 fields, review and verification."""
    corpus = light_corpus()
    rows = read(path)
    cv = corpus.corpus_version()
    for r in rows:
        sids = list(dict.fromkeys(g["section_id"] for g in r.get("gold", []) if g["section_id"] in corpus.sections))
        prim = [g["section_id"] for g in r.get("gold", []) if g.get("role") == "primary" and g["section_id"] in corpus.sections]
        if sids:
            first = (prim or sids)[0]
            r["engine_type"] = gold_engine_type(sids, corpus)
            r["subpart"] = corpus.sections[first]["subpart"]
            r["gold_kind"] = corpus.sections[first]["kind"]
            r["lexical_overlap"] = lexical_overlap(r["question"], sids, corpus)[0]
        r["corpus_version"] = cv
        r.setdefault("answer_aliases", [])
        r.setdefault("answer_references", [])
        r.setdefault("answerable", True)
        r.setdefault("derived_from", None)
        r.setdefault("distractor_section_ids", [])
        r.setdefault("source", "claude-generated")
        if generator:
            r["generator"] = generator
        r.setdefault("review", {"status": "unreviewed", "reviewer": None, "notes": None})
        v = r.setdefault("verification", {})
        for k in VERIFICATION_KEYS:
            v.setdefault(k, "n/a" if k in FAMILY_CHECKS and r.get("family") != FAMILY_CHECKS[k] else "pending")
        v.setdefault("notes", None)
    write(path, [ordered(r) for r in rows])
    return len(rows)


# ---------------------------------------------------------------- checks


def check(paths: list[str | Path], final: bool = False) -> dict:
    """Errors and warnings per question id for set-2 files (all families may be mixed)."""
    corpus = light_corpus()
    v1 = {q["question_id"]: q for q in read(V1_QUESTIONS)}
    errors: dict[str, list[str]] = {}
    warnings: dict[str, list[str]] = {}

    def err(qid, msg):
        errors.setdefault(qid, []).append(msg)

    def warn(qid, msg):
        warnings.setdefault(qid, []).append(msg)

    rows: list[dict] = []
    for p in paths:
        qs, problems = ev.load_questions(p, corpus)
        for kind, items in problems.items():
            for it in items:
                qid = it[0] if isinstance(it, tuple) else it
                err(qid, f"{kind}: {it}")
        rows += qs
    by_id: dict[str, dict] = {}
    for r in rows:
        if r["question_id"] in by_id:
            err(r["question_id"], "duplicate question_id across files")
        by_id[r["question_id"]] = r
    cv = corpus.corpus_version()
    twin_pairs: Counter = Counter()
    per_variant: Counter = Counter()

    for r in rows:
        qid, fam, text = r["question_id"], r.get("family"), r.get("question", "")
        if fam not in FAMILIES:
            err(qid, f"family {fam!r} not in {FAMILIES}")
            continue
        missing = [k for k in (*V1_FIELDS, *COMMON_FIELDS, *FAMILY_FIELDS[fam]) if k not in r]
        if missing:
            err(qid, f"missing fields {missing} (run `qgen fill` for the derived ones)")
            continue
        if not ID_RE[fam].fullmatch(qid):
            err(qid, f"question_id does not match {ID_RE[fam].pattern}")
        if qid in v1:
            err(qid, "question_id collides with v1")
        if r["answerable"] is not True or not r["gold"]:
            err(qid, "set 2 is answerable only: answerable must be true with gold")
        if r["question_type"] not in QUESTION_TYPES:
            err(qid, f"question_type {r['question_type']!r}")
        if r["difficulty"] not in DIFFICULTY:
            err(qid, f"difficulty {r['difficulty']!r}")
        if r["asks_about_engine_type"] not in ASKS_ABOUT:
            err(qid, f"asks_about_engine_type {r['asks_about_engine_type']!r}")
        if r["derived_from"] is not None or r["distractor_section_ids"]:
            err(qid, "derived_from / distractor_section_ids are for v1 unanswerables; leave null / []")
        if r["corpus_version"] != cv:
            err(qid, "corpus_version differs from the current groups.jsonl (run `qgen fill`)")
        if (r.get("review") or {}).get("status") not in REVIEW_STATUS:
            err(qid, "review.status")
        for g in r["gold"]:
            if g.get("role") not in ("primary", "supporting"):
                err(qid, f"gold role {g.get('role')!r}")
            if not 25 <= len(g["quote"]) <= 600:
                warn(qid, f"quote length {len(g['quote'])} (aim for one or two sentences)")
        if len(r["_gold"]) != len(r["gold"]):
            continue  # a quote failed validation (reported above); the rules below need every span
        sids = list(dict.fromkeys(g["section_id"] for g in r["_gold"]))
        prim = [g["section_id"] for g in r["_gold"] if g["role"] == "primary"]
        if not prim:
            err(qid, "no primary gold")
        first = (prim or sids)[0]
        if r["engine_type"] != gold_engine_type(sids, corpus) or r["subpart"] != corpus.sections[first]["subpart"] \
                or r["gold_kind"] != corpus.sections[first]["kind"]:
            err(qid, "engine_type / subpart / gold_kind do not match the gold (run `qgen fill`)")
        ov = lexical_overlap(text, sids, corpus)[0]
        if r["lexical_overlap"] is None or abs(r["lexical_overlap"] - ov) > 1e-3:
            err(qid, f"lexical_overlap {r['lexical_overlap']} != {ov} (run `qgen fill`)")
        t1_ids, any_ids = named_ids(text, corpus)
        if bool(r["mentions_section_id"]) != bool(any_ids):
            err(qid, f"mentions_section_id={r['mentions_section_id']} but ids found: {any_ids}")
        if fam == "twin_topic" or fam == "multi_section":
            if any_ids:
                err(qid, f"{fam} questions must not name section ids (found {any_ids})")
            if re.search(r"\bsubparts?\s+[A-F]\b", text, re.I):
                err(qid, f"{fam} questions must not name a subpart; say 'piston' or 'turbine' instead")
        v = r["verification"] or {}
        for k in VERIFICATION_KEYS:
            if v.get(k) not in CHECK_STATUS:
                err(qid, f"verification.{k} = {v.get(k)!r}")
        if final:
            need = ["answer_from_gold", "single_source"] + [k for k, f in FAMILY_CHECKS.items() if f == fam]
            bad = [k for k in need if v.get(k) != "pass"]
            if bad or v.get("second_judge") == "fail":
                err(qid, f"--final: verification not passed: {bad or ['second_judge']}")
        if r["review"].get("status") == "rejected":
            warn(qid, "review.status = rejected: remove it before freezing")

        # ---- family rules
        if fam == "paraphrase":
            m = ID_RE[fam].fullmatch(qid)
            orig = v1.get(r["variant_of"])
            if orig is None or not orig["answerable"]:
                err(qid, f"variant_of {r['variant_of']!r} is not an answerable v1 question")
                continue
            if m and (f"q{m.group(1)}" != r["variant_of"] or {"h": "heavy", "u": "user_voice"}[m.group(2)] != r["paraphrase_level"]):
                err(qid, "question_id must be pa<v1 number><h|u> matching variant_of and paraphrase_level")
            if r["paraphrase_level"] not in PARAPHRASE_LEVELS:
                err(qid, f"paraphrase_level {r['paraphrase_level']!r}")
                continue
            per_variant[(r["variant_of"], r["paraphrase_level"])] += 1
            if [(g["section_id"], g["quote"], g["role"]) for g in r["gold"]] != \
                    [(g["section_id"], g["quote"], g["role"]) for g in orig["gold"]]:
                err(qid, "gold must be copied unchanged from variant_of")
            for k in ("answer", "answer_aliases", "answer_references", "question_type", "requires_all_gold"):
                if r[k] != orig[k]:
                    err(qid, f"{k} must be copied unchanged from variant_of")
            o_ov = lexical_overlap(orig["question"], sids, corpus)[0]
            cap = OVERLAP_CAP[r["paraphrase_level"]]
            if ov > cap:
                err(qid, f"lexical_overlap {ov} > cap {cap} for {r['paraphrase_level']}")
            if ov > o_ov - OVERLAP_MIN_DROP:
                err(qid, f"lexical_overlap {ov} is not {OVERLAP_MIN_DROP} below the original's {o_ov}")
            if named_ids(orig["question"], corpus)[1] != any_ids:
                err(qid, "a paraphrase names exactly the section ids its original names")

        elif fam == "section_id":
            form, intent = r["id_form"], r["id_intent"]
            if form not in ID_FORMS or intent not in ID_INTENTS:
                err(qid, f"id_form {form!r} / id_intent {intent!r}")
                continue
            if not any_ids:
                err(qid, "section_id questions must name a section id")
                continue
            if form == "loose" and t1_ids:
                err(qid, f"id_form loose, but T1's regex finds {t1_ids}: misspell it the way people do")
            if form in ("canonical", "paragraph") and not t1_ids:
                err(qid, f"id_form {form}, but T1's regex finds no id; write 'CS-E 740', 'AMC E 20(f)' or 'AMC to CS-E 50(l)'")
            if form == "paragraph" and not has_paragraph_ref(text, corpus):
                err(qid, "id_form paragraph names a paragraph below a section id, e.g. 'CS-E 740(c)(4)'")
            if form == "canonical" and has_paragraph_ref(text, corpus):
                err(qid, "id_form canonical names section ids exactly as in the document; a paragraph below one is id_form paragraph")
            if intent == "redirect":
                gk = corpus.sections[first]["kind"]
                cs_named = [s for s in any_ids if corpus.sections[s]["kind"] == "cs"]
                if gk != "amc" or not any(corpus.sections[s]["group_id"] == corpus.sections[first]["group_id"] for s in cs_named):
                    err(qid, "redirect: the question names a CS and the gold is an AMC of that CS")
                if first in any_ids:
                    err(qid, "redirect: the question must not name the gold AMC itself")
            elif not set(sids) & set(any_ids):
                err(qid, f"{intent}: the named id(s) {any_ids} must resolve to the gold section {sids}")
            if intent == "summary":
                if len(sids) != 1 or len(r["gold"]) < SUMMARY_MIN_QUOTES or not r["requires_all_gold"]:
                    err(qid, f"summary: one section, >= {SUMMARY_MIN_QUOTES} quotes, requires_all_gold true")
                if any(g["role"] != "primary" for g in r["gold"]):
                    err(qid, "summary: every quote is primary")
                if corpus.n_windows[first] < 2:
                    err(qid, "summary: pick a section of at least 2 windows (see section_id.md)")
                if r["question_type"] != "summary":
                    err(qid, "summary: question_type summary")
            else:
                if r["requires_all_gold"]:
                    err(qid, f"{intent}: requires_all_gold false")
                if len(sids) != 1:
                    err(qid, f"{intent}: one gold section")

        elif fam == "twin_topic":
            key = r["twin_key"]
            tw = TWINS.get(key)
            side = r["asks_about_engine_type"]
            if tw is None or side not in ("piston", "turbine"):
                err(qid, f"twin_key {key!r} / asks_about_engine_type {side!r}")
                continue
            if r["engine_cue"] not in ENGINE_CUES:
                err(qid, f"engine_cue {r['engine_cue']!r}")
            if not set(sids) <= set(tw[side]):
                err(qid, f"gold {sids} is not on the {side} side of {key}: {tw[side]}")
            m = ID_RE[fam].fullmatch(qid)
            if m and {"p": "piston", "t": "turbine"}[m.group(2)] != side:
                err(qid, "question_id suffix p/t must match asks_about_engine_type")
            has_type_word = bool(TYPE_WORD_RE.search(text))
            if r["engine_cue"] == "explicit" and not has_type_word:
                err(qid, "engine_cue explicit: say 'piston' or 'turbine'")
            if r["engine_cue"] == "implied" and has_type_word:
                err(qid, "engine_cue implied: no 'piston' / 'turbine' / 'reciprocating'; use a cue word (twin_topic.md)")
            if r["requires_all_gold"] or len(sids) != 1:
                err(qid, "twin_topic: one gold section, requires_all_gold false")
            mate = by_id.get(r["twin_of"])
            if mate is None:
                err(qid, f"twin_of {r['twin_of']!r} not found in the given files")
                continue
            if mate.get("twin_of") != qid or mate.get("twin_key") != key or mate.get("asks_about_engine_type") == side:
                err(qid, "twin_of must be reciprocal, same twin_key, the other engine type")
            if m and not ID_RE[fam].fullmatch(r["twin_of"] or "") or (m and r["twin_of"][:4] != qid[:4]):
                err(qid, "twins share the pair number: tw07p <-> tw07t")
            if mate.get("engine_cue") != r["engine_cue"]:
                err(qid, "both questions of a pair use the same engine_cue")
            if side == "piston":
                twin_pairs[key] += 1
                if tw["class"] == "common_section" and mate.get("_gold"):
                    a = [(g["char_start"], g["char_end"]) for g in r["_gold"]]
                    b = [(g["char_start"], g["char_end"]) for g in mate["_gold"]]
                    if any(s1 < e2 and s2 < e1 for s1, e1 in a for s2, e2 in b):
                        err(qid, "common_section twins quote different paragraphs (spans overlap)")
                elif tw["class"] != "common_section" and mate.get("_gold"):
                    if set(sids) & {g["section_id"] for g in mate["_gold"]}:
                        err(qid, "cross-subpart twins have different gold sections")

        elif fam == "multi_section":
            hop = r["hop_type"]
            if hop not in HOP_TYPES:
                err(qid, f"hop_type {hop!r}")
                continue
            if not r["requires_all_gold"] or len(sids) < 2:
                err(qid, "multi_section: requires_all_gold true and >= 2 gold sections")
                continue
            roles = {s: {g["role"] for g in r["_gold"] if g["section_id"] == s} for s in sids}
            if any(len(x) > 1 for x in roles.values()):
                err(qid, "one role per gold section")
            sup = [s for s in sids if "supporting" in roles[s]]
            groups = {corpus.sections[s]["group_id"] for s in sids}
            if hop == "cs_amc":
                kinds = [corpus.sections[s]["kind"] for s in sids]
                if len(groups) != 1 or kinds.count("cs") != 1 or "amc" not in kinds:
                    err(qid, "cs_amc: one CS and at least one of its own AMCs (same group)")
            elif hop == "reference":
                if len(sids) != 2 or len(prim) != 1 or len(sup) != 1:
                    err(qid, "reference: two sections, one primary (cites), one supporting (cited)")
                elif (prim[0], sup[0]) not in corpus.refs:
                    err(qid, f"reference: {prim[0]} does not cite {sup[0]} in data/cross_refs.jsonl")
                elif len(groups) != 2:
                    err(qid, "reference: the two sections are in different groups (use cs_amc within a group)")
            elif hop == "comparison":
                if sup:
                    err(qid, "comparison: every gold section is primary")
                key = r.get("twin_key")
                if key is not None:
                    tw = TWINS.get(key)
                    if tw is None or not (set(sids) & set(tw["piston"]) and set(sids) & set(tw["turbine"])):
                        err(qid, f"comparison with twin_key {key!r}: one gold section from each side")

    for (vo, lvl), n in per_variant.items():
        if n > 1:
            err(f"pa:{vo}", f"{n} paraphrases of {vo} at level {lvl}; one per level")
    for key, n in twin_pairs.items():
        if n > TWINS[key]["pairs"]:
            warnings.setdefault(f"tw:{key}", []).append(f"{n} pairs, quota {TWINS[key]['pairs']}")

    # near-duplicates against v1 and each other (warnings: paraphrases of one original are expected to share words)
    try:
        from rapidfuzz import fuzz

        texts = [(q["question_id"], q["question"]) for q in v1.values()] + [(r["question_id"], r["question"]) for r in rows]
        for r in rows:
            # twins and the two paraphrases of one original are written to be alike
            expected = {r["question_id"], r.get("twin_of"), r.get("variant_of")}
            expected |= {x["question_id"] for x in rows if r.get("variant_of") and x.get("variant_of") == r["variant_of"]}
            for oid, ot in texts:
                if oid not in expected and fuzz.token_set_ratio(r["question"], ot) >= 90:
                    warn(r["question_id"], f"near-duplicate of {oid}")
    except ImportError:  # pragma: no cover
        pass
    return {"n": len(rows), "errors": errors, "warnings": warnings}


# ---------------------------------------------------------------- pool for the single-source check


class SectionBM25:
    """Okapi BM25 over whole sections (header line + body), Snowball stems, content words only."""

    def __init__(self, corpus: LightCorpus):
        self.ids = corpus.section_ids
        self.docs = [Counter(content_stems(corpus.sections[s]["text"])) for s in self.ids]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def top(self, text: str, k: int) -> list[str]:
        q = set(content_stems(text))
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * self.len[i] / self.avg))
            scores.append((s, -i))
        return [self.ids[-i] for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def pool_for(r: dict, corpus: LightCorpus, bm25: SectionBM25) -> list[str]:
    """Candidate sections for one question, in document order (no ranking is shown to the judge)."""
    gold = list(dict.fromkeys(g["section_id"] for g in r["gold"]))
    cand: set[str] = set(gold) | set(bm25.top(r["question"], POOL_BM25_K)) | set(named_ids(r["question"], corpus)[1])
    for sid in gold:
        cand |= set(corpus.groups[corpus.sections[sid]["group_id"]]["section_ids"])
        linked = Counter(d for s, d in corpus.refs if s == sid) + Counter(s for s, d in corpus.refs if d == sid)
        cand |= {x for x, _ in linked.most_common(POOL_REF_K)}
    key = r.get("twin_key")
    if key in TWINS:
        cand |= set(TWINS[key]["piston"]) | set(TWINS[key]["turbine"])
    return sorted(cand, key=corpus.order.get)


def pool(paths: list[str | Path], out: str | Path | None = None) -> list[dict]:
    corpus = light_corpus()
    bm25 = SectionBM25(corpus)
    rows = []
    for p in paths:
        for r in read(p):
            c = pool_for(r, corpus, bm25)
            rows.append({"question_id": r["question_id"], "candidates": c,
                         "n_chars": sum(len(corpus.sections[s]["text"]) for s in c)})
    if out:
        write(out, rows)
    return rows


# ---------------------------------------------------------------- references for multi_section


def _sentence(text: str, i: int, j: int) -> str:
    a = max(text.rfind(". ", 0, i) + 2 if text.rfind(". ", 0, i) >= 0 else 0, text.rfind("\n", 0, i) + 1)
    ends = [x for x in (text.find(". ", j), text.find("\n", j)) if x >= 0]
    b = min(ends) + 1 if ends else len(text)
    return " ".join(text[a:b].split())


def cross_group_refs(section: str | None = None) -> list[dict]:
    """One row per (citing section, cited section) in different groups: the first reference and its sentence.
    ``(See AMC E ...)`` pointers are dropped (they stay inside a group anyway)."""
    corpus = light_corpus()
    out, seen = [], set()
    for r in read_jsonl(corpus.data_dir / "cross_refs.jsonl"):
        src, dst = r["src_section_id"], r["dst_section_id"]
        if r["index"] != "units" or (src, dst) in seen:
            continue
        if corpus.sections[src]["group_id"] == corpus.sections[dst]["group_id"]:
            continue
        if section and section not in (src, dst):
            continue
        body = corpus.sections[src]["body"]
        i = body.find(r["ref_text"])
        sent = _sentence(body, i, i + len(r["ref_text"])) if i >= 0 else ""
        if re.fullmatch(r"\(See [^)]*\)\.?", sent):
            continue
        seen.add((src, dst))
        out.append({"src": src, "dst": dst, "ref_text": r["ref_text"], "sentence": sent})
    return out


# ---------------------------------------------------------------- stats


def stats(paths: list[str | Path]) -> dict:
    corpus = light_corpus()
    rows = [r for p in paths for r in read(p)]
    out: dict = {"n": len(rows)}
    for fam in FAMILIES:
        rs = [r for r in rows if r.get("family") == fam]
        if not rs:
            continue
        ov = sorted(r["lexical_overlap"] for r in rs if isinstance(r.get("lexical_overlap"), (int, float)))
        fields = ("question_type", "difficulty", "engine_type", "asks_about_engine_type", "gold_kind", "subpart",
                  *FAMILY_FIELDS[fam])
        out[fam] = {
            "n": len(rs),
            "overlap_mean": round(sum(ov) / len(ov), 3) if ov else None,
            "overlap_median": ov[len(ov) // 2] if ov else None,
            "gold_sections": len({g["section_id"] for r in rs for g in r["gold"]}),
            **{f: dict(Counter(str(r.get(f)) for r in rs).most_common()) for f in fields if f not in ("twin_of", "variant_of")},
            "verification": {k: dict(Counter(str((r.get("verification") or {}).get(k)) for r in rs)) for k in VERIFICATION_KEYS},
            "review": dict(Counter((r.get("review") or {}).get("status") for r in rs)),
        }
    v1 = read(V1_QUESTIONS)
    covered = {g["section_id"] for q in v1 for g in q["gold"]} | {g["section_id"] for r in rows for g in r["gold"]}
    out["sections_with_a_question_v1_plus_set2"] = f"{len(covered)} of {len(corpus.sections)}"
    return out


# ---------------------------------------------------------------- CLI


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m csrag.qgen", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fill", help="write derived fields in place")
    a.add_argument("paths", nargs="+")
    a.add_argument("--generator", help="model id of the generating session, written to `generator`")
    a = sub.add_parser("check", help="validate set-2 files")
    a.add_argument("paths", nargs="+")
    a.add_argument("--final", action="store_true", help="also require every verification check to pass")
    a = sub.add_parser("pool", help="candidate sections for the single-source judge")
    a.add_argument("paths", nargs="+")
    a.add_argument("--out", help="JSONL output (default: data/eval/set2/pools/<file name>)")
    a = sub.add_parser("stats")
    a.add_argument("paths", nargs="+")
    a = sub.add_parser("overlap", help="overlap of a question with sections, and the words that are not shared")
    a.add_argument("question")
    a.add_argument("section_ids", nargs="+")
    a = sub.add_parser("refs", help="cross-group references (candidates for multi_section reference questions)")
    a.add_argument("--section", help="only references from or to this section")
    sub.add_parser("twins", help="print the twin table")
    args = ap.parse_args(argv)

    if args.cmd == "fill":
        for p in args.paths:
            print(f"{p}: {fill(p, args.generator)} records")
        return 0
    if args.cmd == "check":
        res = check(args.paths, final=args.final)
        print(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"{res['n']} questions, {len(res['errors'])} with errors, {len(res['warnings'])} with warnings", file=sys.stderr)
        return 1 if res["errors"] else 0
    if args.cmd == "pool":
        for p in args.paths:
            out = args.out or SET2 / "pools" / Path(p).name
            rows = pool([p], out)
            print(f"{p}: {len(rows)} pools -> {out} (mean {sum(len(r['candidates']) for r in rows) / max(1, len(rows)):.1f} sections)")
        return 0
    if args.cmd == "stats":
        print(json.dumps(stats(args.paths), indent=1, ensure_ascii=False))
        return 0
    if args.cmd == "overlap":
        ov, shared, other = lexical_overlap(args.question, args.section_ids)
        print(json.dumps({"overlap": ov, "shared": shared, "not_shared": other, "ids_named": named_ids(args.question)},
                         ensure_ascii=False))
        return 0
    if args.cmd == "refs":
        for row in cross_group_refs(args.section):
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if args.cmd == "twins":
        print(json.dumps(TWINS, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_main())
