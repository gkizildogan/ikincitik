"""Engine-type keyword rule for X3 (plan: x2_to_x3_plan.md section 3).

The rule decides from the question text alone whether a question is about piston
or turbine engines, using the lists in ``engine_keywords.toml`` (edited by hand).
The answer feeds the same Qdrant filter as F0 and F1 (``engine_type in {type, all}``);
no LLM and no label is involved.

>>> rules = load()                                  # engine_keywords.toml, validated and normalised
>>> detect("What is the turbofan oil requirement?", rules)
Detection(engine_type='turbine', matched={'turbine': ['turbofan']}, reason='one_type')
>>> rules_hash(rules)                               # goes into the run config

Matching: the question and the keywords are lower-cased, hyphens become spaces and
whitespace is collapsed; every ``ignore.phrases`` entry is removed from the question
(plain text removal); a keyword matches as a whole word with an optional plural
``s`` or ``es``. Exactly one type found -> that type; none or both -> no filter.

The rules dict is plain data (``engine_types``: type -> sorted unique normalised
keywords, ``ignore``: sorted normalised phrases), so it can be hashed, printed and
passed around. It depends on neither Qdrant nor the corpus.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .runs import sha

DEFAULT_PATH = Path("engine_keywords.toml")  # relative to the repo root, like every path in src/
TYPES = ("piston", "turbine")  # the engine_type payload values a table may be named after ("all" is not a table)


def normalise(text: str) -> str:
    """Lower case, hyphens as spaces, whitespace collapsed."""
    return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()


@lru_cache(maxsize=None)
def _pattern(keyword: str) -> re.Pattern:
    """Whole-word pattern of a normalised keyword, with an optional plural ``s`` / ``es``."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?:s|es)?(?![a-z0-9])")


def parse(cfg: dict, source: str = "engine keyword rules") -> dict:
    """Validate a parsed TOML document and return the normalised rules dict. Raises ValueError."""
    unknown = set(cfg) - {"engine_types", "ignore"}
    if unknown:
        raise ValueError(f"{source}: unknown top-level keys {sorted(unknown)} (allowed: engine_types, ignore)")
    tables = cfg.get("engine_types") or {}
    bad = sorted(set(tables) - set(TYPES))
    if bad:
        raise ValueError(f"{source}: engine type table(s) {bad}: a table must be named after an engine_type value, one of {list(TYPES)}")
    if not tables:
        raise ValueError(f"{source}: no [engine_types.<type>] table")
    types: dict[str, list[str]] = {}
    for t, tab in tables.items():
        extra = set(tab) - {"keywords"}
        if extra:
            raise ValueError(f"{source}: [engine_types.{t}] has unknown keys {sorted(extra)} (allowed: keywords)")
        raw = tab.get("keywords")
        if not isinstance(raw, list) or not raw or not all(isinstance(k, str) for k in raw):
            raise ValueError(f"{source}: [engine_types.{t}] needs a non-empty `keywords` list of strings")
        norm = [normalise(k) for k in raw]
        if not all(norm):
            raise ValueError(f"{source}: [engine_types.{t}] has an empty keyword")
        types[t] = sorted(set(norm))
    both = sorted(set(types.get("piston", [])) & set(types.get("turbine", [])))
    if both:
        raise ValueError(f"{source}: keyword(s) listed for both engine types: {both}")
    ign = cfg.get("ignore") or {}
    extra = set(ign) - {"phrases"}
    if extra:
        raise ValueError(f"{source}: [ignore] has unknown keys {sorted(extra)} (allowed: phrases)")
    phrases = ign.get("phrases", [])
    if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
        raise ValueError(f"{source}: [ignore] phrases must be a list of strings")
    norm_ign = [normalise(p) for p in phrases]
    if not all(norm_ign):
        raise ValueError(f"{source}: [ignore] has an empty phrase")
    return {"engine_types": types, "ignore": sorted(set(norm_ign))}


def load(path: str | Path | None = None) -> dict:
    """Read, validate and normalise ``engine_keywords.toml`` (or ``path``)."""
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found (the working directory must be the repo root: {Path.cwd()})")
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    return parse(cfg, str(p))


def rules_hash(rules: dict) -> str:
    """Short hash of the normalised lists: it goes into the run config, so an edited file means a new run id."""
    return sha(rules, 8)


@dataclass(frozen=True)
class Detection:
    engine_type: str | None  # "piston", "turbine" or None (no filter)
    matched: dict[str, list[str]] = field(default_factory=dict)  # type -> keywords found, only for types with a match
    reason: str = "no_cue"  # "one_type", "no_cue" or "both_types"


def detect(question: str, rules: dict) -> Detection:
    """The engine type a question is about, from its text only."""
    text = normalise(question)
    for phrase in rules["ignore"]:
        text = text.replace(phrase, " ")
    matched = {t: [k for k in kws if _pattern(k).search(text)] for t, kws in rules["engine_types"].items()}
    matched = {t: ks for t, ks in matched.items() if ks}
    if len(matched) == 1:
        return Detection(next(iter(matched)), matched, "one_type")
    return Detection(None, matched, "both_types" if matched else "no_cue")
