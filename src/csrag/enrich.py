"""Deterministic enrichment before indexing (plan section 3.3, notebook 04).

- ``engine_type`` from the subpart: A -> all, B/C -> piston, D/E/F -> turbine
- whitespace normalisation with a position map, used for every character span
  (base units, windows and gold quotes all record spans in their section's
  whitespace-normalised ``body``)
- ``cross_refs``: regex-extracted references to other CS-E / AMC sections,
  normalised to section ids that exist, in order of mention
- ``point_id`` (uuid5 of the unit id) and ``corpus_version`` (sha256 of a unit file)
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Iterable

ENGINE_TYPE = {
    "SUBPART A": "all",
    "SUBPART B": "piston",
    "SUBPART C": "piston",
    "SUBPART D": "turbine",
    "SUBPART E": "turbine",
    "SUBPART F": "turbine",
}

# Fixed namespace: point ids are stable across re-ingestion and machines.
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ikincitik/csrag/points")


def engine_type(subpart: str) -> str:
    return ENGINE_TYPE[subpart]


def point_id(unit_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, unit_id))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def corpus_version(unit_file: str | Path) -> str:
    """sha256 of the index's unit file (``chunks.jsonl`` or ``windows_500.jsonl``)."""
    return file_sha256(unit_file)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- whitespace normalisation


def normalize_ws(s: str) -> str:
    """Every whitespace run -> one space, trimmed. Same as ``re.sub(r"\\s+", " ", s).strip()``."""
    return " ".join(s.split())


def normalize_map(s: str) -> tuple[str, list[int]]:
    """Normalised text and, for each normalised character, its index in ``s``.

    A collapsed whitespace run maps to its last raw whitespace character.
    """
    out: list[str] = []
    idx: list[int] = []
    pending = False
    for i, ch in enumerate(s):
        if ch.isspace():
            pending = bool(out)
            continue
        if pending:
            out.append(" ")
            idx.append(i - 1)
            pending = False
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def raw_to_norm(s: str) -> list[int]:
    """``r2n[r]`` = normalised position of raw position ``r`` (0..len(s)).

    For a raw span [a, b) that starts and ends on non-space characters,
    ``norm[r2n[a]:r2n[b]] == normalize_ws(s[a:b])``.
    """
    _, idx = normalize_map(s)
    r2n = [0] * (len(s) + 1)
    j = 0
    for r in range(len(s) + 1):
        while j < len(idx) and idx[j] < r:
            j += 1
        r2n[r] = j
    return r2n


def norm_to_raw_span(idx: list[int], start: int, end: int) -> tuple[int, int]:
    """Raw [a, b) covering the non-empty normalised span [start, end)."""
    if not 0 <= start < end <= len(idx):
        raise ValueError(f"bad normalised span [{start}, {end}) for {len(idx)} chars")
    return idx[start], idx[end - 1] + 1


# ---------------------------------------------------------------- spans of base units


def unit_spans(chunks: Iterable[dict], sections: dict[str, dict], tail_chars: int = 200) -> dict[str, dict]:
    """``char_start``/``char_end`` of every base unit in its section's normalised body.

    Base units partition their section in order, so each one is searched from
    the end of the previous one. A unit that starts with a repeated table
    header (``csparse.merge._split_table_rows``) is not a contiguous substring;
    it is located by its tail and starts where the previous unit ended
    (``span_exact = False``).
    """
    by_sec: dict[str, list[dict]] = {}
    for c in chunks:
        by_sec.setdefault(c["section_id"], []).append(c)
    out: dict[str, dict] = {}
    for sid, cs in by_sec.items():
        nb = normalize_ws(sections[sid]["body"])
        cursor = 0
        for c in sorted(cs, key=lambda c: c["part_index"]):
            t = normalize_ws(c["body"])
            i = nb.find(t, cursor)
            if i >= 0:
                span = (i, i + len(t), True)
            else:
                tail = t[-tail_chars:]
                j = nb.find(tail, cursor)
                if j < 0:
                    raise ValueError(f"{c['chunk_id']}: body not found in section {sid}")
                start = cursor + 1 if nb[cursor : cursor + 1] == " " else cursor
                span = (start, j + len(tail), False)
            out[c["chunk_id"]] = {"char_start": span[0], "char_end": span[1], "span_exact": span[2]}
            cursor = span[1]
    return out


# ---------------------------------------------------------------- cross-references

REF_RE = re.compile(r"(?<![\w-])(CS-E|AMC E|AMC to CS-E) (\d+)((?:\([a-z0-9]+\))*)")


def _amc_lookup(base: str, ids: list[str], idset: set[str]) -> list[str]:
    for prefix in ("AMC E ", "AMC to CS-E "):
        if prefix + base in idset:
            return [prefix + base]
        compound = [i for i in ids if i.startswith(prefix + base + " and ")]
        if compound:
            return compound
    return [i for i in ids if i.startswith(("AMC E " + base + "(", "AMC to CS-E " + base + "("))]


def normalize_ref(family: str, num: str, parens: str, section_ids: list[str], idset: set[str] | None = None) -> list[str]:
    """Existing section ids for one reference, e.g.

    - ``CS-E 740(c)(4)`` -> ``CS-E 740`` (CS sections are not split into paragraphs)
    - ``AMC E 740(c)(4)`` -> ``AMC E 740(c)(4)``
    - ``AMC E 40(b)(3)`` -> ``AMC E 40(b)(3) and (b)(4)``
    - ``AMC E 520`` (no such section) -> all of ``AMC E 520(...)``
    """
    idset = idset if idset is not None else set(section_ids)
    plist = re.findall(r"\([a-z0-9]+\)", parens)
    if family == "CS-E":
        for k in range(len(plist), -1, -1):
            cand = f"CS-E {num}" + "".join(plist[:k])
            if cand in idset:
                return [cand]
        return []
    for k in range(len(plist), -1, -1):
        hit = _amc_lookup(num + "".join(plist[:k]), section_ids, idset)
        if hit:
            return hit
    return []


def extract_refs(text: str, section_ids: list[str], own_section_id: str | None = None) -> list[tuple[str, str]]:
    """``[(dst_section_id, ref_text)]`` in order of first mention; self-references dropped."""
    idset = set(section_ids)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in REF_RE.finditer(text):
        for dst in normalize_ref(m.group(1), m.group(2), m.group(3), section_ids, idset):
            if dst == own_section_id or dst in seen:
                continue
            seen.add(dst)
            out.append((dst, m.group(0)))
    return out


def find_section_ids(text: str, section_ids: list[str]) -> list[str]:
    """Section ids named in free text (T1 id lookup), in order of mention."""
    return [dst for dst, _ in extract_refs(text, section_ids, None)]


def cross_ref_records(units: Iterable[dict], index: str, section_ids: list[str]) -> list[dict]:
    """Rows of ``data/cross_refs.jsonl`` for one index (refs are extracted from each unit's body)."""
    rows = []
    for u in units:
        for order, (dst, ref_text) in enumerate(extract_refs(u["body"], section_ids, u["section_id"])):
            rows.append(
                {
                    "index": index,
                    "src_unit_id": u["unit_id"],
                    "src_section_id": u["section_id"],
                    "dst_section_id": dst,
                    "ref_text": ref_text,
                    "order": order,
                }
            )
    return rows
