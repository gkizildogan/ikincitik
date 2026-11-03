"""Assemble cached region outputs into sections, CS+AMC groups and chunks.

Outputs (all JSONL, one record per line):
  sections.jsonl  one record per CS-E / AMC / appendix section, full text
  groups.jsonl    one record per CS-E group: CS text followed by its AMCs (each under its own "## " heading)
  chunks.jsonl    retrieval units; a section is split only if it is longer
                  than ``max_tokens``, and only at its own paragraph markers
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .segment import AMDT_RE, Section

HEADER_LEAK_RE = re.compile(r"Annex to ED Decision|Page \d+ of 268|CS-E\s+[—–-]\s+Amendment 9")

# ---------------------------------------------------------------- stitching


def _is_row(ln: str) -> bool:
    return ln.lstrip().startswith("|")


def _is_sep(ln: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{3,}", ln)) and set(ln.strip()) <= set("|-: ")


def _cols(ln: str) -> int:
    r = ln.strip().strip("|")
    return len(r.split("|"))


def _table_header(lines: list[str]) -> str | None:
    """Header row of the table that ``lines`` ends with."""
    i = len(lines) - 1
    while i > 0 and _is_row(lines[i - 1]):
        i -= 1
    if i + 1 < len(lines) and _is_sep(lines[i + 1]):
        return lines[i]
    return None


def stitch(parts: Iterable[str]) -> str:
    """Join the Markdown of consecutive regions of one section.

    - A table that continues onto the next page becomes one table: a repeated
      header row + separator at the top of the continuation is dropped.
    - A sentence cut by the page break (no closing punctuation, next part
      starts lower-case) is re-joined into one line.
    """
    out: list[str] = []
    for part in parts:
        lines = [ln for ln in part.splitlines()]
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        if not out:
            out = lines
            continue
        prev = out[-1]
        first = lines[0]
        if _is_row(prev) and _is_row(first):
            hdr = _table_header(out)
            if len(lines) > 1 and _is_sep(lines[1]):
                if hdr is not None and _norm_row(first) == _norm_row(hdr):
                    lines = lines[2:]  # repeated header on the new page
                elif _cols(first) != _cols(prev):
                    out.append("")  # a different table starts here
            out.extend(lines)
        elif prev.strip() and not re.search(r"[.:;!?|)\]]$", prev.strip()) and re.match(r"^[a-z]", first):
            out[-1] = prev.rstrip() + " " + first.strip()
            out.extend(lines[1:])
        else:
            out.extend(lines)
    return "\n".join(out).strip()


def _norm_row(row: str) -> str:
    return re.sub(r"\s+", " ", row).strip().lower()


# ---------------------------------------------------------------- indentation

_MARKERS = [
    ("paren_lower", re.compile(r"^\(([a-z])\)(?=\s|$)")),
    ("paren_roman", re.compile(r"^\(((?:x{0,3})(?:ix|iv|v?i{0,3}))\)(?=\s|$)")),
    ("paren_digit", re.compile(r"^\((\d{1,3})\)(?=\s|$)")),
    ("paren_upper", re.compile(r"^\(([A-Z])\)(?=\s|$)")),
    ("lower_dot", re.compile(r"^([a-z])\.(?=\s)")),
    ("upper_dot", re.compile(r"^([A-Z])\.(?=\s)")),
    ("decimal", re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\.?(?=\s)")),
    ("digit_dot", re.compile(r"^(\d{1,2})\.(?=\s)")),
    ("dash", re.compile(r"^(-)\s")),
]
_ROMAN = {"i": 1, "v": 5, "x": 10}
_FIRST = {"paren_lower": "a", "paren_roman": "i", "paren_digit": "1", "paren_upper": "A",
          "lower_dot": "a", "upper_dot": "A", "digit_dot": "1"}


def _roman(v: str) -> int:
    n = 0
    for i, ch in enumerate(v):
        x = _ROMAN[ch]
        n += -x if i + 1 < len(v) and _ROMAN[v[i + 1]] > x else x
    return n


def _follows(t: str, prev: str, v: str) -> bool:
    """True if marker value ``v`` is the next item after ``prev`` in a list of type ``t``."""
    if t == "dash":
        return True
    if t == "paren_roman":
        return _roman(v) == _roman(prev) + 1
    if t in ("paren_digit", "digit_dot"):
        return int(v) == int(prev) + 1
    if t == "decimal":
        return v.rsplit(".", 1)[0] == prev.rsplit(".", 1)[0]
    return len(v) == len(prev) == 1 and ord(v) == ord(prev) + 1


def _candidates(text: str) -> list[tuple[str, str, int]]:
    """All (type, value, match_end) readings of a marker at the start of ``text``."""
    return [(t, m.group(1), m.end()) for t, rx in _MARKERS if (m := rx.match(text)) and m.group(1) != ""]


def marker_of(line: str, stack: list[list[str]] | None = None) -> tuple[str, str] | None:
    """(type, value) of the list marker starting ``line``, or None.

    (i), (v), (x) can be letters or roman numerals: a reading that continues
    an open list on ``stack`` wins; otherwise (i) starts a roman list and
    other letters are letters.
    """
    hits = _candidates(line.strip())
    if not hits:
        return None
    if len(hits) > 1 and stack:
        for t, prev in reversed(stack):
            for ht, hv, _ in hits:
                if ht == t and _follows(t, prev, hv):
                    return ht, hv
    if {"paren_lower", "paren_roman"} <= {t for t, _, _ in hits}:
        v = dict((t, v) for t, v, _ in hits)
        return ("paren_roman", v["paren_roman"]) if v["paren_roman"] == "i" else ("paren_lower", v["paren_lower"])
    return hits[0][0], hits[0][1]


def _place(stack: list[list[str]], t: str, v: str) -> int:
    """Update the open-list stack for marker (t, v); return its depth.

    - continues an open list (v follows that list's last item) -> that level
    - a list's first item ((1), (a), (i), (A)) that continues nothing -> new deeper level,
      even if the same style is already open higher up (CS-E nests (a)(1)(i)(A)(1))
    - otherwise (numbering skipped or style seen before) -> nearest open list of that style
    """
    for i in range(len(stack) - 1, -1, -1):
        if stack[i][0] == t and _follows(t, stack[i][1], v):
            del stack[i + 1 :]
            stack[i][1] = v
            return i
    if v == _FIRST.get(t) or not any(e[0] == t for e in stack):
        stack.append([t, v])
        return len(stack) - 1
    i = max(j for j, e in enumerate(stack) if e[0] == t)
    del stack[i + 1 :]
    stack[i][1] = v
    return i


def indent_markers(md: str, unit: str = "  ") -> str:
    """Re-create nesting from the item markers (2 spaces per level).
    Continuation lines get their item's content indent; tables and figure
    placeholders stay flush; ### headings close all open lists."""
    stack: list[list[str]] = []
    out = []
    for raw in md.splitlines():
        ln = raw.strip()
        if not ln:
            out.append("")
            continue
        if ln.startswith("#"):
            stack = []
            out.append(ln)
            continue
        if _is_row(ln) or ln.startswith("[Figure omitted"):
            out.append(ln)
            continue
        mk = marker_of(ln, stack)
        if mk is None:
            out.append(unit * len(stack) + ln)
            continue
        depth = _place(stack, *mk)
        rest = ln
        for _ in range(3):  # further markers on the same line, e.g. "(b) (1) The test ..."
            end = next(e for t, v, e in _candidates(rest) if (t, v) == mk)
            rest = rest[end:].lstrip()
            mk = marker_of(rest, stack)
            if mk is None or mk[0] == "dash":
                break
            _place(stack, *mk)
        out.append(unit * depth + ln)
    return "\n".join(out)


def clean_text(md: str) -> str:
    md = AMDT_RE.sub("", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------------------------------------------------------------- sections


def heading_line(s: Section) -> str:
    return f"{s.section_id} {s.heading}".strip()


def assemble_sections(sections: list[Section], region_records: dict[str, list[dict | None]]) -> list[dict]:
    out = []
    for s in sections:
        recs = region_records.get(s.section_id) or []
        missing = [i for i, r in enumerate(recs) if r is None or r.get("status") == "error"]
        body = clean_text(indent_markers(stitch((r or {}).get("markdown", "") for r in recs)))
        statuses = [r.get("status") for r in recs if r]
        out.append(
            {
                "section_id": s.section_id,
                "kind": s.kind,
                "group_id": s.group_id,
                "subpart": s.subpart,
                "subpart_title": s.subpart_title,
                "title": heading_line(s),
                "order": s.order,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "body": body,
                "text": f"## {heading_line(s)}\n\n{body}".strip(),
                "region_status": statuses,
                "incomplete": bool(missing),
            }
        )
    return out


def assemble_groups(section_recs: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for r in sorted(section_recs, key=lambda r: r["order"]):
        g = groups.setdefault(
            r["group_id"],
            {
                "group_id": r["group_id"],
                "subpart": r["subpart"],
                "title": None,
                "section_ids": [],
                "parts": [],
                "page_start": r["page_start"],
                "page_end": r["page_end"],
            },
        )
        if r["kind"] == "cs" or g["title"] is None:
            g["title"] = r["title"]
        g["section_ids"].append(r["section_id"])
        g["parts"].append(r["text"])
        g["page_end"] = max(g["page_end"], r["page_end"])
    out = []
    for g in groups.values():
        parts = g.pop("parts")
        g["text"] = "\n\n".join(parts)  # each part already starts with its own "## <title>"
        out.append(g)
    return out


# ---------------------------------------------------------------- chunking


def default_counter() -> Callable[[str], int]:
    """cl100k token count (a close stand-in for most embedding tokenizers);
    falls back to a word-based estimate if tiktoken has no cached encoding."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    except Exception:
        return lambda s: int(len(s.split()) * 1.35)


@dataclass
class _Block:
    lines: list[str]
    label: str | None  # e.g. "(c)" or a "### " heading text


def _depth(ln: str, unit: str = "  ") -> int:
    n = len(ln) - len(ln.lstrip(" "))
    return n // len(unit)


def _split_blocks(lines: list[str], depth: int) -> list[_Block]:
    """Cut ``lines`` before every marker line at ``depth`` and every heading."""
    blocks: list[_Block] = []
    cur: list[str] = []
    label = None
    for ln in lines:
        st = ln.strip()
        is_head = st.startswith("### ")
        is_mark = bool(st) and _depth(ln) == depth and marker_of(st) is not None and not st.startswith("- ")
        if (is_head or is_mark) and cur:
            blocks.append(_Block(cur, label))
            cur = []
        if is_head:
            label = st[4:].strip()
        elif is_mark:
            label = st.split()[0]
        cur.append(ln)
    if cur:
        blocks.append(_Block(cur, label))
    return blocks


def _split_table_rows(lines: list[str], max_tokens: int, count) -> list[list[str]]:
    """Last resort for one oversized block: cut between lines; inside a table,
    cut between rows and repeat the header + separator in every piece."""
    pieces, cur, hdr = [], [], None
    for i, ln in enumerate(lines):
        if _is_row(ln) and i + 1 < len(lines) and _is_sep(lines[i + 1]) and not (cur and _is_row(cur[-1])):
            hdr = [ln, lines[i + 1]]
        elif not _is_row(ln) and not _is_sep(ln):
            hdr = None
        if cur and count("\n".join(cur + [ln])) > max_tokens:
            pieces.append(cur)
            cur = list(hdr) if (hdr and _is_row(ln) and not _is_sep(ln) and ln not in hdr) else []
        cur.append(ln)
    if cur:
        pieces.append(cur)
    return pieces


def _pieces(lines: list[str], max_tokens: int, count, depth: int = 0, path: tuple = ()) -> list[tuple[list[str], tuple]]:
    """Recursively split until every piece fits, packing small neighbours."""
    if count("\n".join(lines)) <= max_tokens:
        return [(lines, path)]
    blocks = _split_blocks(lines, depth)
    if len(blocks) <= 1:
        if depth < 4:
            return _pieces(lines, max_tokens, count, depth + 1, path)
        return [(p, path) for p in _split_table_rows(lines, max_tokens, count)]
    out: list[tuple[list[str], tuple]] = []
    buf: list[str] = []
    labels: list[str] = []
    for b in blocks:
        cand = buf + b.lines
        if buf and count("\n".join(cand)) > max_tokens:
            out.append((buf, path + (_label(labels),)))
            buf, labels = [], []
        if count("\n".join(b.lines)) > max_tokens:
            if buf:
                out.append((buf, path + (_label(labels),)))
                buf, labels = [], []
            out.extend(_pieces(b.lines, max_tokens, count, depth + 1, path + ((b.label or "…"),)))
            continue
        buf = buf + b.lines
        labels.append(b.label or "")
    if buf:
        out.append((buf, path + (_label(labels),)))
    return out


def _label(labels: list[str]) -> str:
    labels = [l for l in labels if l]
    if not labels:
        return "intro"
    return labels[0] if len(labels) == 1 else f"{labels[0]}–{labels[-1]}"


def make_chunks(section_recs: list[dict], max_tokens: int = 1500, count: Callable[[str], int] | None = None) -> list[dict]:
    count = count or default_counter()
    chunks = []
    for r in section_recs:
        crumb = f"{r['subpart']} > {r['title']}"
        lines = r["body"].splitlines()
        pieces = _pieces(lines, max_tokens, count)
        n = len(pieces)
        for i, (pl, path) in enumerate(pieces):
            body = "\n".join(pl).strip("\n")
            sub = " > ".join(p for p in path if p)
            header = r["title"] + ((f" — {sub}" if sub else "") + f" (part {i + 1}/{n})" if n > 1 else "")
            chunks.append(
                {
                    "chunk_id": r["section_id"] if n == 1 else f"{r['section_id']}#{i + 1:02d}",
                    "section_id": r["section_id"],
                    "group_id": r["group_id"],
                    "kind": r["kind"],
                    "subpart": r["subpart"],
                    "breadcrumb": crumb + (f" > {sub}" if n > 1 and sub else ""),
                    "part_index": i + 1,
                    "part_count": n,
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "n_tokens": count(body),
                    "body": body,
                    "text": f"{header}\n\n{body}".strip(),
                }
            )
    return chunks


# ---------------------------------------------------------------- checks / io


def check_outputs(section_recs: list[dict], chunks: list[dict]) -> dict:
    """Boundary and cleanliness checks; every list in the result should be empty."""
    by_sec: dict[str, list[dict]] = {}
    for c in chunks:
        by_sec.setdefault(c["section_id"], []).append(c)
    not_reconstructed = []
    for r in section_recs:
        cs = sorted(by_sec.get(r["section_id"], []), key=lambda c: c["part_index"])
        joined = "\n".join(c["body"] for c in cs)
        norm = lambda s: re.sub(r"\s+", " ", s).strip()
        if norm(joined) != norm(r["body"]):
            # repeated table headers in split tables are the only allowed difference
            src_lines = set(r["body"].splitlines())
            if any(ln not in src_lines for c in cs for ln in c["body"].splitlines()):
                not_reconstructed.append(r["section_id"])
    return {
        "amdt_left": [r["section_id"] for r in section_recs if AMDT_RE.search(r["text"])],
        "header_leaks": [r["section_id"] for r in section_recs if HEADER_LEAK_RE.search(r["body"])],
        "empty_sections": [r["section_id"] for r in section_recs if not r["body"].strip()],
        "incomplete_sections": [r["section_id"] for r in section_recs if r["incomplete"]],
        "chunks_not_from_own_section": not_reconstructed,
    }


def write_jsonl(recs: Iterable[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


class Corpus:
    """Lookup helpers over the written JSONL files.

    >>> corpus = Corpus("data")
    >>> corpus.get_group("CS-E 20")["text"]       # CS-E 20 + AMC E 20 + AMC E 20(f)
    >>> corpus.expand(hit_chunk_id, to="section")  # full section of a retrieved chunk
    """

    def __init__(self, data_dir: str | Path = "data"):
        d = Path(data_dir)
        self.sections = {r["section_id"]: r for r in read_jsonl(d / "sections.jsonl")}
        self.groups = {r["group_id"]: r for r in read_jsonl(d / "groups.jsonl")}
        self.chunks = {r["chunk_id"]: r for r in read_jsonl(d / "chunks.jsonl")}

    def get_section(self, section_id: str) -> dict:
        return self.sections[section_id]

    def get_group(self, group_id: str) -> dict:
        return self.groups[group_id]

    def group_sections(self, group_id: str, kind: str | None = None) -> list[dict]:
        ids = self.groups[group_id]["section_ids"]
        return [self.sections[i] for i in ids if kind is None or self.sections[i]["kind"] == kind]

    def expand(self, chunk_id: str, to: str = "group") -> dict:
        c = self.chunks[chunk_id]
        return self.get_group(c["group_id"]) if to == "group" else self.get_section(c["section_id"])
