"""In-memory corpus for both indexes (plan section 3.2).

The JSONL files from notebooks 00-04 are canonical; there is no database. The
corpus is ~155k bge-m3 tokens, so every lookup is a dictionary lookup.

>>> corpus = Corpus("data")
>>> corpus.units("win500")["CS-E 740#w3"]["text"]
>>> corpus.units_of("CS-E 740", "units")        # its base units, in order
>>> corpus.refs_of("CS-E 740#w3", "win500")      # sections it references, in order of mention
>>> corpus.merge_spans(["CS-E 740#w3", "CS-E 740#w4"], "win500")  # one passage
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from csparse.merge import Corpus as _BaseCorpus
from csparse.merge import read_jsonl

from .enrich import engine_type, extract_refs, normalize_map, norm_to_raw_span, unit_spans

INDEXES = ("units", "win500")
COLLECTIONS = {"units": "cse_units_v1", "win500": "cse_win500_v1"}
UNIT_FILES = {"units": "chunks.jsonl", "win500": "windows_500.jsonl"}

DEFAULT_TOKENIZER = os.environ.get(
    "BGE_M3_TOKENIZER",
    "../etoeragcb/model-cache/huggingface/hub/models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json",
)


@lru_cache(maxsize=4)
def bge_tokenizer(path: str | None = None):
    """bge-m3's ``tokenizer.json`` through ``tokenizers`` (no torch needed)."""
    from tokenizers import Tokenizer

    p = Path(path or DEFAULT_TOKENIZER)
    if not p.is_file():
        raise FileNotFoundError(f"bge-m3 tokenizer.json not found at {p}; set BGE_M3_TOKENIZER")
    tok = Tokenizer.from_file(str(p))
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text: str, path: str | None = None) -> int:
    """bge-m3 tokens of ``text`` without the special tokens."""
    return len(bge_tokenizer(path).encode(text, add_special_tokens=False).ids)


class Corpus(_BaseCorpus):
    """``csparse.merge.Corpus`` plus the windows, spans, cross-references and token counts.

    Every lookup takes ``index="units" | "win500"``. A unit record (base unit or
    window) always has: ``unit_id, index, section_id, group_id, kind, subpart,
    engine_type, part_index, part_count, char_start, char_end, page_start,
    page_end, breadcrumb, n_tokens_bge, body, text, cross_refs``.
    """

    def __init__(self, data_dir: str | Path = "data", tokenizer_path: str | None = None):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.tokenizer_path = tokenizer_path
        self.section_ids = [s["section_id"] for s in sorted(self.sections.values(), key=lambda s: s["order"])]
        self._norm: dict[str, tuple[str, list[int]]] = {}
        self._tok_cache: dict[str, int] = {}
        self._units: dict[str, dict[str, dict]] = {"units": self._base_units()}
        wpath = self.data_dir / UNIT_FILES["win500"]
        self.has_windows = wpath.is_file()
        if self.has_windows:
            self._units["win500"] = self._windows(read_jsonl(wpath))
        self._refs = self._load_refs()
        for index, units in self._units.items():
            for uid, u in units.items():
                u["cross_refs"] = self._refs[index].get(uid, [])
        self._by_section: dict[str, dict[str, list[str]]] = {}
        for index, units in self._units.items():
            d: dict[str, list[str]] = {}
            for uid, u in units.items():
                d.setdefault(u["section_id"], []).append(uid)
            for sid in d:
                d[sid].sort(key=lambda i: units[i]["part_index"])
            self._by_section[index] = d

    # ------------------------------------------------------------ loading

    def _base_units(self) -> dict[str, dict]:
        spans = unit_spans(self.chunks.values(), self.sections)
        out = {}
        for cid, c in self.chunks.items():
            u = {k: v for k, v in c.items() if k != "chunk_id"}
            u["unit_id"] = cid
            u["index"] = "units"
            u["engine_type"] = engine_type(c["subpart"])
            u.update(spans[cid])
            u["n_tokens_bge"] = self.n_tokens(c["body"])
            out[cid] = u
        return out

    def _windows(self, rows: list[dict]) -> dict[str, dict]:
        out = {}
        for w in rows:
            u = dict(w)
            u["unit_id"] = w["window_id"]
            u["index"] = "win500"
            u.setdefault("span_exact", True)
            out[u["unit_id"]] = u
        return out

    def _load_refs(self) -> dict[str, dict[str, list[str]]]:
        """``data/cross_refs.jsonl`` if notebook 04 wrote it, else extracted now (same function)."""
        refs: dict[str, dict[str, list[str]]] = {i: {} for i in self._units}
        path = self.data_dir / "cross_refs.jsonl"
        if path.is_file():
            for r in sorted(read_jsonl(path), key=lambda r: (r["index"], r["src_unit_id"], r["order"])):
                if r["index"] in refs:
                    refs[r["index"]].setdefault(r["src_unit_id"], []).append(r["dst_section_id"])
            missing = [i for i in self._units if not any(True for _ in refs[i])]
        else:
            missing = list(self._units)
        for index in missing:
            for uid, u in self._units[index].items():
                refs[index][uid] = [d for d, _ in extract_refs(u["body"], self.section_ids, u["section_id"])]
        return refs

    # ------------------------------------------------------------ lookups

    def indexes(self) -> list[str]:
        return list(self._units)

    def units(self, index: str) -> dict[str, dict]:
        if index not in self._units:
            raise KeyError(f"index {index!r} not loaded (windows_500.jsonl missing? run notebook 04)")
        return self._units[index]

    def unit(self, unit_id: str, index: str) -> dict:
        return self.units(index)[unit_id]

    def units_of(self, section_id: str, index: str) -> list[dict]:
        """The units or windows of a section, in order."""
        units = self.units(index)
        return [units[i] for i in self._by_section[index].get(section_id, [])]

    def units_of_group(self, group_id: str, index: str) -> list[dict]:
        return [u for sid in self.groups[group_id]["section_ids"] for u in self.units_of(sid, index)]

    def refs_of(self, unit_id: str, index: str) -> list[str]:
        """The sections a unit or window references, in order of mention."""
        return list(self.units(index)[unit_id]["cross_refs"])

    def amcs_of(self, section_id: str) -> list[str]:
        """The AMC sections in the same group as a CS section (empty for non-CS sections)."""
        s = self.sections[section_id]
        if s["kind"] != "cs":
            return []
        return [i for i in self.groups[s["group_id"]]["section_ids"] if self.sections[i]["kind"] == "amc"]

    # ------------------------------------------------------------ tokens and spans

    def n_tokens(self, text: str) -> int:
        """bge-m3 token count (no special tokens), cached by text."""
        n = self._tok_cache.get(text)
        if n is None:
            n = count_tokens(text, self.tokenizer_path)
            self._tok_cache[text] = n
        return n

    def section_norm(self, section_id: str) -> tuple[str, list[int]]:
        """(whitespace-normalised body, normalised -> raw position map) of a section."""
        if section_id not in self._norm:
            self._norm[section_id] = normalize_map(self.sections[section_id]["body"])
        return self._norm[section_id]

    def span_text(self, section_id: str, char_start: int, char_end: int) -> str:
        """Raw section text (newlines and Markdown kept) for a normalised span."""
        _, idx = self.section_norm(section_id)
        a, b = norm_to_raw_span(idx, char_start, char_end)
        return self.sections[section_id]["body"][a:b]

    def merge_spans(self, unit_ids: Iterable[str], index: str) -> list[dict]:
        """Join overlapping units of the same section into one passage.

        Passages come in order of their first member's position in ``unit_ids``.
        A passage of one unit keeps that unit's ``text``; a merged passage is the
        section's header line plus the raw text of the joined span, so overlap
        text is never handed over twice.
        """
        units = self.units(index)
        first: dict[str, int] = {}
        for pos, uid in enumerate(unit_ids):
            first.setdefault(uid, pos)
        by_sec: dict[str, list[dict]] = {}
        for uid in first:
            u = units[uid]
            by_sec.setdefault(u["section_id"], []).append(u)
        passages = []
        for sid, us in by_sec.items():
            us.sort(key=lambda u: (u["char_start"], u["char_end"]))
            clusters: list[list[dict]] = []
            end = -1
            for u in us:
                if clusters and u["char_start"] < end:
                    clusters[-1].append(u)
                    end = max(end, u["char_end"])
                else:
                    clusters.append([u])
                    end = u["char_end"]
            for cl in clusters:
                cs, ce = min(u["char_start"] for u in cl), max(u["char_end"] for u in cl)
                if len(cl) == 1:
                    text = cl[0]["text"]
                else:
                    text = f"{self.sections[sid]['title']}\n\n{self.span_text(sid, cs, ce)}"
                passages.append(
                    {
                        "section_id": sid,
                        "unit_ids": [u["unit_id"] for u in sorted(cl, key=lambda u: first[u["unit_id"]])],
                        "char_start": cs,
                        "char_end": ce,
                        "text": text,
                        "n_tokens": self.n_tokens(text),
                        "first": min(first[u["unit_id"]] for u in cl),
                    }
                )
        passages.sort(key=lambda p: p["first"])
        return passages

    def context_tokens(self, unit_ids: Iterable[str], index: str) -> int:
        """Tokens handed over for a set of units, after merging overlaps."""
        return sum(p["n_tokens"] for p in self.merge_spans(unit_ids, index))
