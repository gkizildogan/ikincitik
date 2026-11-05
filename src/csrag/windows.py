"""Section-bounded 500-token windows with a >= 50-token overlap (plan section 3.3).

- Counted in bge-m3 tokens (no special tokens), on the section ``body``.
- A section of <= ``size`` tokens is one window holding the whole section.
- A longer section is cut into windows of at most ``size`` tokens. A window
  ends at the last line end or sentence end within its final ``tail`` tokens,
  or at a word end if there is none.
- The next window starts ``overlap`` tokens before the previous one ends,
  moved back to the start of a word, so consecutive windows share at least
  ``overlap`` tokens.
- ``text`` = the section's header line + the window body. The header is not
  counted in the 500 tokens.
- Windows never cross a section, so every window has exactly one ``section_id``.
"""

from __future__ import annotations

from typing import Callable

from .enrich import engine_type, normalize_ws, raw_to_norm

SENTENCE_END = ".!?"


def _boundary(body: str, lo: int, hi: int, start: int) -> int:
    """Best window end in (lo, hi]: last line/sentence end, else last word end, else ``hi``."""
    best = -1
    for p in range(hi, lo, -1):
        if p < len(body) and body[p] == "\n":
            best = p
            break
        if body[p - 1] in SENTENCE_END and (p == len(body) or body[p].isspace()):
            best = p
            break
    if best > start:
        return best
    for p in range(hi, lo, -1):
        if p < len(body) and body[p].isspace() and not body[p - 1].isspace():
            return p
    return hi


def _word_start(body: str, pos: int) -> int:
    while pos < len(body) and body[pos].isspace():
        pos += 1
    while pos > 0 and not body[pos - 1].isspace():
        pos -= 1
    return pos


def cut_section(body: str, tokenizer, size: int = 500, overlap: int = 50, tail: int = 100) -> list[tuple[int, int]]:
    """Raw character spans [start, end) of the windows of one section body."""
    count: Callable[[str], int] = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
    if count(body) <= size:
        return [(0, len(body.rstrip()))]
    offs = tokenizer.encode(body, add_special_tokens=False).offsets
    spans: list[tuple[int, int]] = []
    start = _word_start(body, 0)
    while True:
        if count(body[start:].rstrip()) <= size:
            spans.append((start, len(body.rstrip())))
            break
        t0 = next(t for t, (_, e) in enumerate(offs) if e > start)
        limit = min(t0 + size, len(offs))
        hi = offs[limit - 1][1]
        while True:
            lo = offs[max(t0, limit - tail)][0]
            end = _boundary(body, lo, hi, start)
            piece = body[start:end].rstrip()
            if count(piece) <= size:
                break
            limit -= 1  # re-tokenising the slice can differ by a token or two
            hi = offs[limit - 1][1]
        end = start + len(piece)
        spans.append((start, end))
        t_end = sum(1 for (_, e) in offs if e <= end)
        nxt = _word_start(body, offs[max(t_end - overlap, t0 + 1)][0])
        if nxt <= start:
            raise RuntimeError(f"window did not advance at char {start}")
        start = nxt
    return spans


def make_windows(sections: list[dict], tokenizer, size: int = 500, overlap: int = 50, tail: int = 100) -> list[dict]:
    """``data/windows_500.jsonl`` records, in corpus order."""
    out = []
    for s in sorted(sections, key=lambda s: s["order"]):
        body = s["body"]
        spans = cut_section(body, tokenizer, size, overlap, tail)
        r2n = raw_to_norm(body)
        n = len(spans)
        for i, (a, b) in enumerate(spans):
            wbody = body[a:b]
            header = s["title"] + (f" (part {i + 1}/{n})" if n > 1 else "")
            out.append(
                {
                    "window_id": f"{s['section_id']}#w{i + 1}",
                    "section_id": s["section_id"],
                    "group_id": s["group_id"],
                    "kind": s["kind"],
                    "subpart": s["subpart"],
                    "engine_type": engine_type(s["subpart"]),
                    "part_index": i + 1,
                    "part_count": n,
                    "char_start": r2n[a],
                    "char_end": r2n[b],
                    "page_start": s["page_start"],
                    "page_end": s["page_end"],
                    "breadcrumb": f"{s['subpart']} > {s['title']}",
                    "n_tokens_bge": len(tokenizer.encode(wbody, add_special_tokens=False).ids),
                    "body": wbody,
                    "text": f"{header}\n\n{wbody}".strip(),
                }
            )
    return out


def check_windows(windows: list[dict], sections: dict[str, dict], tokenizer, size: int = 500, overlap: int = 50) -> dict:
    """Plan section 8 window checks; every list in the result should be empty."""
    count = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
    by_sec: dict[str, list[dict]] = {}
    for w in windows:
        by_sec.setdefault(w["section_id"], []).append(w)
    problems: dict[str, list] = {
        "sections_without_windows": [sid for sid in sections if sid not in by_sec],
        "not_covered": [],
        "span_text_mismatch": [],
        "too_long": [],
        "overlap_too_small": [],
        "short_section_not_single": [],
    }
    for sid, ws in by_sec.items():
        ws.sort(key=lambda w: w["part_index"])
        nb = normalize_ws(sections[sid]["body"])
        for w in ws:
            if nb[w["char_start"] : w["char_end"]] != normalize_ws(w["body"]):
                problems["span_text_mismatch"].append(w["window_id"])
            if w["n_tokens_bge"] > size or count(w["body"]) > size:
                problems["too_long"].append((w["window_id"], w["n_tokens_bge"]))
        # coverage: the windows' spans together cover [0, len(nb))
        if ws[0]["char_start"] != 0 or ws[-1]["char_end"] != len(nb):
            problems["not_covered"].append(sid)
        for a, b in zip(ws, ws[1:]):
            if b["char_start"] > a["char_end"]:
                problems["not_covered"].append(sid)
            shared = nb[b["char_start"] : a["char_end"]]
            if count(shared) < overlap:
                problems["overlap_too_small"].append((a["window_id"], b["window_id"], count(shared)))
        if count(sections[sid]["body"]) <= size and len(ws) != 1:
            problems["short_section_not_single"].append(sid)
    return problems


def window_stats(windows: list[dict]) -> dict:
    import numpy as np

    n = np.array([w["n_tokens_bge"] for w in windows])
    split = [w for w in windows if w["part_count"] > 1]
    return {
        "windows": len(windows),
        "sections": len({w["section_id"] for w in windows}),
        "single_window_sections": sum(1 for w in windows if w["part_count"] == 1),
        "boundaries": sum(1 for w in split if w["part_index"] > 1),
        "tokens_median": float(np.median(n)),
        "tokens_p90": float(np.percentile(n, 90)),
        "tokens_max": int(n.max()),
        "tokens_total": int(n.sum()),
    }
