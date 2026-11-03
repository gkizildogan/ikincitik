"""Grounded VLM extraction: page-region image + PDF text layer -> Markdown.

Qwen sees the rendered region (for layout: lists, indentation, tables,
superscripts) and the region's own text from the PDF (for the exact wording).
It must reproduce that wording and add structure only. Each output is scored
against the source text; outputs that drift are retried once with feedback,
then replaced by a rule-based rendering of the native text.

Regions of one section run in order, because each call gets the tail of the
previous region's output to continue lists and tables across page breaks.
Sections run in parallel. Every result is cached on disk, so a run can be
stopped and resumed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from openai import OpenAI

from .segment import AMDT_RE, Region, Section, norm_ws, region_has_figure, region_png, region_text

SYSTEM_PROMPT = """\
You convert one region of a page from EASA CS-E (Certification Specifications for Engines) into Markdown.

You get two inputs:
1. An IMAGE of the region. Use it ONLY to see layout: paragraph nesting, indentation, tables, columns, bold sub-headings, superscripts/subscripts.
2. SOURCE TEXT taken from the PDF's text layer for the same region. This is the authoritative wording.

Rules:
- Use exactly the words and numbers of the SOURCE TEXT, in reading order. Do not add, drop, summarise, translate or correct anything. Fix only line-break hyphenation and broken line wraps.
- Paragraph hierarchy: start every list item on its own line with its marker followed by its text, e.g. "(a) The Engine must ...", "(1) ...", "(i) ...", "a. ...", "1.2 ...". Do NOT indent anything: nesting is rebuilt from the markers afterwards. A continuation paragraph of an item is a plain line without a marker.
- Ruled tables (cells with borders) -> GitHub Markdown tables with a header row and a |---| separator. One table row per visual row; join text that wraps inside a cell with spaces. If a cell spans several rows or columns, repeat its text in every cell it covers. Keep table titles/captions ("Table 1 ...") as a separate line above the table.
- Two-column term / definition layouts without borders (glossaries) are NOT tables: write each entry on one line as "- **Term**: definition" (the closing ** goes right after the term).
- Superscript exponents: write 10^-7, m^2, etc. Keep subscripts as plain characters as in the source text (t2).
- Bold sub-headings inside the region (e.g. "1. General", "Appendix 1 ...", a bold title line) -> "### <text>".
- Figures, diagrams, charts and photos are skipped: write a single line "[Figure omitted: <its caption from the source text>]", or "[Figure omitted]" if it has no caption. Do not transcribe text that sits inside a figure, but do keep the caption.
- Leave out amendment notes like "[Amdt No: E/9]" and any page header or footer text.
- If PREVIOUS OUTPUT is given, this region continues it from the previous page: carry on at the same list level, and if the previous output ends inside a table, continue with more rows of that table in the same column layout, without repeating the header row or separator.
- Output only the Markdown for this region: no code fences, no comments, no explanations.
"""

PROMPT_VERSION = hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:10]

_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*?)\n?```\s*$", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FIG_RE = re.compile(r"\[Figure omitted:?", re.IGNORECASE)


@dataclass
class VLMConfig:
    # read when the config is created, so load_dotenv() can run after import
    base_url: str = field(default_factory=lambda: os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.environ.get("VLLM_MODEL", "RedHatAI/Qwen3.8-27B-INT4"))
    dpi: int = 150
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 600.0
    tail_lines: int = 15
    min_recall: float = 0.995  # share of source words found in the output
    min_precision: float = 0.995  # share of output words found in the source
    workers: int = 4  # matches max-num-seqs in vllm/serve.yaml
    cache_dir: Path = Path("data/vlm_cache")


# ---------------------------------------------------------------- scoring


def _tokens(s: str) -> Counter:
    # NFKC folds math italics (𝑧𝑚 -> zm), circled letters (Ⓐ -> A), ligatures...
    s = unicodedata.normalize("NFKC", s)
    s = _FIG_RE.sub(" ", s)
    # Hyphens are separators: "30-Second" and a "30-" / "Second" line wrap
    # score the same, and text from neighbouring table columns is never glued.
    s = re.sub(r"<br\s*/?>", " ", s.replace("**", " ")).lower()
    return Counter(_TOKEN_RE.findall(s))


def _drop_repeated_headers(md: str) -> str:
    """Remove table header rows (+ separators) that repeat an earlier header:
    restating column names for a second table is structure, not invented text."""
    lines, seen, out, i = md.splitlines(), set(), [], 0
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if ln.lstrip().startswith("|") and re.match(r"^\s*\|?\s*:?-{3,}", nxt):
            key = re.sub(r"[^a-z0-9]+", " ", ln.lower()).strip()
            if key in seen:
                i += 2
                continue
            seen.add(key)
        out.append(ln)
        i += 1
    return "\n".join(out)


def fidelity(source: str, output: str) -> dict:
    """Bag-of-words recall/precision of output vs. source, plus missing numbers."""
    src = _tokens(source)
    out_lines = _drop_repeated_headers(output).splitlines()
    plain = _tokens("\n".join(l for l in out_lines if not l.lstrip().startswith("|")))
    table = _tokens("\n".join(l for l in out_lines if l.lstrip().startswith("|")))
    # Inside tables, spanning cells are repeated on purpose: a source word that
    # repeats there is never "extra"; a word absent from the source still is.
    out = plain + Counter({t: (c if t not in src else max(1, min(c, src[t] - plain[t]))) for t, c in table.items()})
    inter = sum((src & out).values())
    n_src, n_out = sum(src.values()), sum(out.values())
    missing = sorted((src - out).keys(), key=lambda k: -src[k])
    extra = sorted((out - src).keys(), key=lambda k: -out[k])
    return {
        "recall": inter / n_src if n_src else 1.0,
        "precision": inter / n_out if n_out else (1.0 if not n_src else 0.0),
        "missing": missing[:25],
        "extra": extra[:25],
        "missing_numbers": [t for t in missing if t.isdigit()][:25],
    }


def markdown_issues(md: str) -> list[str]:
    """Structural problems that the bag-of-words score cannot see."""
    issues = []
    lines = md.splitlines()
    for i, ln in enumerate(lines, 1):
        if ln.count("**") % 2:
            issues.append(f"line {i}: unclosed ** bold")
    for i in range(1, len(lines)):
        if lines[i].strip() and lines[i].strip() == lines[i - 1].strip():
            issues.append(f"line {i + 1}: repeated line")
    # every row of one contiguous table block must have the same column count
    block: list[int] = []
    for ln in lines + [""]:
        if ln.lstrip().startswith("|"):
            block.append(_row_cols(ln))
        else:
            if len(set(block)) > 1:
                issues.append(f"table with inconsistent column counts {sorted(set(block))}")
            block = []
    return issues


def _row_cols(row: str) -> int:
    r = row.strip()
    r = r[1:] if r.startswith("|") else r
    r = r[:-1] if r.endswith("|") else r
    return len(re.split(r"(?<!\\)\|", r))


def judge(m: dict, md: str, cfg: VLMConfig, finish_reason: str | None = None) -> str | None:
    """'ok' if the output passes; 'ok_fig' if it only misses words while
    containing a figure placeholder (text drawn inside a vector figure is in
    the text layer but is meant to be skipped); otherwise None."""
    if finish_reason == "length":
        return None
    if passes(m, cfg):
        return "ok"
    if "[Figure omitted" in md and m["precision"] >= cfg.min_precision and not m.get("issues") and m["recall"] >= 0.5:
        return "ok_fig"
    return None


def passes(m: dict, cfg: VLMConfig) -> bool:
    return (
        m["recall"] >= cfg.min_recall
        and m["precision"] >= cfg.min_precision
        and not m["missing_numbers"]
        and not m.get("issues")
    )


# ---------------------------------------------------------------- fallback


_MARKER_RE = re.compile(r"^(\([a-z]{1,4}\)|\([0-9]{1,3}\)|\([ivxlcdm]{1,6}\)|\([A-Z]\)|[a-z]\.|[0-9]{1,2}\.)$")


def native_fallback(text: str) -> str:
    """Rule-based rendering of the text layer: put lone list markers on the
    same line as the text that follows them and re-join wrapped lines. No
    table or nesting reconstruction."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: list[str] = []
    pending = None
    for ln in lines:
        if _MARKER_RE.match(ln):
            pending = ln
            continue
        if pending:
            out.append(f"{pending} {ln}")
            pending = None
        elif out and not re.search(r"[.:;!?]$", out[-1]) and ln[:1].islower():
            out[-1] = f"{out[-1]} {ln}"
        else:
            out.append(ln)
    if pending:
        out.append(pending)
    return "\n".join(out)


# ---------------------------------------------------------------- prompt


def strip_heading(source: str, title: str) -> str:
    """Remove the section heading at the top of the first region (it may wrap
    over several lines). The heading is added back in canonical form during
    assembly, so the VLM never has to reproduce it."""
    target = re.sub(r"\s+", "", title).lower()
    lines = source.splitlines()
    acc = ""
    for i, ln in enumerate(lines[:4]):
        acc += re.sub(r"\s+", "", ln).lower()
        if acc == target:
            return "\n".join(lines[i + 1 :])
        if not target.startswith(acc):
            break
    return source


def build_messages(png: bytes, source: str, section: Section, region_idx: int, prev_tail: str | None, feedback: str | None = None) -> list[dict]:
    b64 = base64.b64encode(png).decode()
    n = len(section.regions)
    parts = [
        f"SECTION: {section.title}",
        f"REGION: page {section.regions[region_idx].page}, part {region_idx + 1} of {n} of this section"
        + (" (the section heading is above this text and is added separately; do not repeat it)" if region_idx == 0 else ""),
    ]
    if prev_tail:
        parts.append(f"PREVIOUS OUTPUT (last lines, from the previous page):\n<<<\n{prev_tail}\n>>>")
    parts.append(f"SOURCE TEXT:\n<<<\n{source}\n>>>")
    if feedback:
        parts.append(feedback)
    parts.append("Write the Markdown for this region now.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "\n\n".join(parts)},
            ],
        },
    ]


def clean_output(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1)
    text = AMDT_RE.sub("", text)
    text = re.sub(r"\s*<br\s*/?>\s*", " ", text)  # line breaks inside table cells
    out = []
    for ln in text.splitlines():
        ln = ln.strip()  # indentation is rebuilt from the markers later
        ln = re.sub(r"^-(?=[^\s-])", "- ", ln)  # "-Term" -> "- Term"
        # "- **Term means ..." -> "- **Term**: means ..."
        if ln.startswith("- **") and ln.count("**") == 1:
            ln = re.sub(r"^- \*\*(.+?)\s+(means\b)", r"- **\1**: \2", ln, count=1)
        out.append(ln)
    return "\n".join(out).strip()


# ---------------------------------------------------------------- runner


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


class Extractor:
    def __init__(self, doc_path: str | Path, cfg: VLMConfig | None = None):
        self.doc_path = str(doc_path)
        self.cfg = cfg or VLMConfig()
        self.cfg.cache_dir = Path(self.cfg.cache_dir)
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key, timeout=self.cfg.timeout)
        self._local = threading.local()  # PyMuPDF documents are not thread-safe

    @property
    def doc(self) -> pymupdf.Document:
        if not hasattr(self._local, "doc"):
            self._local.doc = pymupdf.open(self.doc_path)
        return self._local.doc

    def cache_path(self, section: Section, idx: int) -> Path:
        r = section.regions[idx]
        return self.cfg.cache_dir / f"{section.order:03d}_{_slug(section.section_id)}_p{r.page:03d}_{idx}.json"

    def load(self, section: Section, idx: int) -> dict | None:
        p = self.cache_path(section, idx)
        return json.loads(p.read_text()) if p.exists() else None

    def call(self, messages: list[dict]) -> tuple[str, dict]:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        ch = resp.choices[0]
        u = resp.usage
        return ch.message.content or "", {
            "seconds": round(time.time() - t0, 2),
            "finish_reason": ch.finish_reason,
            "prompt_tokens": u.prompt_tokens if u else None,
            "completion_tokens": u.completion_tokens if u else None,
        }

    def extract_region(self, section: Section, idx: int, prev_tail: str | None) -> dict:
        region: Region = section.regions[idx]
        source = region_text(self.doc, region)
        if idx == 0:
            source = strip_heading(source, section.title)
        rec = {
            "section_id": section.section_id,
            "order": section.order,
            "region_idx": idx,
            "page": region.page,
            "region": [region.y0, region.y1],
            "prompt_version": PROMPT_VERSION,
            "model": self.cfg.model,
            "source": source,
            "attempts": [],
        }
        has_fig = region_has_figure(self.doc, region)
        if not source.strip():
            rec.update(status="figure_only" if has_fig else "empty", markdown="[Figure omitted]" if has_fig else "", metrics=None)
            return rec

        png = region_png(self.doc, region, dpi=self.cfg.dpi)
        feedback = None
        for attempt in range(2):
            raw, info = self.call(build_messages(png, source, section, idx, prev_tail, feedback))
            md = clean_output(raw)
            m = fidelity(source, md)
            m["issues"] = markdown_issues(md)
            rec["attempts"].append({"markdown": md, "metrics": m, **info})
            verdict = judge(m, md, self.cfg, info["finish_reason"])
            if verdict:
                status = "ok_fig" if verdict == "ok_fig" else ("ok" if attempt == 0 else "retry_ok")
                rec.update(status=status, markdown=md, metrics=m)
                return rec
            feedback = (
                "Your previous answer did not match the SOURCE TEXT closely enough. "
                + (f"Words/numbers that were missing: {', '.join(m['missing'][:20])}. " if m["missing"] else "")
                + (f"Words that are not in the source: {', '.join(m['extra'][:20])}. " if m["extra"] else "")
                + (f"Formatting problems: {'; '.join(m['issues'][:10])}. " if m["issues"] else "")
                + "Reproduce every word of the SOURCE TEXT exactly once, adding only Markdown structure."
            )
        best = max(rec["attempts"], key=lambda a: (a["metrics"]["recall"] + a["metrics"]["precision"]))
        fb = native_fallback(source)
        rec.update(status="fallback", markdown=fb, metrics=fidelity(source, fb), vlm_best=best["markdown"])
        return rec

    def extract_section(self, section: Section, redo: bool = False) -> list[dict]:
        out, prev_tail = [], None
        for idx in range(len(section.regions)):
            rec = None if redo else self.load(section, idx)
            if rec is None or rec.get("prompt_version") != PROMPT_VERSION:
                rec = self.extract_region(section, idx, prev_tail)
                self.cache_path(section, idx).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            out.append(rec)
            if rec["markdown"].strip():
                prev_tail = "\n".join(rec["markdown"].splitlines()[-self.cfg.tail_lines :])
        return out

    def run(self, sections: list[Section], redo: set[str] | None = None, progress=None) -> dict[str, list[dict]]:
        """Extract many sections, ``cfg.workers`` at a time. ``redo`` = section
        ids whose cache is ignored. ``progress`` = a tqdm-like bar or None."""
        redo = redo or set()
        results: dict[str, list[dict]] = {}
        with ThreadPoolExecutor(self.cfg.workers) as pool:
            futs = {pool.submit(self.extract_section, s, s.section_id in redo): s for s in sections}
            for f in as_completed(futs):
                s = futs[f]
                try:
                    results[s.section_id] = f.result()
                except Exception as e:  # keep going; the section is retried on the next run
                    results[s.section_id] = [{"section_id": s.section_id, "status": "error", "error": repr(e), "markdown": ""}]
                if progress is not None:
                    progress.update(1)
        return results


def rescore_cache(cache_dir: str | Path, cfg: VLMConfig | None = None) -> list[str]:
    """Re-score cached attempts with the current scoring code (no model calls).
    A fallback region whose best attempt now passes is upgraded to that
    attempt. Returns the upgraded cache file names."""
    cfg = cfg or VLMConfig()
    upgraded = []
    for p in sorted(Path(cache_dir).glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("status") != "fallback":
            continue
        for i, a in enumerate(rec["attempts"]):
            m = fidelity(rec["source"], a["markdown"])
            m["issues"] = markdown_issues(a["markdown"])
            a["metrics"] = m
            verdict = judge(m, a["markdown"], cfg, a.get("finish_reason"))
            if verdict:
                status = "ok_fig" if verdict == "ok_fig" else ("ok" if i == 0 else "retry_ok")
                rec.update(status=status, markdown=a["markdown"], metrics=m, rescored=True)
                rec.pop("vlm_best", None)
                p.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
                upgraded.append(p.name)
                break
    return upgraded


def load_cache(cache_dir: str | Path, sections: list[Section]) -> dict[str, list[dict]]:
    """Read cached region records for ``sections`` (missing regions -> None)."""
    cache_dir = Path(cache_dir)
    out = {}
    for s in sections:
        recs = []
        for idx, r in enumerate(s.regions):
            p = cache_dir / f"{s.order:03d}_{_slug(s.section_id)}_p{r.page:03d}_{idx}.json"
            recs.append(json.loads(p.read_text()) if p.exists() else None)
        out[s.section_id] = recs
    return out
