"""Split CS-E Amendment 9 into sections using the PDF outline (bookmarks).

The outline gives every heading's exact page and y position. A section runs
from its heading to the next outline heading, whatever that heading's level, so
it can never take in text from the section before or after it. Boundaries are
snapped to the top of the heading's text line so no line is cut in half.

The input PDF is already cropped (header and footer outside the page box), so
every page operation clips to ``page.rect``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

# Every spelling seen in the document: "[Amdt. No.: E/1]", "[Amdt No: E/9]", ...
AMDT_RE = re.compile(r"\[\s*Amdt\.?\s*No\.?\s*:?[^\]]*\]", re.IGNORECASE)

# Outline entries before this page are the Table of Contents and the Preamble.
FIRST_BODY_PAGE = 16

_ID_RE = re.compile(
    r"""^(?P<prefix>CS-E|AMC\s+to\s+CS-E|AMC\s+E)\s+
        (?P<num>\d+[A-Z]?)
        (?P<refs>(?:\([^)]+\))*(?:\s+and\s+(?:\([^)]+\))+)?)
        \s*(?P<heading>.*)$""",
    re.VERBOSE,
)
_SUBPART_RE = re.compile(r"^(SUBPART\s+[A-Z])\s*[–—-]\s*(.*)$")


@dataclass
class Region:
    """A vertical slice of one page. ``page`` is 1-based."""

    page: int
    y0: float
    y1: float

    def rect(self, doc: pymupdf.Document) -> pymupdf.Rect:
        r = doc[self.page - 1].rect
        return pymupdf.Rect(r.x0, self.y0, r.x1, self.y1)


@dataclass
class Section:
    section_id: str  # "CS-E 20", "AMC E 20(f)", "AMC General", "CS-E 790 Appendix A"
    kind: str  # cs | amc | amc_general | appendix
    title: str  # full outline title, whitespace-normalised
    heading: str  # title without the id
    group_id: str  # parent CS-E ("CS-E 20"), or the subpart for AMC General
    subpart: str  # "SUBPART A"
    subpart_title: str  # "GENERAL"
    level: int  # outline level (2 or 3)
    order: int  # position in the document
    regions: list[Region] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return self.regions[0].page

    @property
    def page_end(self) -> int:
        return self.regions[-1].page

    def to_dict(self) -> dict:
        d = asdict(self)
        d["page_start"], d["page_end"] = self.page_start, self.page_end
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        d = {k: v for k, v in d.items() if k not in ("page_start", "page_end")}
        d["regions"] = [Region(**r) for r in d["regions"]]
        return cls(**d)


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_title(title: str) -> tuple[str, str, str]:
    """Return (section_id, kind, heading) for a CS-E/AMC outline title."""
    t = norm_ws(title)
    if t == "AMC General":
        return "AMC General", "amc_general", "General"
    m = _ID_RE.match(t)
    if not m:
        raise ValueError(f"Unrecognised outline title: {title!r}")
    prefix = re.sub(r"\s+", " ", m["prefix"])
    sid = f"{prefix} {m['num']}{norm_ws(m['refs'])}".replace(" and(", " and (")
    kind = "cs" if prefix == "CS-E" else "amc"
    return sid, kind, m["heading"].strip()


def _lines(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """Text lines inside the visible page as (y0, y1, text), top to bottom."""
    out = []
    for b in page.get_text("dict", clip=page.rect)["blocks"]:
        for ln in b.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if txt:
                out.append((ln["bbox"][1], ln["bbox"][3], txt))
    return sorted(out)


def _snap(lines: list[tuple[float, float, str]], y: float, title: str) -> tuple[float, str | None]:
    """Snap an outline y to the top of the heading line it points at.

    The outline point sits on, or just above, the heading's first line. Returns
    (boundary_y, heading_line_text), or (y, None) if no line near y starts like
    the title, which the notebook reports as a warning.
    """
    key = norm_ws(title).lower()[:10]
    near = [(y0, txt) for y0, y1, txt in lines if y - 15 <= y0 <= y + 25 or y0 <= y <= y1]
    for y0, txt in near:
        if norm_ws(txt).lower().startswith(key):
            return y0 - 0.5, txt
    return y, None


@dataclass
class Boundary:
    level: int
    title: str
    page: int
    y: float  # snapped boundary (top of heading line)
    outline_y: float
    matched: str | None


def outline_boundaries(doc: pymupdf.Document, first_page: int = FIRST_BODY_PAGE) -> list[Boundary]:
    """All outline headings from ``first_page`` onward, with snapped y positions."""
    out, cache = [], {}
    for level, title, page, dest in doc.get_toc(simple=False):
        if page < first_page:
            continue
        to = dest.get("to")
        oy = float(to.y) if to is not None else 0.0
        if page not in cache:
            cache[page] = _lines(doc[page - 1])
        y, matched = _snap(cache[page], oy, title)
        out.append(Boundary(level, norm_ws(title), page, max(y, 0.0), oy, matched))
    return out


def _regions_between(doc: pymupdf.Document, a: Boundary, b: Boundary | None) -> list[Region]:
    end_page = b.page if b else doc.page_count
    regions = []
    for p in range(a.page, end_page + 1):
        h = doc[p - 1].rect.y1
        y0 = a.y if p == a.page else 0.0
        y1 = b.y if (b and p == b.page) else h
        if y1 - y0 > 1.0:
            regions.append(Region(p, round(y0, 2), round(y1, 2)))
    return regions


def build_sections(doc: pymupdf.Document, first_page: int = FIRST_BODY_PAGE) -> list[Section]:
    bounds = outline_boundaries(doc, first_page)
    sections: list[Section] = []
    subpart = subpart_title = ""
    parent_cs = ""
    seen: dict[str, int] = {}
    for i, bd in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else None
        if bd.level == 1:
            m = _SUBPART_RE.match(bd.title)
            subpart, subpart_title = (m[1], m[2]) if m else (bd.title, "")
            parent_cs = ""
            continue
        if bd.title.startswith("Appendix"):
            kind = "appendix"
            sid = norm_ws(f"{parent_cs} " + re.match(r"Appendix\s+\S+", bd.title)[0])
            heading = norm_ws(re.sub(r"^Appendix\s+\S+", "", bd.title))
        else:
            sid, kind, heading = parse_title(bd.title)
        if kind == "cs":
            parent_cs = sid
        group = sid if kind == "cs" else (parent_cs or subpart)
        if kind == "amc_general":
            sid, group = f"AMC General ({subpart})", subpart
        if sid in seen:  # keep ids unique if the outline ever repeats one
            seen[sid] += 1
            sid = f"{sid} #{seen[sid]}"
        else:
            seen[sid] = 1
        sections.append(
            Section(
                section_id=sid,
                kind=kind,
                title=bd.title,
                heading=heading,
                group_id=group,
                subpart=subpart,
                subpart_title=subpart_title,
                level=bd.level,
                order=len(sections),
                regions=_regions_between(doc, bd, nxt),
            )
        )
    return sections


# ---------------------------------------------------------------- page access


def fold_math(s: str) -> str:
    """Fold Unicode math-italic/bold letters (U+1D400-1D7FF, used by Word's
    equation editor: 𝑃𝑏𝑙𝑒𝑒𝑑 -> Pbleed) and circled letters (Ⓐ -> A) to plain
    characters. Deliberately not full NFKC, which would also turn "2½" into
    "21⁄2"."""
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x1D400 <= cp <= 0x1D7FF or 0x24B6 <= cp <= 0x24E9:
            ch = unicodedata.normalize("NFKC", ch)
        out.append(ch)
    return "".join(out)


def region_text(doc: pymupdf.Document, region: Region, strip_amdt: bool = True) -> str:
    """Native PDF text of a region, reading order, math letters folded, with
    [Amdt ...] notes removed."""
    page = doc[region.page - 1]
    txt = fold_math(page.get_text("text", clip=region.rect(doc), sort=True))
    if strip_amdt:
        txt = AMDT_RE.sub("", txt)
    lines = [ln.rstrip() for ln in txt.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def region_png(doc: pymupdf.Document, region: Region, dpi: int = 150) -> bytes:
    page = doc[region.page - 1]
    pix = page.get_pixmap(clip=region.rect(doc), dpi=dpi, alpha=False)
    return pix.tobytes("png")


def region_has_figure(doc: pymupdf.Document, region: Region, min_area: float = 2000.0) -> bool:
    """True if a raster image of real size lies inside the region (ignores the off-page logo)."""
    page, rect = doc[region.page - 1], region.rect(doc)
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & rect
        if not r.is_empty and r.get_area() >= min_area:
            return True
    return False


# ---------------------------------------------------------------- checks / io


def coverage_report(doc: pymupdf.Document, sections: list[Section], first_page: int = FIRST_BODY_PAGE) -> dict:
    """Assign every text line (by its vertical midpoint) to regions.

    Returns lines that fall in no section (``orphans``; expected only for the
    subpart heading lines) and lines claimed by more than one (``overlaps``,
    must be empty).
    """
    by_page: dict[int, list[tuple[str, Region]]] = {}
    for s in sections:
        for r in s.regions:
            by_page.setdefault(r.page, []).append((s.section_id, r))
    orphans, overlaps = [], []
    for p in range(first_page, doc.page_count + 1):
        for y0, y1, txt in _lines(doc[p - 1]):
            mid = (y0 + y1) / 2
            owners = [sid for sid, r in by_page.get(p, []) if r.y0 <= mid < r.y1]
            if not owners:
                orphans.append((p, round(y0, 1), txt))
            elif len(owners) > 1:
                overlaps.append((p, round(y0, 1), txt, owners))
    return {"orphans": orphans, "overlaps": overlaps}


def save_sections(sections: list[Section], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([s.to_dict() for s in sections], indent=1, ensure_ascii=False))


def load_sections(path: str | Path) -> list[Section]:
    return [Section.from_dict(d) for d in json.loads(Path(path).read_text())]
