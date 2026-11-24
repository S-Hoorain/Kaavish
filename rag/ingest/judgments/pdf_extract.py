"""Layout-aware PDF text extraction for judgments (PyMuPDF), using font size and position.

* Scanned-PDF check first (pdffonts when installed, else PyMuPDF fonts/text per page).
* Running headers / page numbers: lines in the top or bottom band that match page-number
  patterns, are set smaller than body text, or repeat across pages.
* Footnotes: small-type lines below the last body-size line of a page.
* Footnote reference markers (superscript spans) are removed from body text and recorded.
* Body lines are grouped into blocks (paragraph candidates) by blank lines, vertical gaps,
  first-line indentation and leading paragraph numbers; blocks continue across page breaks.
"""
from __future__ import annotations

import re
import shutil
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF


class ScannedPDFError(RuntimeError):
    pass


PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$"),
    re.compile(r"^\s*P\s*a\s*g\s*e\s*\|?\s*\d+\s*$", re.I),
    re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*\[\s*\d{1,3}\s*\]\s*$"),
    re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$"),
]


def is_page_number(text: str) -> bool:
    return any(p.match(text) for p in PAGE_NUM_PATTERNS)


@dataclass
class Line:
    page: int                 # 1-based
    x0: float
    y0: float
    x1: float
    y1: float
    text: str                 # superscript markers removed
    size: float               # dominant font size (by characters)
    bold: bool
    markers: list[str] = field(default_factory=list)   # footnote markers removed from this line
    kind: str = "body"        # body | blank | running_header | page_number | footnote | separator
    full_text: str = ""       # text with markers kept (used for footnote lines)
    lead_marker: bool = False # line starts with a small numeric marker followed by normal text

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


@dataclass
class Footnote:
    number: str
    text: str
    page: int


@dataclass
class Block:
    """A paragraph candidate: consecutive body lines (possibly spanning pages)."""
    lines: list[Line]

    @property
    def text(self) -> str:
        parts: list[str] = []
        for ln in self.lines:
            t = ln.text.strip()
            if not t:
                continue
            if parts and parts[-1].endswith("-") and t[:1].islower() and not parts[-1].endswith(" -"):
                parts[-1] = parts[-1] + t  # rejoin a hyphenated word break
            else:
                parts.append(t)
        return " ".join(parts)

    @property
    def page_start(self) -> int:
        return self.lines[0].page

    @property
    def x0(self) -> float:
        return self.lines[0].x0

    @property
    def markers(self) -> list[str]:
        return [m for ln in self.lines for m in ln.markers]

    @property
    def min_x(self) -> float:
        return min(ln.x0 for ln in self.lines if ln.text.strip()) if any(ln.text.strip() for ln in self.lines) else self.x0

    def is_indented(self, left_margin: float) -> bool:
        """True for block quotations: every line indented from the left margin."""
        n = sum(1 for ln in self.lines if ln.text.strip())
        return self.min_x > left_margin + 12 and (n > 1 or self.x0 > left_margin + 45)


@dataclass
class DocLayout:
    path: str
    n_pages: int
    body_size: float
    left_margin: float
    lines: list[Line]
    footnotes: list[Footnote]
    removed: list[Line]        # running headers, page numbers, separators
    scan_check: dict
    trailers: list[Line] = field(default_factory=list)   # clerk/stenographer initials below the body


# --------------------------------------------------------------------------------------------
def scan_check(pdf_path: Path) -> dict:
    """Detect scanned PDFs. Uses pdffonts when on PATH; always also checks text per page."""
    info: dict = {"tool": None, "fonts": None, "pages_without_text": [], "scanned": False}
    exe = shutil.which("pdffonts")
    if exe:
        try:
            out = subprocess.run([exe, str(pdf_path)], capture_output=True, text=True, timeout=60).stdout
            rows = [r for r in out.splitlines()[2:] if r.strip()]
            info.update(tool="pdffonts", fonts=len(rows))
        except Exception as e:  # pragma: no cover - environment specific
            info["pdffonts_error"] = str(e)
    doc = fitz.open(pdf_path)
    if info["tool"] is None:
        info["tool"] = "pymupdf"
        info["fonts"] = len({f[3] for p in doc for f in p.get_fonts()})
    for i, page in enumerate(doc, start=1):
        if not page.get_text("text").strip():
            info["pages_without_text"].append(i)
    info["scanned"] = (info["fonts"] == 0) or bool(info["pages_without_text"])
    return info


def _raw_lines(doc: fitz.Document) -> list[tuple[int, list[dict], tuple[float, float, float, float]]]:
    """All text lines as (page, spans, bbox), with lines on the same baseline merged."""
    out = []
    for pno, page in enumerate(doc, start=1):
        items = []
        for b in page.get_text("dict")["blocks"]:
            for ln in b.get("lines", []):
                spans = [s for s in ln["spans"]]
                if spans:
                    items.append((ln["bbox"], spans))
        items.sort(key=lambda it: (round((it[0][1] + it[0][3]) / 2), it[0][0]))
        merged: list[list] = []
        for bbox, spans in items:
            cy = (bbox[1] + bbox[3]) / 2
            if merged:
                pb = merged[-1][0]
                pcy = (pb[1] + pb[3]) / 2
                if abs(cy - pcy) <= 3.0:
                    merged[-1][1].extend(spans)
                    merged[-1][0] = (min(pb[0], bbox[0]), min(pb[1], bbox[1]), max(pb[2], bbox[2]), max(pb[3], bbox[3]))
                    continue
            merged.append([bbox, list(spans)])
        for bbox, spans in merged:
            spans.sort(key=lambda s: s["bbox"][0])
            out.append((pno, spans, bbox))
    return out


def _body_size(raw) -> float:
    c: Counter = Counter()
    for _, spans, _ in raw:
        for s in spans:
            if s["text"].strip():
                c[round(s["size"] * 2) / 2] += len(s["text"].strip())
    return c.most_common(1)[0][0] if c else 12.0


def _is_marker_span(s: dict, body: float) -> bool:
    t = s["text"].strip()
    if not t:
        return False
    small = s["size"] <= body * 0.75
    superscript = bool(s["flags"] & 1)
    return (superscript or small) and bool(re.fullmatch(r"[\d*†‡,]{1,4}", t)) and (superscript or s["size"] <= body * 0.7)


def _make_line(pno: int, spans: list[dict], bbox, body: float, keep_markers: bool) -> Line:
    text_parts: list[str] = []
    markers: list[str] = []
    sizes: Counter = Counter()
    bold_chars = 0
    last_x1 = None
    for s in spans:
        t = s["text"]
        if not keep_markers and _is_marker_span(s, body) and any(
                o is not s and o["text"].strip() and o["size"] > s["size"] + 1 for o in spans):
            markers.append(t.strip())
            last_x1 = s["bbox"][2]
            continue
        if last_x1 is not None and s["bbox"][0] - last_x1 > 1.5 and text_parts and \
                not text_parts[-1].endswith(" ") and not t.startswith(" "):
            text_parts.append(" ")
        text_parts.append(t)
        last_x1 = s["bbox"][2]
        if t.strip():
            sizes[round(s["size"] * 2) / 2] += len(t.strip())
            if s["flags"] & 16 or "bold" in s["font"].lower():
                bold_chars += len(t.strip())
    text = re.sub(r"[ \t ]+", " ", "".join(text_parts))
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    n_chars = sum(sizes.values())
    ln = Line(page=pno, x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], text=text.strip(),
              size=size, bold=n_chars > 0 and bold_chars / n_chars > 0.5, markers=markers)
    # text with markers kept, and whether the line starts with a small numeric marker (footnote start)
    nonblank = [s for s in spans if s["text"].strip()]
    ln.full_text = re.sub(r"\s+", " ", " ".join(s["text"].strip() for s in nonblank)).strip()
    ln.lead_marker = bool(nonblank) and len(nonblank) > 1 and \
        re.fullmatch(r"\d{1,3}\.?", nonblank[0]["text"].strip()) is not None and \
        nonblank[0]["size"] <= body * 0.75 and nonblank[1]["size"] > nonblank[0]["size"] + 1
    return ln


def extract_layout(pdf_path: Path) -> DocLayout:
    check = scan_check(pdf_path)
    if check["scanned"]:
        raise ScannedPDFError(f"{pdf_path.name}: looks scanned ({check}). OCR is not part of R0; stopping.")
    doc = fitz.open(pdf_path)
    raw = _raw_lines(doc)
    body = _body_size(raw)
    lines = [_make_line(p, spans, bbox, body, keep_markers=False) for p, spans, bbox in raw]
    heights = {p: doc[p - 1].rect.height for p in range(1, len(doc) + 1)}

    # --- running headers / page numbers ------------------------------------------------------
    top_texts: Counter = Counter()
    bottom_texts: Counter = Counter()
    for ln in lines:
        if not ln.is_blank and ln.y1 < heights[ln.page] * 0.09:
            top_texts[re.sub(r"\d+", "#", ln.text)] += 1
        if not ln.is_blank and ln.y0 > heights[ln.page] * 0.90:
            bottom_texts[re.sub(r"\d+", "#", ln.text)] += 1
    removed: list[Line] = []
    for ln in lines:
        if ln.is_blank:
            ln.kind = "blank"
            continue
        h = heights[ln.page]
        in_top = ln.y1 < h * 0.09
        in_bottom = ln.y0 > h * 0.93
        repeated = top_texts[re.sub(r"\d+", "#", ln.text)] >= 2
        if (in_top or in_bottom) and is_page_number(ln.text):
            ln.kind = "page_number"
        elif ln.y0 > h * 0.90 and bottom_texts[re.sub(r"\d+", "#", ln.text)] >= 2 and len(ln.text) < 60                 and not ln.lead_marker and not re.match(r"^\d{1,3}[.)]?\s", ln.full_text or ln.text):
            ln.kind = "running_footer"
        elif in_top and ln.page > 1 and not repeated and ln.size <= body * 0.85 \
                and re.match(r"^\d{1,3}[.)]?\s+\S", ln.text):
            ln.kind = "footnote_overflow"  # footnote carried over to the top of the next page
            continue
        elif in_top and ln.page > 1 and (repeated or (ln.size < body - 0.5 and len(ln.text) < 50)):
            ln.kind = "running_header"
        elif re.fullmatch(r"[_\-–—=.*…= ]{4,}", ln.text):
            ln.kind = "separator"
        if ln.kind != "body":
            removed.append(ln)

    # --- footnotes: small type below the last body-size line of each page --------------------
    footnotes: list[Footnote] = []
    trailers: list[Line] = []   # small unnumbered lines at the end (clerk / stenographer initials)
    for pno in range(1, len(doc) + 1):
        page_lines = [ln for ln in lines if ln.page == pno and ln.kind == "body"]
        overflow = [ln for ln in lines if ln.page == pno and ln.kind == "footnote_overflow"]
        big = [ln for ln in page_lines if ln.size >= body - 1.0]
        last_big_y = max((ln.y1 for ln in big), default=0.0)
        fn_lines = overflow + [ln for ln in page_lines if big and ln.y0 >= last_big_y - 1 and ln.size <= body * 0.85]
        # footnotes set in body-size type: a line in the lower part of the page that starts with a
        # small numeric marker, and every line after it on that page
        h = heights[pno]
        lead = next((ln for ln in page_lines if ln.lead_marker and ln.y0 > h * 0.55), None)
        if lead is not None:
            fn_lines += [ln for ln in page_lines if ln.y0 >= lead.y0 - 0.5 and ln not in fn_lines]
        if not fn_lines:
            continue
        page_has_fn = False
        for ln in sorted(fn_lines, key=lambda l: l.y0):
            was_overflow = ln.kind == "footnote_overflow"
            small_lead = ln.lead_marker
            full = ln.full_text or ln.text
            m = re.match(r"^(\d{1,3})(?:[.)](?!\d)\s*|\s+)(.*)$", full)   # "3. text", "1 text"; not "08.09.2025"
            expected = str(int(footnotes[-1].number) + 1) if footnotes and footnotes[-1].number.isdigit() else "1"
            if m and (small_lead or not footnotes or m.group(1) == expected):
                ln.kind = "footnote"
                footnotes.append(Footnote(number=m.group(1), text=m.group(2).strip(), page=pno))
                page_has_fn = True
            elif footnotes and (page_has_fn or was_overflow):  # continuation of the current footnote
                ln.kind = "footnote"
                footnotes[-1].text = (footnotes[-1].text + " " + full).strip()
            else:  # small unnumbered text below the body: signature trailer, not a footnote
                ln.kind = "trailer"
                trailers.append(ln)

    body_lines = [ln for ln in lines if ln.kind in ("body", "blank", "separator")]
    xs = Counter(round(ln.x0) for ln in body_lines if ln.kind == "body" and ln.size >= body - 1.0)
    left = min((x for x, c in xs.items() if c >= max(3, 0.05 * sum(xs.values()))), default=72.0)
    return DocLayout(path=str(pdf_path), n_pages=len(doc), body_size=body, left_margin=float(left),
                     lines=body_lines, footnotes=footnotes, removed=removed, scan_check=check, trailers=trailers)


# "2. Text", "02. Text", "4&5. Text", "(3) Text"; not dates such as "29.11.2012" or "08.01.2021"
PARA_NUM = re.compile(r"^\(?(\d{1,3}(?:\s*[&,\-]\s*\d{1,3})?)\s*[.)\]](?!\d)\s*(?=\S)")
_ENDS_SENTENCE = (".", ":", ";", "”", '"', "’")


def group_blocks(layout: DocLayout) -> list[Block]:
    """Group body lines into paragraph-candidate blocks."""
    lines = [ln for ln in layout.lines if ln.kind in ("body", "blank", "separator")]
    text_lines = [ln for ln in lines if ln.kind == "body"]
    heights = [ln.y1 - ln.y0 for ln in text_lines] or [12.0]
    med_h = statistics.median(heights)
    # line spacing differs between pages (and documents), so the gap threshold is per page
    page_gaps: dict[int, list[float]] = {}
    for a, b in zip(text_lines, text_lines[1:]):
        if a.page == b.page and b.y0 > a.y1:
            page_gaps.setdefault(a.page, []).append(b.y0 - a.y1)
    all_gaps = [g for gs in page_gaps.values() for g in gs]
    doc_med = statistics.median(all_gaps) if all_gaps else 4.0
    gap_break = {p: (statistics.median(gs) if len(gs) >= 5 else doc_med) + 0.45 * med_h
                 for p, gs in page_gaps.items()}
    indent = layout.left_margin + 12

    blocks: list[Block] = []
    cur: list[Line] = []
    prev: Line | None = None
    pending_blank = False
    for ln in lines:
        if ln.kind in ("blank", "separator"):
            pending_blank = True
            continue
        new = False
        ends = prev is not None and prev.text.rstrip().endswith(_ENDS_SENTENCE)
        if prev is None:
            new = True
        elif ln.page == prev.page:
            if pending_blank or (ln.y0 - prev.y1) > gap_break.get(ln.page, doc_med + 0.45 * med_h):
                new = True
            elif ln.x0 > indent and prev.x0 <= indent + 2 and PARA_NUM.match(ln.text) is None and ends:
                new = True  # first-line indent after a finished sentence
        else:  # page break: continue unless the new line clearly starts a paragraph
            if PARA_NUM.match(ln.text) or (ln.x0 > indent and ends):
                new = True
        if not new and PARA_NUM.match(ln.text) and ln.x0 <= indent and ends:
            new = True
        if new and cur:
            blocks.append(Block(cur))
            cur = []
        cur.append(ln)
        prev = ln
        pending_blank = False
    if cur:
        blocks.append(Block(cur))
    return blocks
