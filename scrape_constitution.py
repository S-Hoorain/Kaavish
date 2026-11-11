#!/usr/bin/env python3
"""
Constitution of Pakistan (Law Ministry 14th ed., "as amended upto the 27th Amendment")
PDF -> structured JSONL, using the PDF's own layout (fonts + x/y positions).

Install:  pip install pymupdf
Run:      python scrape_constitution.py --pdf constitution.pdf --source-url "<where you downloaded it>"

Layout facts this parser relies on (checked against the real file):
  * Article numbers ("7.", "2A.") are BOLD at the left of the text column.
  * Article titles are BOLD 8pt "side headings" in a separate right-hand column (x > 440).
  * Footnotes sit under a dashed/underscore rule at the bottom of each page; their numbers
    restart on every page, and the inline markers are tiny (<7pt) digits.
  * The running header ("CONSTITUTION OF PAKISTAN" + printed page number) is at y < 80.
  * The contents pages (before the body) list every article number + title, so they are
    parsed too and used as ground truth for validation.

Outputs (in --out, default ./data):
  meta.json, raw_pages/page_NNN.txt, articles.jsonl, structure.json,
  schedules_and_annex.txt, toc.json, validation_report.json, spot_check/*.png
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz

# ------------------------------------------------------------------ CONFIG
HEADER_Y = 80          # anything above this y is the running header
SIDE_X = 440           # side headings: right of this x on odd pages ...
SIDE_L = 150           # ... and left of this x on even pages
SMALL = 7.0            # spans smaller than this (pt) are footnote markers
LAST_ARTICLE = 280
DOC_ID = "CONST-1973"

ART_NUM = re.compile(r"^\s*(\d{1,3}[A-Z]{0,3})\.(?:\s|$)")
PART_RE = re.compile(r"^PART\s+([IVXL]+)$")
CHAPTER_RE = re.compile(r"^C\s?HAPTER\s+(\d+[A-Z]?)\.?\s*[\u2013\u2014\-]+\s*(.+)$", re.I)
CLAUSE_RE = re.compile(r"^\[?\s*\((\d{1,3}[A-Z]{0,2})\)\s*(.*)$", re.S)
RULE_RE = re.compile(r"^[_\u2013\u2014\-\s]{15,}$")
MARK_ONLY = re.compile(r"^\s*(\d{1,2})\s*$")
MARK_LEAD = re.compile(r"^\s*(\d{1,2})(?=[\[\(\u2018\u201c])")
LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}


# ------------------------------------------------------------ low-level helpers
def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    for k, v in LIGATURES.items():
        t = t.replace(k, v)
    t = t.replace("\u00a0", " ").replace("\u00ad", "")
    t = t.replace("\u2018\u2018", "\u201c").replace("\u2019\u2019", "\u201d")
    t = re.sub(r"\u2018\u2018|\u2019\u2019", '"', t)
    return t


def page_lines(page):
    """All text lines on a page with span-level font info, in reading order."""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = []
            for s in l["spans"]:
                if not s["text"]:
                    continue
                spans.append({
                    "t": s["text"], "size": s["size"], "x0": s["bbox"][0],
                    "bold": bool(s["flags"] & 16) or "Bold" in s["font"],
                    "ital": bool(s["flags"] & 2) or "Italic" in s["font"],
                })
            if spans and "".join(s["t"] for s in spans).strip():
                out.append({"x0": l["bbox"][0], "y0": l["bbox"][1], "spans": spans})
    out.sort(key=lambda L: (round(L["y0"]), L["x0"]))
    return out


def split_markers(spans):
    """Separate tiny footnote-marker digits from real text."""
    parts, marks = [], []
    for s in spans:
        t = s["t"]
        if s["size"] < SMALL:
            m = MARK_ONLY.match(t)
            if m:
                marks.append(int(m.group(1)))
                continue
            m = MARK_LEAD.match(t)
            if m:
                marks.append(int(m.group(1)))
                t = t[m.end():]
        parts.append((t, s))
    return parts, marks


def line_text(parts):
    return norm("".join(t for t, _ in parts))


def is_rule(text):
    return bool(RULE_RE.match(text.strip())) and len(text.strip()) >= 15



def _txt(L):
    return "".join(s["t"] for s in L["spans"])


def _size(L):
    return max((s["size"] for s in L["spans"] if s["t"].strip()), default=0)


def _is_side(L):
    return (L["x0"] >= SIDE_X or L["x0"] < SIDE_L) and _size(L) < 9.5 and \
        all(s["bold"] for s in L["spans"] if s["t"].strip())


def find_footnote_start(lines):
    """y where the footnote block begins, or None.
    Footnotes are set in <= 8.5pt type at the foot of the page, usually under a rule
    (but some pages have no rule, and some body tables contain rule-like lines)."""
    main = [L for L in lines if L["y0"] >= HEADER_Y and not _is_side(L)]

    def small_tail(i):  # every real line from i on is footnote-sized (or a rule)
        for L in main[i + 1:]:
            t = _txt(L).strip()
            if len(t) <= 2 or is_rule(t):
                continue
            if _size(L) > 8.5:
                return False
        return True

    for i, L in enumerate(main):                     # 1) an explicit rule
        if L["y0"] > 130 and is_rule(_txt(L)) and small_tail(i):
            return L["y0"]
    for i, L in enumerate(main):                     # 2) no rule: first line of the small-type tail
        t = _txt(L).strip()
        if L["y0"] > 250 and _size(L) <= 8.5 and len(t) > 2 and not re.match(r"^\d{1,2}\[", t) \
                and small_tail(i - 1 if i else 0) and (i == 0 or True):
            if all(_size(M) <= 8.5 or len(_txt(M).strip()) <= 2 for M in main[i:]):
                return L["y0"]
    return None

# ----------------------------------------------------------------- page model
def read_page(page):
    """Split a page into header, body lines, side headings and footnote lines."""
    lines = page_lines(page)
    sep_y = find_footnote_start(lines)
    header_text, body, side, foot = "", [], [], []
    for L in lines:
        text_all = "".join(s["t"] for s in L["spans"])
        if L["y0"] < HEADER_Y:
            header_text += " " + text_all
            continue
        if sep_y is not None and L["y0"] >= sep_y:
            if not is_rule(text_all):
                foot.append(norm(text_all).strip())
            continue
        parts, marks = split_markers(L["spans"])
        txt = line_text(parts)
        if not txt.strip():
            if marks and SIDE_L <= L["x0"] < SIDE_X:
                body.append({"x0": L["x0"], "y0": L["y0"], "parts": [], "marks": marks, "raw": "", "ws": 0,
                             "text": "", "all_bold": False, "all_ital": False})
            continue
        item = {"x0": L["x0"], "y0": L["y0"], "parts": parts, "marks": marks,
                "raw": txt, "ws": len(txt) - len(txt.lstrip()), "text": txt.strip()}
        item["all_bold"] = all(s["bold"] for t, s in parts if t.strip())
        item["all_ital"] = all(s["ital"] for t, s in parts if t.strip())
        small = max(s["size"] for t, s in parts if t.strip()) < 9.5
        if item["all_bold"] and small and (L["x0"] >= SIDE_X or L["x0"] < SIDE_L):
            side.append(item)
        elif SIDE_L <= L["x0"] < SIDE_X:
            body.append(item)
    m = re.search(r"(\d+|[ivxl]+)\s*$", header_text.strip(), re.I)
    return {"body": body, "side": side, "foot": foot, "printed": m.group(1) if m else None}


# ---------------------------------------------------------------- contents (TOC)
def parse_toc(doc, first_body_idx):
    """Return [(article_no, title)] from the CONTENTS pages (ground truth for validation)."""
    entries = []
    for pi in range(0, first_body_idx):
        text = doc[pi].get_text()
        if "ARTICLES" not in text or "PAGES" not in text:
            continue
        for L in page_lines(doc[pi]):
            if L["y0"] < HEADER_Y or L["y0"] > 660:
                continue
            raw = norm("".join(s["t"] for s in L["spans"]))
            m = re.match(r"^\s*(\d{1,3}[A-Z]{0,3})\.\s{2,}(.*)$", raw)
            if m:
                entries.append([m.group(1), m.group(2).strip()])
            elif entries and 15 <= (len(raw) - len(raw.lstrip())) <= 30 and not any(s["ital"] or s["bold"] for s in L["spans"] if s["t"].strip()) \
                    and not re.search(r"PART|C\s?HAPTER|\d+[\u2013\-]\d+\s*$", raw):
                entries[-1][1] = (entries[-1][1] + " " + raw.strip()).strip()
    return [(n, t) for n, t in entries]


# ------------------------------------------------------------------ body parser
def find_body_range(doc):
    start = end = None
    for pi in range(10, len(doc)):
        lines = page_lines(doc[pi])
        top = [norm("".join(s["t"] for s in L["spans"])).strip() for L in lines if L["y0"] < 130 and L["y0"] >= HEADER_Y]
        if start is None and any(PART_RE.match(t) for t in top) and any(t.startswith("Introductory") for t in top):
            start = pi
        if start is not None and pi > start and any(re.match(r"^\d*\[?ANNEX", t) for t in top):
            end = pi
            break
    return start, end


def parse_body(doc, start, end, toc_nums):
    articles, structure = [], []
    cur = None
    part = chapter = None
    pending_part = False
    seen = set()
    last_foot = None  # (article, n) of the previous footnote, for continuation across pages

    for pi in range(start, end):
        pg = read_page(doc[pi])
        pno = pi + 1
        arts_here = []  # (y0, article)
        marks_here = {}  # marker number -> article

        for L in pg["body"]:
            parts, text = L["parts"], L["text"]
            if not text:                      # a line holding only footnote markers
                for n in L["marks"]:
                    if cur is not None:
                        marks_here[n] = cur
                continue
            # --- Part / Chapter headings -------------------------------------
            pm = PART_RE.match(text) if L["all_bold"] else None
            if pm:
                part = {"part": pm.group(1), "part_title": ""}
                chapter = None
                pending_part = True
                structure.append({"type": "part", **part, "pdf_page": pno})
                cur = None if cur is None else cur
                continue
            if pending_part and L["all_bold"] and L["x0"] > 180 and not ART_NUM.match(text):
                part["part_title"] = (part["part_title"] + " " + re.sub(r"\d{1,2}(?=\[)", "", text)).strip()
                structure[-1]["part_title"] = part["part_title"]
                continue
            cm = CHAPTER_RE.match(re.sub(r"\d{1,2}(?=\[)", "", text))
            if cm and L["x0"] > 150 and not L["all_bold"]:
                chapter = {"chapter": cm.group(1), "chapter_title": cm.group(2).strip().strip("[]")}
                structure.append({"type": "chapter", **chapter, "part": part["part"] if part else None, "pdf_page": pno})
                pending_part = False
                continue
            # --- article start ---------------------------------------------
            started = False
            prefix = ""
            for i, (t, s) in enumerate(parts):
                if not t.strip() or t.strip() == "[":
                    prefix += "[" if t.strip() == "[" else ""
                    continue
                m = re.match(r"^\s*(\[)?(\d{1,3}[A-Z]{0,3})(\.)?(?=\s|$)", t) if s["bold"] else None
                if m:
                    num = m.group(2)
                    if m.group(1):
                        prefix += "["
                    nxt = parts[i + 1][0] if i + 1 < len(parts) else ""
                    if m.group(3):                       # "49." in one span
                        rest_first, tail = t[m.end():], parts[i + 1:]
                        ok = True
                    elif nxt.lstrip().startswith("."):   # "49" bold, "." in the next span
                        rest_first, tail = nxt.lstrip()[1:], parts[i + 2:]
                        ok = True
                    else:
                        ok = False
                    if ok and num in toc_nums and num not in seen:
                        rest = prefix + rest_first + "".join(x for x, _ in tail)
                        cur = {"section": num, "article": num, "title": "", "title_lines": [],
                               "part": part["part"] if part else None,
                               "part_title": part["part_title"] if part else None,
                               "chapter": chapter["chapter"] if chapter else None,
                               "chapter_title": chapter["chapter_title"] if chapter else None,
                               "pdf_pages": [pno, pno], "_lines": [], "notes": [],
                               "_first": True}
                        cur["_lines"].append((norm(rest), 0, True))
                        articles.append(cur)
                        arts_here.append((L["y0"], cur))
                        seen.add(num)
                        started = True
                        pending_part = False
                break
            if started:
                for n in L["marks"]:
                    marks_here[n] = cur
                continue
            # --- italic centred sub-headings (e.g. "Ordinances") ----------------
            if L["all_ital"] and L["x0"] > 190 and len(text) < 70 and L["ws"] == 0 and cur is not None:
                cur.setdefault("subheadings", []).append(text)
                continue
            # --- ordinary text line ------------------------------------------
            pending_part = False
            if cur is not None:
                cur["_lines"].append((L["raw"], L["ws"], False))
                cur["pdf_pages"][1] = pno
                for n in L["marks"]:
                    marks_here[n] = cur

        # --- side headings -> articles on this page --------------------------
        for S in pg["side"]:
            target = None
            for y, a in arts_here:
                if y <= S["y0"] + 3:
                    target = a
            if target is None and cur is not None and not arts_here:
                target = cur
            if target is None and arts_here:
                target = arts_here[0][1]
            if target is not None:
                target["title_lines"].append((S["y0"], S["text"]))

        # --- footnotes ----------------------------------------------------------
        expected = 1
        for ln in pg["foot"]:
            m = re.match(r"^(\d{1,2})\s*(\S.*)$", ln)
            if m and int(m.group(1)) == expected:
                n = int(m.group(1))
                art = marks_here.get(n) or (arts_here[-1][1] if arts_here else cur)
                note = {"marker": n, "text": m.group(2).strip(), "pdf_page": pno}
                if art is not None:
                    if n not in marks_here:
                        note["marker_not_located"] = True
                    art["notes"].append(note)
                last_foot = note
                expected += 1
            elif last_foot is not None:
                last_foot["text"] += " " + ln.strip()

    return articles, structure


# ---------------------------------------------------------------- post-process
def build_vocab(articles):
    c = Counter()
    for a in articles:
        for raw, _, _ in a["_lines"]:
            for w in re.findall(r"[A-Za-z]+", raw):
                c[w.lower()] += 1
    return c


def build_paragraphs(lines):
    paras = []
    for raw, ws, first in lines:
        t = raw.strip()
        if not t:
            continue
        if not paras or first or ws >= 5:
            paras.append(t)
        elif re.search(r"[A-Za-z]-$", paras[-1]) and re.match(r"[A-Za-z]", t):
            paras[-1] += "\x01" + t
        else:
            paras[-1] += " " + t
    return paras


def dehyphenate(text, vocab):
    def repl(m):
        left, right = m.group(1), m.group(2)
        joined = (left + right).lower()
        if right[0].islower() and vocab.get(joined, 0) > 0:
            return left + right            # "commenc-" + "ing" -> "commencing"
        return left + "-" + right          # "non-" + "Muslims" -> "non-Muslims"
    return re.sub(r"([A-Za-z]+)-\x01([A-Za-z]+)", repl, text)


def finalize(articles, toc_titles):
    vocab = build_vocab(articles)
    for a in articles:
        paras = [dehyphenate(p, vocab) for p in build_paragraphs(a.pop("_lines"))]
        paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
        a["title_side"] = re.sub(r"\s+", " ", " ".join(t for _, t in sorted(a.pop("title_lines")))).strip()
        a["title"] = toc_titles.get(a["article"], a["title_side"])
        if re.fullmatch(r"\W*omitted.*", a["title"], re.I) or re.fullmatch(r"\W*cease.*", a["title"], re.I):
            a["title"] = ""
        a.pop("_first", None)
        clauses, current = [], None
        for p in paras:
            m = CLAUSE_RE.match(p)
            if m:
                current = {"id": m.group(1), "text": m.group(2).strip()}
                clauses.append(current)
            elif current is not None:
                current["text"] += "\n" + p
        a["text"] = "\n".join(paras)
        a["text_plain"] = re.sub(r"\s{2,}", " ", re.sub(r"[\[\]]", "", a["text"]))
        a["clauses"] = clauses
        head = re.match(r"^\[([^\]]+)\]\s*(Omitted|Cease)", a["text"], re.I)
        toc_t = toc_titles.get(a["article"], "")
        if head or re.match(r"^\W*(omitted|cease)", toc_t, re.I):
            a["status"] = "omitted" if re.search(r"omit", (head.group(2) if head else toc_t), re.I) else "ceased"
            if head and not a["title"]:
                a["title"] = head.group(1).strip().rstrip(".") + "."
        else:
            a["status"] = "in force"
        a["doc_id"], a["doc_type"] = DOC_ID, "constitution"
    return articles


# --------------------------------------------------------------------- checks
def squash(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def validate(articles, toc, doc, body_range, spot_n, outdir):
    rep = {}
    got = {a["article"]: a for a in articles}
    toc_nums = [n for n, _ in toc]
    rep["toc_entries"] = len(toc)
    rep["articles_parsed"] = len(articles)
    rep["in_toc_not_parsed"] = [n for n in toc_nums if n not in got or got[n].get("remark")]
    rep["parsed_not_in_toc"] = [n for n in got if n not in set(toc_nums)]
    order = [a["article"] for a in articles]
    exp = [n for n in toc_nums if n in got]
    rep["order_mismatches"] = [(x, y) for x, y in zip(order, exp) if x != y][:10]
    bad = []
    for n, t in toc:
        if n in got and t and got[n]["title_side"]:
            r = SequenceMatcher(None, squash(t), squash(got[n]["title_side"])).ratio()
            if r < 0.85:
                bad.append({"article": n, "toc": t, "parsed": got[n]["title_side"], "similarity": round(r, 2)})
    rep["title_mismatches_vs_toc"] = bad
    rep["missing_numbers_1_to_280"] = [i for i in range(1, LAST_ARTICLE + 1) if str(i) not in got]
    rep["empty_titles"] = [a["article"] for a in articles if not a["title"] and a["status"] == "in force"]
    rep["empty_text"] = [a["article"] for a in articles if len(a["text_plain"]) < 15 and a["status"] == "in force"]
    rep["stub_articles_from_contents_only"] = [a["article"] for a in articles if a.get("remark")]
    rep["omitted_articles"] = [a["article"] for a in articles if a["status"] == "omitted"]
    rep["very_long_check_manually"] = [a["article"] for a in articles if len(a["text"]) > 15000]
    full = " ".join(a["text"] for a in articles)
    rep["replacement_chars"] = full.count("\ufffd")
    rep["notes_attached"] = sum(len(a["notes"]) for a in articles)
    rep["articles_without_notes_but_with_marker_digits"] = None
    known = {"1": "federal republic", "25": "equal before law", "35": "marriage, the family",
             "227": "injunctions of islam"}
    rep["known_string_failures"] = [k for k, s in known.items()
                                    if k not in got or s not in got[k]["text_plain"].lower().replace("  ", " ")]
    random.seed(7)
    sc = outdir / "spot_check"
    sc.mkdir(exist_ok=True)
    pages = sorted(random.sample(range(body_range[0], body_range[1]), min(spot_n, body_range[1] - body_range[0])))
    for p in pages:
        doc[p].get_pixmap(dpi=100).save(sc / f"page_{p + 1:03d}.png")
    rep["spot_check_pages"] = [p + 1 for p in pages]
    return rep


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="constitution.pdf")
    ap.add_argument("--source-url", default="UNKNOWN - fill in where you downloaded this PDF")
    ap.add_argument("--out", default="data")
    ap.add_argument("--spot", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "raw_pages").mkdir(parents=True, exist_ok=True)
    pdf = Path(args.pdf)
    doc = fitz.open(pdf)

    cover = doc[2].get_text().replace("\n", " ") if len(doc) > 2 else ""
    edition = re.search(r"\(As amended[^)]*\)", cover)
    meta = {"source_url": args.source_url, "file": pdf.name,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "processed": str(date.today()), "pdf_pages": len(doc),
            "edition_statement": edition.group(0) if edition else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for i, p in enumerate(doc, 1):
        (out / "raw_pages" / f"page_{i:03d}.txt").write_text(p.get_text(sort=True), encoding="utf-8")

    start, end = find_body_range(doc)
    if start is None or end is None:
        raise SystemExit(f"Could not locate the body (start={start}, end={end}). Check find_body_range().")
    print(f"Body = PDF pages {start + 1}-{end} ; edition: {meta['edition_statement']}")

    toc = parse_toc(doc, start)
    (out / "toc.json").write_text(json.dumps(toc, indent=1, ensure_ascii=False))
    print(f"TOC entries: {len(toc)}")

    articles, structure = parse_body(doc, start, end, {n for n, _ in toc})
    articles = finalize(articles, dict(toc))
    # Articles that exist only in the contents list (e.g. 270C: just a footnote in the body)
    have = {a["article"] for a in articles}
    for n, t in toc:
        if n not in have and re.match(r"^\W*(omitted|cease)", t, re.I):
            articles.append({"section": n, "article": n, "title": "", "part": None, "part_title": None,
                             "chapter": None, "chapter_title": None, "pdf_pages": [], "notes": [],
                             "title_side": "", "text": "", "text_plain": "", "clauses": [],
                             "status": "omitted", "doc_id": DOC_ID, "doc_type": "constitution",
                             "remark": "Listed in the contents as omitted; the body has only a footnote."})
    order = {n: i for i, (n, _) in enumerate(toc)}
    articles.sort(key=lambda a: order.get(a["article"], 9999))
    with open(out / "articles.jsonl", "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out / "structure.json").write_text(json.dumps(structure, indent=1, ensure_ascii=False))

    tail = []
    for pi in range(end, len(doc)):
        tail.append(f"=== PDF page {pi + 1} ===\n" + doc[pi].get_text(sort=True))
    (out / "schedules_and_annex.txt").write_text("\n".join(tail), encoding="utf-8")

    rep = validate(articles, toc, doc, (start, end), args.spot, out)
    (out / "validation_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()