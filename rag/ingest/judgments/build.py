"""Parse one judgment PDF (reusing the existing JSON where it is good) and turn it into chunks."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag.config import ChunkingConfig
from rag.ingest.chunker import Piece, TokenCounter, merge_small, split_by_sentence
from rag.ingest.citations import extract_case_cites, extract_statute_refs, quote_ratio
from rag.ingest.judgments.header import Header, human_date, parse_header
from rag.ingest.judgments.pdf_extract import extract_layout, group_blocks
from rag.ingest.judgments.structure import split_structure
from rag.text_utils import clean_for_embedding, one_line, split_sentences

PARSER_VERSION = "judgments-pdf-v2"


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def merge_with_existing(h: Header, existing: dict | None) -> dict[str, Any]:
    """Reuse the existing JSON's header values first (they were checked by hand when it was built);
    fill gaps from the PDF parse; record every disagreement."""
    out: dict[str, Any] = {}
    conflicts: list[dict] = []
    ex = existing or {}
    ex_parties = ex.get("parties") or {}

    def pick(field: str, pdf_val, ex_val):
        if ex_val not in (None, "", []):
            out[field] = ex_val
            src = "existing_json"
            if pdf_val not in (None, "", []) and _norm(str(pdf_val)) != _norm(str(ex_val)):
                conflicts.append({"field": field, "existing_json": ex_val, "pdf_parse": pdf_val})
        else:
            out[field] = pdf_val
            src = "pdf_parse" if pdf_val not in (None, "", []) else "missing"
        out.setdefault("_sources", {})[field] = src

    pick("petitioner", h.petitioner, ex_parties.get("petitioner"))
    pick("respondent", h.respondent, ex_parties.get("respondent") or None)
    pick("judge", h.judge, ex.get("judge"))
    ex_dec = (ex.get("date_of_decision") or "").split(",")[-1].strip() or None
    pick("date", h.date, ex_dec)
    pick("case_numbers", h.case_numbers, None)  # PDF parse gives normalised numbers
    out["case_numbers_existing_json"] = ex.get("case_numbers")
    for f in ("court", "bench", "case_type", "counsel", "dates", "date_basis"):
        out[f] = getattr(h, f)
    out["conflicts"] = conflicts
    out["unparsed"] = [f for f in ("court", "bench", "case_no", "parties", "date", "judge", "counsel")
                       if not ({"case_no": out["case_numbers"], "parties": out["petitioner"] and out["respondent"]}
                               .get(f, out.get(f)))]
    return out


def compare_with_existing(existing: dict | None, layout, structure, paragraphs) -> list[str]:
    """Gaps of the existing JSON relative to what this pipeline needs (recorded in HANDOFF.md)."""
    if existing is None:
        return ["no existing JSON for this PDF"]
    gaps = ["existing JSON has no paragraph segmentation (full_text and per-page text only)",
            "existing JSON has no counsel, case type, cited authorities or caption"]
    ft = existing.get("full_text", "")
    residue = sorted({r.text for r in layout.removed if r.kind in ("running_header", "page_number")
                      and len(r.text) > 3 and r.text in ft})
    if residue:
        gaps.append(f"running headers / page labels left in existing JSON text: {residue[:4]}")
    n_markers = sum(len(p["markers"]) for p in paragraphs)
    if n_markers:
        gaps.append(f"{n_markers} footnote reference markers are glued into words/dates in the existing JSON "
                    f"text (e.g. '26.07.20241'); removed here using superscript font flags")
    ex_fn = len(existing.get("footnotes") or [])
    if ex_fn != len(layout.footnotes):
        gaps.append(f"footnote count differs: existing JSON {ex_fn}, PDF layout parse {len(layout.footnotes)}")
    for f in ("judge",):
        if not existing.get(f):
            gaps.append(f"existing JSON missing '{f}'")
    if not (existing.get("parties") or {}).get("petitioner"):
        gaps.append("existing JSON missing parties")
    # word-level coverage
    mine = " ".join(structure.header_texts + [p["text"] for p in paragraphs] + structure.signature
                    + [f.text for f in layout.footnotes])
    tok = lambda s: Counter(re.findall(r"[A-Za-z]{3,}", s.lower()))
    a, b = tok(ft + " " + " ".join(f.get("text", "") for f in existing.get("footnotes") or [])), tok(mine)
    only_json = sum((a - b).values())
    only_pdf = sum((b - a).values())
    total = max(1, sum(a.values()))
    if only_json / total > 0.01 or only_pdf / total > 0.01:
        gaps.append(f"word-level difference vs existing JSON: {only_json} words only in JSON, {only_pdf} only in "
                    f"PDF parse (of {total}); top JSON-only: {[w for w, _ in (a - b).most_common(6)]}")
    return gaps


def parse_judgment(pdf_path: Path, existing_json: Path | None, doc_id: str, length_class: str) -> dict[str, Any]:
    layout = extract_layout(pdf_path)
    blocks = group_blocks(layout)
    st = split_structure(layout, blocks)
    first_text = st.units[0].text if st.units else None
    hdr = parse_header(st.header_texts, first_text)
    existing = json.loads(existing_json.read_text(encoding="utf-8")) if existing_json and existing_json.exists() else None
    meta = merge_with_existing(hdr, existing)
    fn_by_num = {f.number: f for f in layout.footnotes}
    paragraphs = []
    for u in st.units:
        fns = [{"number": m, "text": fn_by_num[m].text} for m in u.markers if m in fn_by_num]
        paragraphs.append({
            "label": u.label, "numbering": u.numbering, "page_start": u.page_start, "text": u.text,
            "markers": u.markers, "footnotes": fns, "indented_chars": st.indented_chars.get(u.label, 0),
        })
    return {
        "doc_id": doc_id,
        "parser_version": PARSER_VERSION,
        "source_file": pdf_path.as_posix(),
        "existing_json": existing_json.as_posix() if existing_json else None,
        "length_class": length_class,
        "scan_check": layout.scan_check,
        "n_pages": layout.n_pages,
        "body_font_size": layout.body_size,
        "title": st.title,
        "body_start_method": st.body_start_method,
        "header_text": st.header_texts,
        "header": meta,
        "paragraphs": paragraphs,
        "footnotes": [{"number": f.number, "text": f.text, "page": f.page} for f in layout.footnotes],
        "signature": st.signature + [t.text for t in layout.trailers],
        "removed_lines": [{"page": r.page, "kind": r.kind, "text": r.text} for r in layout.removed],
        "anomalies": st.anomalies,
        "gaps_vs_existing_json": compare_with_existing(existing, layout, st, paragraphs),
    }


def _case_short(case_no: str | None) -> str:
    if not case_no:
        return "case no. unknown"
    return re.sub(r"C\.P\. No\. (\S+) of (\d{4})", r"C.P. \1/\2", case_no)


def _subject(paragraphs: list[dict], max_words: int = 70) -> str:
    """Extractive subject line: the judgment's opening sentence(s), up to max_words."""
    if not paragraphs:
        return ""
    sents = split_sentences(one_line(paragraphs[0]["text"]))
    out: list[str] = []
    for s in sents:
        out.append(s)
        if len(" ".join(out).split()) >= 25:
            break
    words = " ".join(out).split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def judgment_chunks(parsed: dict, count: TokenCounter, cfg: ChunkingConfig, court_display: str,
                    jurisdiction: str) -> list[dict]:
    doc_id = parsed["doc_id"]
    h = parsed["header"]
    paras = parsed["paragraphs"]
    parties = f'{h.get("petitioner") or "Unknown"} v. {h.get("respondent") or "Unknown"}'
    case_no = (h.get("case_numbers") or [None])[0]
    date = h.get("date")
    base = {
        "doc_id": doc_id, "doc_type": "judgment", "act_title": None, "year": int(date[:4]) if date else None,
        "section": None, "article": None, "court": court_display, "bench": h.get("bench"),
        "case_no": case_no, "case_type": h.get("case_type"), "parties": parties, "judge": h.get("judge"),
        "date": date, "jurisdiction": jurisdiction, "status": "unverified", "as_of": None,
        "source_file": parsed["source_file"],
    }
    chunks: list[dict] = []

    # caption chunk (built from parsed header fields + the opening sentence; not verbatim PDF text)
    cap_lines = [
        f"Court: {court_display}" + (f" ({h['bench']})" if h.get("bench") else ""),
        f"Case: {', '.join(h.get('case_numbers') or []) or 'unknown'}" + (f" ({h['case_type']})" if h.get("case_type") else ""),
        f"Parties: {parties}",
        f"Date of decision: {date or 'unknown'}",
    ]
    if h.get("judge"):
        cap_lines.append(f"Judge: {h['judge']}")
    subj = _subject(paras)
    if subj:
        cap_lines.append(f"Subject: {subj}")
    cap_text = "\n".join(cap_lines)
    cap_header = f"{parties}, {_case_short(case_no)}, {court_display}, {human_date(date)}, case caption:"
    chunks.append({**base, "chunk_id": f"{doc_id}:caption#1", "parent_id": f"{doc_id}:caption",
                   "unit_ids": [f"{doc_id}:caption"], "unit_type": "caption", "title": parties,
                   "paragraph": None, "piece": 1, "n_pieces": 1, "pdf_page_start": 1, "text": cap_text,
                   "embed_text": f"{cap_header}\n{cap_text}" if cfg.header_in_embed_text else cap_text,
                   "contains_quote": False, "cites_cases": [], "cites_statutes": [],
                   "flags": ["constructed_caption"], "counsel": h.get("counsel")})

    # paragraph pieces
    last_label = paras[-1]["label"] if paras else None
    pieces: list[Piece] = []
    for p in paras:
        uid = f"{doc_id}:{p['label']}"
        unit_type = "operative_order" if p["label"] == last_label else "paragraph"
        meta = {"unit_type": unit_type, "page_start": p["page_start"], "footnotes": p["footnotes"],
                "indented_chars": p["indented_chars"], "numbering": p["numbering"], "full_len": len(p["text"])}
        if count(p["text"]) > cfg.hard_cap_tokens:
            parts = split_by_sentence(p["text"], count, cfg.target_max_tokens, cfg.hard_cap_tokens, cfg.overlap_tokens)
            for i, t in enumerate(parts, start=1):
                pieces.append(Piece(text=t, unit_ids=[uid], labels=[p["label"]], split=True, piece_no=i,
                                    meta={**meta, "n_pieces": len(parts)}))
        else:
            pieces.append(Piece(text=p["text"], unit_ids=[uid], labels=[p["label"]], meta={**meta, "n_pieces": 1}))

    def barrier(a: Piece, b: Piece) -> bool:
        return (a.split or b.split or a.meta.get("unit_type") == "operative_order"
                or b.meta.get("unit_type") == "operative_order")

    pieces = merge_small(pieces, count, cfg.merge_below_tokens, cfg.hard_cap_tokens, barrier=barrier, joiner="\n\n")

    for pc in pieces:
        label0 = pc.labels[0]
        para_field = label0 if len(pc.labels) == 1 else f"{pc.labels[0]}-{pc.labels[-1]}"
        unit_type = pc.meta.get("unit_type", "paragraph")
        fns = [f for lab in pc.labels for p in paras if p["label"] == lab for f in p["footnotes"]]
        para_word = "para" if len(pc.labels) == 1 else "paras"
        suffix = " (operative order)" if unit_type == "operative_order" else ""
        part = f" (part {pc.piece_no} of {pc.meta['n_pieces']})" if pc.split else ""
        header = f"{parties}, {_case_short(case_no)}, {court_display}, {human_date(date)}, {para_word} {para_field}{suffix}{part}:"
        body = clean_for_embedding(pc.text)
        indented = pc.meta.get("indented_chars", 0) if not pc.split and len(pc.labels) == 1 else 0
        qr = quote_ratio(pc.text, indented)
        fn_text = " ".join(f["text"] for f in fns)
        flags = []
        if pc.meta.get("numbering") in ("inferred", "intro"):
            flags.append(f"paragraph_number_{pc.meta['numbering']}")
        if pc.meta.get("merged"):
            flags.append("merged_small_paragraphs")
        chunks.append({
            **base,
            "chunk_id": f"{doc_id}:{label0}#{pc.piece_no}",
            "parent_id": f"{doc_id}:{label0}",
            "unit_ids": pc.unit_ids,
            "unit_type": unit_type,
            "title": parties,
            "paragraph": para_field,
            "piece": pc.piece_no,
            "n_pieces": pc.meta.get("n_pieces", 1),
            "pdf_page_start": pc.meta.get("page_start"),
            "text": pc.text,
            "embed_text": f"{header}\n{body}" if cfg.header_in_embed_text else body,
            "contains_quote": qr > 0.5,
            "quote_ratio": round(qr, 3),
            "cites_cases": extract_case_cites(pc.text + " " + fn_text),
            "cites_statutes": extract_statute_refs(pc.text + " " + fn_text),
            "footnotes": fns,
            "flags": flags,
        })
    return chunks
