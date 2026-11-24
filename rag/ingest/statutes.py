"""Statute and Constitution ingestion: one chunk per section/article, split at clause boundaries
when a unit is over the hard cap."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from rag.config import ChunkingConfig
from rag.ingest.chunker import Piece, TokenCounter, merge_small, pack_segments
from rag.ingest.citations import extract_statute_refs
from rag.text_utils import clean_for_embedding, one_line


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _locate_starts(text: str, seg_texts: list[str], lo: int = 0, hi: int | None = None) -> list[int] | None:
    """Return the start offset of each segment inside text[lo:hi], searched in order.
    Returns None if any segment cannot be found verbatim."""
    hi = len(text) if hi is None else hi
    starts, pos = [], lo
    for s in seg_texts:
        probe = s.strip()[:40]
        if not probe:
            continue
        i = text.find(probe, pos, hi)
        if i < 0:
            return None
        starts.append(i)
        pos = i + len(probe)
    return starts


def _slices(text: str, starts: list[int], lo: int = 0, hi: int | None = None) -> list[str]:
    hi = len(text) if hi is None else hi
    if not starts:
        return [text[lo:hi]]
    starts = sorted(set(starts))
    if starts[0] != lo:
        starts = [lo] + starts  # keep any lead-in (e.g. "(1) ") with the first clause
    bounds = starts + [hi]
    return [text[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def clause_segments(rec: dict[str, Any], count: TokenCounter, cap: int) -> tuple[list[str], str]:
    """Split a unit's verbatim text at clause boundaries. Uses `clauses`; a clause still over the cap
    is refined with the finer `paragraphs` boundaries. Returns (segments, method)."""
    text = rec.get("text") or ""
    clauses = [c.get("text", "") for c in rec.get("clauses") or []]
    paras = [p.get("text", "") for p in rec.get("paragraphs") or []]
    method = "clauses"
    starts = _locate_starts(text, clauses) if clauses else None
    if not starts:
        method = "paragraphs"
        starts = _locate_starts(text, paras) if paras else None
    if not starts:
        return [text], "none"
    segs = _slices(text, starts)
    if method == "clauses" and paras:
        refined: list[str] = []
        offset = 0
        for seg in segs:
            seg_lo = text.find(seg, offset)
            offset = seg_lo + len(seg)
            if count(seg) > cap:
                inner = [p for p in paras if p.strip()[:40] and seg.find(p.strip()[:40]) >= 0]
                pstarts = _locate_starts(text, inner, seg_lo, seg_lo + len(seg))
                if pstarts:
                    refined.extend(_slices(text, pstarts, seg_lo, seg_lo + len(seg)))
                    method = "clauses+paragraphs"
                    continue
            refined.append(seg)
        segs = refined
    return segs, method


def statute_header(display_title: str, unit_name: str, num: str, title: str | None,
                   piece_no: int, n_pieces: int) -> str:
    t = one_line(title or "")
    if t and not t.endswith((".", ":", "?")):
        t += "."
    head = f"{display_title}, {unit_name} {num}: {t}".rstrip(": ").strip()
    if n_pieces > 1:
        head += f" (part {piece_no} of {n_pieces})"
    return head


def ingest_statutes(raw_dir: Path, documents: dict[str, Any], source_dates: dict[str, Any],
                    skip_statuses: list[str], count: TokenCounter, cfg: ChunkingConfig
                    ) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Returns (chunk records without corpus_version, skipped records, stats per doc)."""
    chunks: list[dict] = []
    skipped: list[dict] = []
    stats: dict[str, Any] = {}
    for doc_id, d in documents.items():
        path = raw_dir / d["file"]
        doc_type_default = d.get("doc_type")
        as_of = str((source_dates.get(doc_id) or {}).get("as_of") or "").strip() or None
        st = stats.setdefault(doc_id, {"records": 0, "skipped": 0, "units_indexed": 0, "chunks": 0,
                                       "units_split": 0, "split_methods": {}})
        for rec in _read_jsonl(path):
            st["records"] += 1
            doc_type = rec.get("doc_type") or doc_type_default or "act"
            is_const = doc_type == "constitution"
            num = str(rec.get("article") if is_const else (rec.get("section_key") or rec.get("section")))
            unit_id = f"{doc_id}:{num}"
            status = (rec.get("status") or "").strip().lower()
            if status in skip_statuses:
                skipped.append({"unit_id": unit_id, "doc_id": doc_id, "status": status,
                                "title": rec.get("title"), "reason": f"status '{status}' is not indexed",
                                "text": rec.get("text")})
                st["skipped"] += 1
                continue
            text = rec.get("text") or ""
            if count(text) <= cfg.hard_cap_tokens:
                segs, method = [text], "whole"
                pieces_txt = [text.strip()]
            else:
                segs, method = clause_segments(rec, count, cfg.hard_cap_tokens)
                pieces_txt = pack_segments(segs, count, cfg.target_max_tokens, cfg.hard_cap_tokens,
                                           cfg.overlap_tokens, joiner="")
                st["units_split"] += 1
                st["split_methods"][method] = st["split_methods"].get(method, 0) + 1
            pieces = [Piece(text=p.strip(), unit_ids=[unit_id], labels=[num]) for p in pieces_txt if p.strip()]
            if len(pieces) > 1:
                pieces = merge_small(pieces, count, cfg.merge_below_tokens, cfg.hard_cap_tokens, joiner="\n")
            n = len(pieces)
            st["units_indexed"] += 1
            for i, p in enumerate(pieces, start=1):
                header = statute_header(d["display_title"], d["unit_name"], num, rec.get("title"), i, n)
                body = clean_for_embedding(p.text)
                embed_text = f"{header}\n{body}" if cfg.header_in_embed_text else body
                flags = []
                if as_of == "TODO":
                    flags.append("as_of_todo")
                if n > 1:
                    flags.append(f"split:{method}")
                chunks.append({
                    "chunk_id": f"{unit_id}#{i}",
                    "parent_id": unit_id,
                    "unit_ids": [unit_id],
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "unit_type": "article" if is_const else "section",
                    "title": rec.get("title"),
                    "act_title": d["display_title"],
                    "year": d.get("year") or rec.get("year"),
                    "section": None if is_const else num,
                    "article": num if is_const else None,
                    "paragraph": None,
                    "piece": i,
                    "n_pieces": n,
                    "court": None, "bench": None, "case_no": None, "case_type": None,
                    "parties": None, "judge": None, "date": None,
                    "jurisdiction": d["jurisdiction"],
                    "status": status or None,
                    "as_of": as_of,
                    "source_file": f"data/raw/{d['pdf']}",
                    "pdf_page_start": (rec.get("pdf_pages") or [None])[0],
                    "text": p.text,
                    "embed_text": embed_text,
                    "contains_quote": False,
                    "cites_cases": [],
                    "cites_statutes": extract_statute_refs(p.text),
                    "flags": flags,
                    "part": rec.get("part"), "part_title": rec.get("part_title"),
                    "chapter": rec.get("chapter"), "chapter_title": rec.get("chapter_title"),
                    "amendment_notes": rec.get("notes") or [],
                })
                st["chunks"] += 1
    return chunks, skipped, stats
