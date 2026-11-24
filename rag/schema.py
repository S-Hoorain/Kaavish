"""Chunk record schema, validation of required metadata, and Chroma metadata flattening."""
from __future__ import annotations

from typing import Any

# Order of keys in chunks.jsonl (documented in HANDOFF.md section 6).
CHUNK_FIELDS = [
    "chunk_id", "parent_id", "unit_ids", "doc_id", "doc_type", "unit_type", "title",
    "act_title", "year", "section", "article", "paragraph", "piece", "n_pieces",
    "court", "bench", "case_no", "case_type", "parties", "judge", "date",
    "jurisdiction", "status", "as_of", "source_file", "pdf_page_start",
    "text", "embed_text", "token_count", "embed_token_count", "contains_quote",
    "cites_cases", "cites_statutes", "flags", "corpus_version",
]

REQUIRED = {
    # FR-KB-02 / task spec. Constitution articles are validated as statutes with `article` as the section.
    "statute": ["act_title", "year", "section", "jurisdiction", "as_of"],
    "judgment": ["court", "date", "jurisdiction", "paragraph"],
}

# Values that count as "missing".
_MISSING = (None, "", [], "TODO", "UNKNOWN")

# Only these scalar fields go to Chroma metadata (strings, numbers, booleans only; no None, no lists).
CHROMA_FIELDS = [
    "chunk_id", "parent_id", "doc_id", "doc_type", "unit_type", "title", "act_title", "year",
    "section", "article", "paragraph", "piece", "n_pieces", "court", "case_no", "case_type",
    "date", "jurisdiction", "status", "as_of", "source_file", "pdf_page_start", "token_count",
    "contains_quote", "corpus_version",
]


def family(doc_type: str) -> str:
    return "judgment" if doc_type == "judgment" else "statute"


def missing_fields(rec: dict[str, Any], strict_as_of: bool = False) -> list[str]:
    fam = family(rec.get("doc_type", ""))
    req = list(REQUIRED[fam])
    missing = []
    for f in req:
        v = rec.get(f)
        if f == "section" and fam == "statute" and rec.get("doc_type") == "constitution":
            v = rec.get("article")
        if f == "paragraph" and rec.get("unit_type") == "caption":
            continue  # a caption chunk describes the whole judgment, it has no paragraph
        if f == "as_of" and v == "TODO" and not strict_as_of:
            continue  # flagged instead (see flags: as_of_todo)
        if v in _MISSING:
            missing.append(f)
    return missing


def order_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = {k: rec.get(k) for k in CHUNK_FIELDS}
    extra = {k: v for k, v in rec.items() if k not in out}
    out.update(extra)
    return out


def chroma_metadata(rec: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for k in CHROMA_FIELDS:
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            meta[k] = v
        else:
            meta[k] = str(v)
    return meta
