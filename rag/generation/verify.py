"""Citation check (FR-EVID-02, NFR-INT-05): every quote must be a verbatim substring of the cited
chunk's `text` after normalising whitespace and quote characters only.

Failures are kept and marked (verified=false with a reason); nothing is dropped."""
from __future__ import annotations

from typing import Any, Mapping

from rag.text_utils import normalise_for_match

REASONS = {
    "ok": "quote found verbatim in the cited passage",
    "no_citation": "claim has no chunk_id/quote (closed-book C0 answer)",
    "missing_quote": "claim cites a chunk but gives no quote",
    "missing_chunk_id": "claim gives a quote but no chunk_id",
    "unknown_chunk": "chunk_id does not exist in the corpus",
    "chunk_not_in_context": "chunk_id exists but was not among the passages given to the model",
    "quote_not_found": "quote is not a verbatim substring of the cited passage",
}


def quote_in_text(quote: str, text: str) -> bool:
    q = normalise_for_match(quote)
    return bool(q) and q in normalise_for_match(text)


def verify_claims(claims: list[dict[str, Any]], context: Mapping[str, str],
                  corpus: Mapping[str, Any] | None = None, mode: str = "c1") -> tuple[list[dict], dict]:
    """context: chunk_id -> text of the passages shown to the model.
    corpus: optional chunk_id -> record, to tell 'unknown' from 'not in context'."""
    out = []
    for c in claims or []:
        cid = (c.get("chunk_id") or "").strip() if isinstance(c.get("chunk_id"), str) else c.get("chunk_id")
        quote = c.get("quote")
        rec = dict(c)
        if not cid and not quote:
            rec["verified"], rec["reason"] = (None, "no_citation") if mode == "c0" else (False, "no_citation")
        elif not quote:
            rec["verified"], rec["reason"] = False, "missing_quote"
        elif not cid:
            rec["verified"], rec["reason"] = False, "missing_chunk_id"
        elif cid not in context:
            known = corpus is not None and cid in corpus
            rec["verified"], rec["reason"] = False, ("chunk_not_in_context" if known else "unknown_chunk")
        elif quote_in_text(quote, context[cid]):
            rec["verified"], rec["reason"] = True, "ok"
        else:
            rec["verified"], rec["reason"] = False, "quote_not_found"
        out.append(rec)
    checked = [r for r in out if r["verified"] is not None]
    n_ok = sum(1 for r in checked if r["verified"])
    summary = {
        "n_claims": len(out),
        "n_checked": len(checked),
        "n_verified": n_ok,
        "n_failed": len(checked) - n_ok,
        "verification_rate": round(n_ok / len(checked), 4) if checked else None,
        "failures": [{"chunk_id": r.get("chunk_id"), "reason": r["reason"], "quote": r.get("quote")}
                     for r in checked if not r["verified"]],
    }
    return out, summary
