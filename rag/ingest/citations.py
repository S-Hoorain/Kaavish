"""Extraction of cited case law and statute references, and the quoted-text ratio of a paragraph.

These lists are stored in chunks.jsonl only (not in Chroma metadata, which cannot hold lists).
They are regex-based and will miss unusual citation styles; misses are acceptable for R0 because
nothing downstream filters on them yet.
"""
from __future__ import annotations

import re

_REPORTERS = (
    r"SCMR|MLD|CLC(?:\s*\(N\))?|Y\s?\.?\s?L\s?\.?\s?R\.?|P\.?\s?Cr\.?\s?L\.?\s?J\.?|PCr\.?LJ|CLD|PLC(?:\s*\(C\.S\.\))?|"
    r"PTD|SBLR|NLR|KLR|CLR|PLJ|GBLR|MLN|YLR|PSC"
)
_COURT_WORD = r"(?:SC|S\.C\.|FSC|F\.S\.C\.|Supreme\s+Court|Kar(?:achi)?\.?|Lah(?:ore)?\.?|Quetta|Pesh(?:awar)?\.?|Isl(?:amabad)?\.?|" \
              r"Sindh|Bal(?:uchistan)?\.?|Azad\s+J\s?&\s?K|AJ&K|SC\s*\(AJ&K\)|Note|[A-Z][a-z]+)"

CASE_PATTERNS = [
    # PLD 2026 SC 20 | PLD 1976 Quetta 97 | PLD 2011 Supreme Court 1
    re.compile(r"\bPLD\s+(?:\(\s*)?(?:19|20)\d{2}\s+" + _COURT_WORD + r"\s+\d{1,4}\b"),
    # 2023 SCMR 1434 | 2014 MLD (Lahore) 1137 | 2006 Y L R 2623 | 2010 P.Cr.L.J 123
    re.compile(r"\b(?:19|20)\d{2}\s+(?:" + _REPORTERS + r")\s*(?:\([A-Za-z.& ]{2,30}\)\s*)?\d{1,4}\b"),
]

_ACT_NAME = r"(?:the\s+)?(?:[A-Z][\w&'’.-]*\s+){0,8}?(?:Act|Ordinance|Code|Order|Rules|Constitution(?:\s+of\s+(?:the\s+)?Islamic\s+Republic\s+of\s+Pakistan)?)(?:,?\s*\(?(?:18|19|20)\d{2}\)?)?"
_ABBR = r"(?:C\.?P\.?C\.?|Cr\.?\s?P\.?C\.?|P\.?P\.?C\.?|G\s?&\s?W\.?\s?Act|GWA(?:\s*1890)?|FCA(?:\s*1964)?|MFLO(?:\s*1961)?|DMMA(?:\s*1939)?)"
_NUM = r"\d+[A-Z]?(?:-?[A-Z])?(?:\s*\(\s*[\dA-Za-z]+\s*\))*"
STATUTE_PATTERN = re.compile(
    r"\b(?:[Ss]ections?|[Ss]ec\.|[Ss]s?\.|[Aa]rticles?|[Aa]rt\.|[Oo]rder\s+[IVXL]+\s+[Rr]ule)\s*" + _NUM
    + r"(?:\s*(?:,|and|&|/|to|r/w|read\s+with)\s*(?:[Ss]ections?\s*)?" + _NUM + r")*"
    + r"\s+(?:(?:of|under)\s+)?(?:" + _ABBR + r"|" + _ACT_NAME + r")"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().rstrip(",.;")


def extract_case_cites(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for pat in CASE_PATTERNS:
        for m in pat.finditer(text):
            seen.setdefault(_norm(m.group(0)), None)
    return list(seen)


def extract_statute_refs(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in STATUTE_PATTERN.finditer(text):
        ref = _norm(m.group(0))
        # require a recognisable law name at the end (avoid "Section 5 of the plaint")
        if re.search(r"(Act|Ordinance|Code|Order|Rules|Constitution|Pakistan|\d{4}|C\.?P\.?C|P\.?C|GWA|FCA|MFLO|DMMA)\.?\)?$", ref):
            seen.setdefault(ref, None)
    return list(seen)


_QUOTED = re.compile(r"[“\"]([^“”\"]{20,}?)[”\"]", re.S)


def quote_ratio(text: str, indented_chars: int = 0) -> float:
    """Share of the paragraph's characters that are quoted from elsewhere: text inside double
    quotation marks plus (when known from the PDF layout) indented block-quote lines."""
    if not text:
        return 0.0
    quoted = sum(len(m.group(1)) for m in _QUOTED.finditer(text))
    quoted = max(quoted, indented_chars)
    return min(1.0, quoted / max(1, len(text.strip())))
