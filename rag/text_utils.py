"""Small text helpers shared by ingestion, prompting and citation checking."""
from __future__ import annotations

import re
import unicodedata

# Sentence boundary: end punctuation followed by space and an upper-case letter, digit, quote or '('.
# Avoids splitting on common legal abbreviations (No., s., Art., Mst., Mr., v., r/w, etc.).
_ABBREV = {
    "no", "nos", "s", "ss", "art", "arts", "mst", "mr", "mrs", "ms", "dr", "sr", "jr", "v", "vs",
    "st", "cl", "para", "paras", "sec", "secs", "ord", "i.e", "e.g", "viz", "etc", "co", "ltd", "pvt",
    "c.p", "cp", "j", "jj", "pld", "scmr", "clc", "mld", "ylr", "p.cr.l.j", "pcrlj", "addl", "asstt",
    "dy", "govt", "hon'ble", "honble", "u/s", "r/w", "a.a.g", "a.g", "spl", "p", "pp", "vol", "ed",
}
_SENT_END = re.compile(r'(?<=[.!?;:])["”’)]?\s+(?=["“‘(]?[A-Z0-9])')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping every character (pieces re-join to the original)."""
    pieces: list[str] = []
    start = 0
    for m in _SENT_END.finditer(text):
        cand = text[start:m.start()]
        last_word = re.findall(r"([A-Za-z.'/]+)\.\s*$", cand + " ")
        if cand.rstrip().endswith(".") and last_word:
            w = last_word[-1].lower().rstrip(".")
            if w in _ABBREV or len(w) == 1:
                continue
        pieces.append(text[start:m.end()])
        start = m.end()
    if start < len(text):
        pieces.append(text[start:])
    return [p for p in pieces if p.strip()]


_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
}
_QUOTE_TRANS = str.maketrans(_QUOTE_MAP)


def normalise_for_match(text: str) -> str:
    """Normalisation used by the quote verifier: unify quote characters and collapse whitespace.

    Deliberately nothing else (no case folding, no dash or punctuation changes), so a match
    still means the quote is verbatim.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text).translate(_QUOTE_TRANS)
    t = t.replace(" ", " ").replace("​", "")
    return re.sub(r"\s+", " ", t).strip()


# --- cleaning for embed_text ---------------------------------------------------------------
_FOOTNOTE_BEFORE_BRACKET = re.compile(r"(?<![\w)])\d{1,3}(?=\[)")   # "1[(2) It extends" -> "[(2) It extends"
_STARS = re.compile(r"(\*\s*){3,}")
_BRACKETS = re.compile(r"[\[\]]")


def clean_for_embedding(text: str) -> str:
    """Remove amendment brackets, '* * *' omission marks and footnote numbers; collapse spaces."""
    t = _FOOTNOTE_BEFORE_BRACKET.sub("", text)
    t = _STARS.sub(" ", t)
    t = _BRACKETS.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s*\n\s*", "\n", t)
    return t.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
