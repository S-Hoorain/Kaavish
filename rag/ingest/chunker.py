"""Chunking rules shared by statutes and judgments.

Rules (config/chunking_variants.yaml, BGE-M3 token counts):
* A unit (statute section/article, judgment paragraph) that fits under the hard cap stays whole.
* An over-cap statute unit is split at clause boundaries; clauses are packed greedily up to
  target_max tokens. A single clause over the cap is split by sentence.
* An over-cap judgment paragraph is split by sentence with ~overlap_tokens of overlap.
* Pieces under merge_below tokens are merged with the NEXT piece of the same document
  (or the previous one when they are last). Never across documents; callers pass barriers so
  nothing is merged across statute sections, captions or the operative order.
* Every piece's text is a verbatim substring (or a blank-line join of verbatim substrings) of the
  source, so quotes can be checked against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from rag.text_utils import split_sentences

TokenCounter = Callable[[str], int]


@dataclass
class Piece:
    text: str
    unit_ids: list[str]                 # parent units covered, e.g. ["SHC-L05:3", "SHC-L05:4"]
    labels: list[str]                   # paragraph numbers / clause labels covered
    split: bool = False                 # True when this is one of several pieces of one unit
    piece_no: int = 1                   # 1-based index within its parent
    meta: dict = field(default_factory=dict)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for s in split_sentences(text):
        i = text.find(s, pos)
        if i < 0:  # should not happen: split_sentences keeps every character
            i = pos
        spans.append((i, i + len(s)))
        pos = i + len(s)
    return spans


def _word_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    import re

    return [(start + m.start(), start + m.end()) for m in re.finditer(r"\S+\s*", text[start:end])]


def split_by_sentence(text: str, count: TokenCounter, target_max: int, cap: int,
                      overlap: int, min_tail: int = 60) -> list[str]:
    """Split text into windows of whole sentences (<= target_max tokens, never > cap)
    with roughly `overlap` tokens repeated between consecutive windows. A last window shorter
    than `min_tail` tokens is folded into the previous one when that stays under the cap."""
    if count(text) <= cap:
        return [text.strip()]
    pieces = _sentence_windows(text, count, target_max, overlap)
    if len(pieces) >= 2 and count(pieces[-1]) < min_tail:
        start_prev = text.find(pieces[-2])
        end_last = text.find(pieces[-1], start_prev) + len(pieces[-1])
        joined = text[start_prev:end_last].strip()
        if start_prev >= 0 and count(joined) <= cap:
            pieces[-2:] = [joined]
    return pieces


def _sentence_windows(text: str, count: TokenCounter, target_max: int, overlap: int) -> list[str]:
    spans = _sentence_spans(text)
    # break sentences that are themselves too long into word windows
    units: list[tuple[int, int]] = []
    for s, e in spans:
        if count(text[s:e]) <= target_max:
            units.append((s, e))
            continue
        words = _word_spans(text, s, e)
        cur_s = words[0][0]
        cur_e = cur_s
        for ws, we in words:
            if cur_e > cur_s and count(text[cur_s:we]) > target_max:
                units.append((cur_s, cur_e))
                cur_s = ws
            cur_e = we
        units.append((cur_s, cur_e))

    pieces: list[str] = []
    i = 0
    n = len(units)
    while i < n:
        j = i
        while j + 1 < n and count(text[units[i][0]:units[j + 1][1]]) <= target_max:
            j += 1
        pieces.append(text[units[i][0]:units[j][1]].strip())
        if j + 1 >= n:
            break
        # step back so the next window starts with ~overlap tokens of this one
        k = j + 1
        back = j
        while back > i and count(text[units[back][0]:units[j][1]]) <= overlap:
            k = back
            back -= 1
        # only overlap if the next window can still move past this one
        if k <= i or count(text[units[k][0]:units[j + 1][1]]) > target_max:
            k = j + 1
        i = k
    return [p for p in pieces if p]


def pack_segments(segments: Sequence[str], count: TokenCounter, target_max: int, cap: int,
                  overlap: int, joiner: str = "\n") -> list[str]:
    """Greedily pack consecutive segments (clauses) into pieces of <= target_max tokens.
    A single segment over the cap is split by sentence."""
    out: list[str] = []
    cur: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if count(seg) > cap:
            if cur:
                out.append(joiner.join(cur))
                cur = []
            out.extend(split_by_sentence(seg, count, target_max, cap, overlap))
            continue
        trial = joiner.join(cur + [seg])
        if cur and count(trial) > target_max:
            out.append(joiner.join(cur))
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        out.append(joiner.join(cur))
    return out


def merge_small(pieces: list[Piece], count: TokenCounter, merge_below: int, cap: int,
                barrier: Callable[[Piece, Piece], bool] | None = None,
                joiner: str = "\n\n") -> list[Piece]:
    """Merge every piece under `merge_below` tokens into the next piece (or into the previous one
    if it is the last), unless a barrier forbids it or the result would exceed the cap."""
    barrier = barrier or (lambda a, b: False)
    out = list(pieces)
    changed = True
    while changed:
        changed = False
        for idx, p in enumerate(out):
            if count(p.text) >= merge_below:
                continue
            nxt = out[idx + 1] if idx + 1 < len(out) else None
            prv = out[idx - 1] if idx > 0 else None
            if nxt is not None and not barrier(p, nxt) and count(p.text + joiner + nxt.text) <= cap:
                out[idx: idx + 2] = [_combine(p, nxt, joiner)]
                changed = True
                break
            if nxt is None and prv is not None and not barrier(prv, p) \
                    and count(prv.text + joiner + p.text) <= cap:
                out[idx - 1: idx + 1] = [_combine(prv, p, joiner)]
                changed = True
                break
    return out


def _combine(a: Piece, b: Piece, joiner: str) -> Piece:
    ids = a.unit_ids + [u for u in b.unit_ids if u not in a.unit_ids]
    labels = a.labels + [l for l in b.labels if l not in a.labels]
    meta = {**b.meta, **a.meta}
    meta["merged"] = True
    return Piece(text=a.text + joiner + b.text, unit_ids=ids, labels=labels,
                 split=a.split or b.split, piece_no=a.piece_no, meta=meta)
