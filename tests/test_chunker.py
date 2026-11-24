"""Chunker merge and split rules (whitespace token counter, so no model download is needed)."""
from __future__ import annotations

from rag.config import ChunkingConfig
from rag.ingest.chunker import Piece, merge_small, pack_segments, split_by_sentence
from rag.ingest.statutes import ingest_statutes
from rag.tokens import whitespace_counter as wc

CFG = ChunkingConfig(name="test", target_min_tokens=20, target_max_tokens=40, hard_cap_tokens=60,
                     merge_below_tokens=10, overlap_tokens=8, header_in_embed_text=True)


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def sentences(n: int, per: int, tag: str = "W") -> str:
    return " ".join(f"{words(per - 1, f'{tag}{s}_')} end{s}." for s in range(n))


def test_unit_under_cap_is_one_piece():
    t = sentences(3, 10)
    assert split_by_sentence(t, wc, 40, 60, 8) == [t]


def test_split_by_sentence_respects_target_and_overlap():
    t = sentences(12, 10)  # 120 tokens, sentences of 10 tokens
    pieces = split_by_sentence(t, wc, 40, 60, 8)
    assert len(pieces) >= 3
    for p in pieces:
        assert wc(p) <= 60
        assert p in t, "every piece must be a verbatim substring"
    # ~overlap: the last sentence of a window starts the next one when it fits in the overlap budget
    # (here sentences are 10 tokens > 8, so no overlap is possible) -> consecutive, no gaps
    joined = " ".join(pieces)
    assert all(f"end{s}." in joined for s in range(12))


def test_split_overlap_repeats_short_sentences():
    t = " ".join(f"{words(3, f'S{s}_')} end{s}." for s in range(30))  # 4-token sentences (capitalised)
    pieces = split_by_sentence(t, wc, 40, 60, 8)
    assert len(pieces) >= 3
    tail = pieces[0].split(". ")[-1]
    assert tail.rstrip(".") in pieces[1], "the next window should start with the previous window's last sentence(s)"


def test_tiny_tail_is_folded_into_previous_window():
    t = sentences(4, 10) + " short tail."
    pieces = split_by_sentence(t, wc, 40, 60, 0)
    assert wc(pieces[-1]) >= 10


def test_pack_segments_greedy_and_cap():
    segs = [words(15, "a"), words(15, "b"), words(15, "c"), words(70, "d")]
    out = pack_segments(segs, wc, 40, 60, 8, joiner=" ")
    assert out[0].startswith("a0") and "b14" in out[0] and "c0" not in out[0]
    assert all(wc(p) <= 60 for p in out)


def test_merge_small_with_next_and_last_with_previous():
    ps = [Piece("x " * 3, ["D:1"], ["1"]), Piece(words(20, "y"), ["D:2"], ["2"]), Piece("z " * 4, ["D:3"], ["3"])]
    out = merge_small(ps, wc, 10, 60)
    assert len(out) == 1
    assert out[0].unit_ids == ["D:1", "D:2", "D:3"]


def test_merge_small_respects_barrier_and_cap():
    ps = [Piece("x " * 3, ["D:1"], ["1"]), Piece(words(20, "y"), ["D:2"], ["2"], meta={"unit_type": "operative_order"})]
    out = merge_small(ps, wc, 10, 60, barrier=lambda a, b: b.meta.get("unit_type") == "operative_order")
    assert len(out) == 2
    big = [Piece("x " * 3, ["D:1"], ["1"]), Piece(words(59, "y"), ["D:2"], ["2"])]
    assert len(merge_small(big, wc, 10, 60)) == 2, "merging must not exceed the hard cap"


def _statute_docs(tmp_path, records):
    import json

    (tmp_path / "acts").mkdir()
    with (tmp_path / "acts" / "T.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return {"T-2000": {"file": "acts/T.jsonl", "pdf": "acts/T.pdf", "display_title": "Test Act, 2000",
                       "unit_name": "Section", "jurisdiction": "Pakistan"}}


def test_statute_sections_never_merge_and_split_at_clauses(tmp_path):
    long_clauses = [{"id": str(i), "text": f"({i}) " + words(25, f"c{i}_") + "."} for i in range(1, 5)]
    long_text = "\n".join(c["text"] for c in long_clauses)
    records = [
        {"doc_id": "T-2000", "section": "1", "section_key": "1", "title": "Short.", "status": "in force",
         "text": "(1) tiny.", "clauses": [], "paragraphs": [], "pdf_pages": [1, 1]},
        {"doc_id": "T-2000", "section": "2", "section_key": "2", "title": "Short too.", "status": "in force",
         "text": "(1) also tiny.", "clauses": [], "paragraphs": [], "pdf_pages": [1, 1]},
        {"doc_id": "T-2000", "section": "3", "section_key": "3", "title": "Long.", "status": "in force",
         "text": long_text, "clauses": long_clauses, "paragraphs": [], "pdf_pages": [2, 2]},
        {"doc_id": "T-2000", "section": "4", "section_key": "4", "title": "Gone.", "status": "repealed",
         "text": "Rep. by ...", "clauses": [], "paragraphs": [], "pdf_pages": [3, 3]},
    ]
    docs = _statute_docs(tmp_path, records)
    chunks, skipped, stats = ingest_statutes(tmp_path, docs, {"T-2000": {"as_of": "1-1-2020"}},
                                             ["omitted", "repealed"], wc, CFG)
    ids = [c["chunk_id"] for c in chunks]
    assert "T-2000:1#1" in ids and "T-2000:2#1" in ids, "small sections stay separate (never merged across sections)"
    s3 = [c for c in chunks if c["parent_id"] == "T-2000:3"]
    assert len(s3) >= 2
    assert [c["chunk_id"] for c in s3] == [f"T-2000:3#{i}" for i in range(1, len(s3) + 1)]
    for c in s3:
        assert c["text"] in long_text
        assert c["embed_text"].startswith("Test Act, 2000, Section 3: Long.")
        assert wc(c["text"]) <= CFG.hard_cap_tokens
    assert [s["unit_id"] for s in skipped] == ["T-2000:4"]
