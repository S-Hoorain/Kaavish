"""R1 BM25 + reciprocal rank fusion, with a fake store and a fake Chroma collection (no model download)."""
from __future__ import annotations

from rag.retrieval.bm25 import BM25Index, rrf, tokenize
from rag.retrieval.retriever import RetrievedChunk, Retriever

DOCS = {
    "GWA-1890:17#1": "Matters to be considered by the Court in appointing guardian. welfare of the minor",
    "GWA-1890:25#1": "Title of guardian to custody of ward. If a ward leaves or is removed from the custody",
    "CONST:63A#1": "Disqualification on grounds of defection. member of a parliamentary party",
    "SHC-L16#3": "hizanat of the minor remains with the mother; visitation schedule for the father",
}


class FakeStore:
    def __init__(self):
        self.by_id = {cid: {"chunk_id": cid, "text": t, "embed_text": t, "unit_ids": [cid.split("#")[0]]}
                      for cid, t in DOCS.items()}

    def get(self, cid):
        return self.by_id.get(cid)


class FakeCollection:
    def get(self, ids, include):
        return {"ids": ids, "metadatas": [{"doc_id": i.split(":")[0].split("#")[0]} for i in ids]}


def make_retriever():
    r = Retriever.__new__(Retriever)          # skip __init__: no Chroma, no embedder
    r.store, r.collection, r._bm25 = FakeStore(), FakeCollection(), None
    r.candidate_pool, r.rrf_k = 50, 60
    return r


def test_tokenize_keeps_urdu_and_numbers():
    assert tokenize("Section 25 کی تحویل G&W") == ["section", "25", "کی", "تحویل"]   # single letters dropped


def test_bm25_ranks_exact_term_first():
    idx = BM25Index({cid: t for cid, t in DOCS.items()})
    top = idx.search("hizanat visitation", 3)
    assert top[0][0] == "SHC-L16#3"
    assert idx.search("zzzz", 3) == []


def test_rrf_rewards_agreement():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == fused["b"] > fused["c"] == fused["d"]


def test_fusion_adds_bm25_only_hit_and_keeps_dense_score():
    r = make_retriever()
    # dense put the wrong passage (Art. 63A) first and missed the judgment entirely
    dense = [RetrievedChunk("CONST:63A#1", 1, 0.52, DOCS["CONST:63A#1"], {"doc_id": "CONST"}, ["CONST:63A"]),
             RetrievedChunk("GWA-1890:25#1", 2, 0.50, DOCS["GWA-1890:25#1"], {"doc_id": "GWA-1890"}, ["GWA-1890:25"])]
    out = r.bm25_fusion("custody hizanat visitation of the minor", dense, n=10)
    ids = [c.chunk_id for c in out]
    assert "SHC-L16#3" in ids                                  # found by BM25 only
    assert ids.index("GWA-1890:25#1") < ids.index("CONST:63A#1")  # BM25 agreement lifts the custody section
    by = {c.chunk_id: c for c in out}
    assert by["CONST:63A#1"].metadata["dense_score"] == 0.52
    assert by["SHC-L16#3"].metadata["dense_score"] is None
    assert by["SHC-L16#3"].metadata["bm25_rank"] == 1
    assert by["SHC-L16#3"].text == DOCS["SHC-L16#3"]


class FakeCrossEncoder:
    """Scores a passage by how many query words it contains (stands in for bge-reranker-v2-m3)."""

    def score(self, query, passages):
        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) / (len(q) or 1) for p in passages]


class _S:
    embed_device = "cpu"


def test_rerank_resorts_and_keeps_previous_score(monkeypatch):
    import rag.retrieval.retriever as retr
    monkeypatch.setattr(retr, "get_reranker", lambda *a, **k: FakeCrossEncoder())
    r = make_retriever()
    r.s, r.rerank_cfg = _S(), {"top_n": 3}
    cands = [RetrievedChunk(cid, i, 0.9 - i / 10, DOCS[cid], {}, [cid]) for i, cid in
             enumerate(["CONST:63A#1", "GWA-1890:25#1", "GWA-1890:17#1", "SHC-L16#3"], start=1)]
    out = r.rerank("custody of ward guardian", cands)
    assert len(out) == 3                                     # only top_n re-scored and kept
    assert out[0].chunk_id == "GWA-1890:25#1"                # most query words
    assert out[-1].chunk_id == "CONST:63A#1"                 # wrong passage pushed down
    assert all(0.0 <= c.score <= 1.0 for c in out)
    assert out[-1].metadata["pre_rerank_rank"] == 1


def _c(cid, doc_type, rank):
    return RetrievedChunk(cid, rank, 1.0 - rank / 100, cid, {"doc_type": doc_type}, [cid])


def test_source_mix_reserves_statute_slots():
    r = make_retriever()
    r.mix_cfg = {"min_law": 2}
    cands = [_c(f"SHC-L{i:02d}:1#1", "judgment", i) for i in range(1, 9)]
    cands += [_c("CONST:63A#1", "constitution", 9), _c("GWA-1890:25#1", "act", 10),
              _c("GWA-1890:17#1", "act", 11), _c("GWA-1890:12#1", "act", 12)]
    out = r.source_mix(cands, 5)
    ids = [c.chunk_id for c in out]
    assert len(out) == 5
    assert ids[:3] == ["SHC-L01:1#1", "SHC-L02:1#1", "SHC-L03:1#1"]       # best judgments kept, in order
    assert ids[3:] == ["GWA-1890:25#1", "GWA-1890:17#1"]                   # two best Act sections added
    assert "CONST:63A#1" not in ids                                        # Constitution not counted as law


def test_source_mix_no_statutes_available_keeps_top_k():
    r = make_retriever()
    r.mix_cfg = {"min_law": 2}
    cands = [_c(f"SHC-L{i:02d}:1#1", "judgment", i) for i in range(1, 9)]
    assert [c.chunk_id for c in r.source_mix(cands, 5)] == [c.chunk_id for c in cands[:5]]


def test_law_lane_adds_statutes_when_main_list_has_none():
    import numpy as np

    class LawOnlyCollection(FakeCollection):
        def query(self, query_embeddings, n_results, where, include):
            assert where == {"doc_type": {"$in": ["act", "ordinance"]}}
            ids = ["GWA-1890:17#1", "GWA-1890:12#1"][:n_results]
            return {"ids": [ids], "distances": [[0.6, 0.65][: len(ids)]],
                    "metadatas": [[{"doc_type": "act"} for _ in ids]]}

    r = make_retriever()
    r.collection, r.mix_cfg, r._last_vec = LawOnlyCollection(), {"min_law": 2, "law_pool": 10}, np.zeros(3)
    main = [_c(f"SHC-L{i:02d}:1#1", "judgment", i) for i in range(1, 6)]
    out = r.source_mix(r.law_lane(main), 5)
    ids = [c.chunk_id for c in out]
    assert ids == ["SHC-L01:1#1", "SHC-L02:1#1", "SHC-L03:1#1", "GWA-1890:17#1", "GWA-1890:12#1"]
    assert out[-1].metadata["law_lane"] is True
