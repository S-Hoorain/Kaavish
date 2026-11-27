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
