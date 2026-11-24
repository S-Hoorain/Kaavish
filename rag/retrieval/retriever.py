"""R0 retrieval: dense search only (FR-RET-02), as a pipeline of switchable stages (FR-RET-06).

    retrieve(question, k=5, filters=None) -> list[RetrievedChunk]

Stage order (each stage is a method; disabled stages are skipped and logged as "off"):

    1. normalise_query   EXTENSION POINT: Roman Urdu spelling normaliser (FR-LANG-03)
    2. dense_search      R0, implemented
    3. bm25_fusion       EXTENSION POINT: R1, BM25 + reciprocal rank fusion (FR-RET-03)
    4. metadata_filter   EXTENSION POINT: R2, jurisdiction/status/date policy (FR-RET-04)
    5. rerank            EXTENSION POINT: R3, cross-encoder over the top 30-50 (FR-RET-05)

`filters` is passed straight to Chroma as a `where` clause on scalar metadata (for example
{"doc_id": "GWA-1890"}); it is plumbing for callers and tests, not the R2 policy filter.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from rag.config import Settings, get_settings
from rag.index.embeddings import get_embedder
from rag.index.store import get_collection, load_chunk_store


@dataclass
class RetrievedChunk:
    chunk_id: str
    rank: int
    score: float                 # cosine similarity (1 - cosine distance)
    text: str                    # verbatim passage text (shown to the LLM and used for quote checks)
    metadata: dict[str, Any]
    unit_ids: list[str] = field(default_factory=list)   # parent units covered (for evaluation)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    stages: dict[str, str]


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.stages_cfg = self.s.raw["retrieval"]["stages"]
        self.collection = get_collection(self.s, check_model=True)   # refuses on model mismatch
        self.embedder = get_embedder(self.s)
        self.store = load_chunk_store(str(self.s.chunks_path))

    # ---- stages -------------------------------------------------------------------------------
    def normalise_query(self, q: str) -> str:
        # EXTENSION POINT (FR-LANG-03): map Roman Urdu spelling variants (talak -> talaq, khula/khulla, ...)
        # to one form before embedding. Not implemented in R0.
        if self.stages_cfg.get("query_normaliser"):
            raise NotImplementedError("Roman Urdu normaliser is not part of R0")
        return q

    def dense_search(self, q: str, n: int, where: dict | None) -> tuple[list[RetrievedChunk], float, float]:
        t0 = time.perf_counter()
        vec = self.embedder.embed_query(q)
        t1 = time.perf_counter()
        res = self.collection.query(query_embeddings=[vec.tolist()], n_results=n, where=where or None,
                                    include=["distances", "metadatas"])
        t2 = time.perf_counter()
        out = []
        for rank, (cid, dist, meta) in enumerate(zip(res["ids"][0], res["distances"][0], res["metadatas"][0]), start=1):
            rec = self.store.get(cid) or {}
            out.append(RetrievedChunk(chunk_id=cid, rank=rank, score=round(1.0 - float(dist), 6),
                                      text=rec.get("text", ""), metadata=meta,
                                      unit_ids=rec.get("unit_ids") or [meta.get("parent_id")]))
        return out, (t1 - t0) * 1000, (t2 - t1) * 1000

    def bm25_fusion(self, q: str, dense: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # EXTENSION POINT (R1, FR-RET-03): build a BM25 index over chunks.jsonl `embed_text`, take its
        # top-N, and merge with `dense` by reciprocal rank fusion (score = sum 1/(60 + rank)).
        raise NotImplementedError("BM25 fusion is R1, not part of R0")

    def metadata_filter(self, chunks: list[RetrievedChunk], context: dict | None) -> list[RetrievedChunk]:
        # EXTENSION POINT (R2, FR-RET-04): drop status in {repealed, overruled}, out-of-jurisdiction
        # judgments, and rank binding courts first. Metadata needed is already in Chroma
        # (status, jurisdiction, court, date, doc_type).
        raise NotImplementedError("Metadata filtering is R2, not part of R0")

    def rerank(self, q: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # EXTENSION POINT (R3, FR-RET-05): score (q, chunk.text) pairs with a multilingual
        # cross-encoder over the top 30-50 candidates and re-sort; keep dense score in metadata.
        raise NotImplementedError("Reranking is R3, not part of R0")

    # ---- pipeline -------------------------------------------------------------------------------
    def retrieve_with_trace(self, question: str, k: int = 5, filters: dict | None = None) -> RetrievalResult:
        stages = {"query_normaliser": "off", "dense": "on", "bm25_fusion": "off", "metadata_filter": "off",
                  "reranker": "off"}
        q = question
        if self.stages_cfg.get("query_normaliser"):
            q = self.normalise_query(q)
            stages["query_normaliser"] = "on"
        n_candidates = k  # R1/R3 will need a larger candidate pool (e.g. 50)
        chunks, t_embed, t_search = self.dense_search(q, n_candidates, filters)
        if self.stages_cfg.get("bm25_fusion"):
            chunks = self.bm25_fusion(q, chunks)
            stages["bm25_fusion"] = "on"
        if self.stages_cfg.get("metadata_filter"):
            chunks = self.metadata_filter(chunks, None)
            stages["metadata_filter"] = "on"
        if self.stages_cfg.get("reranker"):
            chunks = self.rerank(q, chunks)
            stages["reranker"] = "on"
        chunks = chunks[:k]
        for i, c in enumerate(chunks, start=1):
            c.rank = i
        return RetrievalResult(query=q, chunks=chunks, stages=stages,
                               timings_ms={"embed": round(t_embed, 1), "search": round(t_search, 1)})

    def retrieve(self, question: str, k: int = 5, filters: dict | None = None) -> list[RetrievedChunk]:
        return self.retrieve_with_trace(question, k, filters).chunks


_RETRIEVERS: dict[str, Retriever] = {}


def get_retriever(variant: str | None = None) -> Retriever:
    s = get_settings(variant)
    if s.chunking.name not in _RETRIEVERS:
        _RETRIEVERS[s.chunking.name] = Retriever(s)
    return _RETRIEVERS[s.chunking.name]


def retrieve(question: str, k: int = 5, filters: dict | None = None, variant: str | None = None) -> list[RetrievedChunk]:
    """Module-level convenience matching the task spec signature."""
    return get_retriever(variant).retrieve(question, k=k, filters=filters)
