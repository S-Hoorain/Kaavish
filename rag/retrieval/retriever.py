"""R0 retrieval: dense search only (FR-RET-02), as a pipeline of switchable stages (FR-RET-06).

    retrieve(question, k=5, filters=None) -> list[RetrievedChunk]

Stage order (each stage is a method; disabled stages are skipped and logged as "off"):

    1. normalise_query   EXTENSION POINT: Roman Urdu spelling normaliser (FR-LANG-03)
    2. dense_search      R0, implemented
    3. bm25_fusion       R1, implemented: BM25 + reciprocal rank fusion (FR-RET-03), see bm25.py
    4. metadata_filter   EXTENSION POINT: R2, jurisdiction/status/date policy (FR-RET-04)
    5. rerank            R3, implemented: cross-encoder over the top 30 (FR-RET-05), see reranker.py

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
from rag.retrieval.bm25 import BM25Index, rrf
from rag.retrieval.reranker import get_reranker


@dataclass
class RetrievedChunk:
    chunk_id: str
    rank: int
    score: float                 # cosine similarity; RRF score if bm25_fusion; reranker score (0..1) if reranker
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
        self._bm25: BM25Index | None = None
        rcfg = self.s.raw["retrieval"]
        self.candidate_pool = int(rcfg.get("candidate_pool", 50))   # used when fusion or rerank is on
        self.rrf_k = int(rcfg.get("rrf_k", 60))
        self.rerank_cfg = rcfg.get("rerank", {}) or {}
        self.mix_cfg = rcfg.get("source_mix", {}) or {}

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

    def bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            docs = {cid: (rec.get("embed_text") or rec.get("text") or "") for cid, rec in self.store.by_id.items()}
            self._bm25 = BM25Index(docs)
        return self._bm25

    def bm25_fusion(self, q: str, dense: list[RetrievedChunk], n: int | None = None) -> list[RetrievedChunk]:
        """R1 (FR-RET-03): BM25 over chunks.jsonl `embed_text`, merged with the dense list by
        reciprocal rank fusion (score = sum 1/(rrf_k + rank)). `score` becomes the RRF score; the
        cosine similarity and the BM25 rank are kept in metadata (dense_score, dense_rank, bm25_rank)."""
        n = n or self.candidate_pool
        bm25_hits = self.bm25_index().search(q, n)
        dense_by_id = {c.chunk_id: c for c in dense}
        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}
        fused = rrf([[c.chunk_id for c in dense], [cid for cid, _ in bm25_hits]], k=self.rrf_k)

        # chunks found only by BM25 need their Chroma metadata
        missing = [cid for cid, _ in fused if cid not in dense_by_id]
        meta_by_id: dict[str, dict] = {}
        if missing:
            got = self.collection.get(ids=missing, include=["metadatas"])
            meta_by_id = dict(zip(got["ids"], got["metadatas"]))

        out = []
        for cid, score in fused:
            if cid in dense_by_id:
                c = dense_by_id[cid]
                meta = dict(c.metadata or {})
                meta.update(dense_score=c.score, dense_rank=c.rank)
                text, units = c.text, c.unit_ids
            else:
                rec = self.store.get(cid) or {}
                meta = dict(meta_by_id.get(cid) or {})
                meta.update(dense_score=None, dense_rank=None)
                text, units = rec.get("text", ""), (rec.get("unit_ids") or [meta.get("parent_id")])
            meta["bm25_rank"] = bm25_rank.get(cid)
            out.append(RetrievedChunk(chunk_id=cid, rank=0, score=round(score, 6), text=text,
                                      metadata=meta, unit_ids=units))
        return out

    def metadata_filter(self, chunks: list[RetrievedChunk], context: dict | None) -> list[RetrievedChunk]:
        # EXTENSION POINT (R2, FR-RET-04): drop status in {repealed, overruled}, out-of-jurisdiction
        # judgments, and rank binding courts first. Metadata needed is already in Chroma
        # (status, jurisdiction, court, date, doc_type).
        raise NotImplementedError("Metadata filtering is R2, not part of R0")

    def rerank(self, q: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """R3 (FR-RET-05): re-score the top `rerank.top_n` candidates with a multilingual cross-encoder
        and re-sort. `score` becomes the reranker score (0..1); the previous score and rank are kept in
        metadata (pre_rerank_score, pre_rerank_rank). Candidates beyond top_n are dropped."""
        cfg = self.rerank_cfg
        top = chunks[: int(cfg.get("top_n", 30))]
        model = get_reranker(cfg.get("model", "BAAI/bge-reranker-v2-m3"), int(cfg.get("max_length", 512)),
                             int(cfg.get("batch_size", 8)), self.s.embed_device)
        scores = model.score(q, [c.text for c in top])
        for c, sc in zip(top, scores):
            meta = dict(c.metadata or {})
            meta.update(pre_rerank_score=c.score, pre_rerank_rank=c.rank)
            c.metadata, c.score = meta, round(sc, 6)
        return sorted(top, key=lambda c: c.score, reverse=True)

    def _is_law(self, c: RetrievedChunk) -> bool:
        types = self.mix_cfg.get("law_doc_types", ["act", "ordinance"])
        return (c.metadata or {}).get("doc_type") in types

    def source_mix(self, chunks: list[RetrievedChunk], n: int) -> list[RetrievedChunk]:
        """Keep n chunks, reserving up to `min_law` slots for statute passages (Acts/Ordinances), so judgments
        that discuss a section cannot crowd out the section itself. Order of the input is kept."""
        m = int(self.mix_cfg.get("min_law", 2))
        law = [c for c in chunks if self._is_law(c)][:m]
        chosen_ids = {c.chunk_id for c in law}
        rest = [c for c in chunks if c.chunk_id not in chosen_ids][: n - len(law)]
        chosen_ids |= {c.chunk_id for c in rest}
        return [c for c in chunks if c.chunk_id in chosen_ids]

    # ---- pipeline -------------------------------------------------------------------------------
    def retrieve_with_trace(self, question: str, k: int = 5, filters: dict | None = None) -> RetrievalResult:
        stages = {"query_normaliser": "off", "dense": "on", "bm25_fusion": "off", "metadata_filter": "off",
                  "reranker": "off", "source_mix": "off"}
        mixing = bool(self.stages_cfg.get("source_mix"))
        q = question
        if self.stages_cfg.get("query_normaliser"):
            q = self.normalise_query(q)
            stages["query_normaliser"] = "on"
        widen = self.stages_cfg.get("bm25_fusion") or self.stages_cfg.get("reranker") or mixing
        n_candidates = max(k, self.candidate_pool) if widen else k
        chunks, t_embed, t_search = self.dense_search(q, n_candidates, filters)
        timings = {"embed": round(t_embed, 1), "search": round(t_search, 1)}
        if self.stages_cfg.get("bm25_fusion"):
            if filters:
                raise NotImplementedError("bm25_fusion does not apply `filters` yet; turn one of them off")
            t0 = time.perf_counter()
            chunks = self.bm25_fusion(q, chunks, n_candidates)
            timings["bm25_fusion"] = round((time.perf_counter() - t0) * 1000, 1)
            stages["bm25_fusion"] = "on"
        if self.stages_cfg.get("metadata_filter"):
            chunks = self.metadata_filter(chunks, None)
            stages["metadata_filter"] = "on"
        if self.stages_cfg.get("reranker"):
            for i, c in enumerate(chunks, start=1):
                c.rank = i
            if mixing:   # make sure the reranker also sees statute candidates, not only the top judgments
                top_n = int(self.rerank_cfg.get("top_n", 30))
                m = int(self.mix_cfg.get("min_law", 2))
                law = [c for c in chunks if self._is_law(c)][: 2 * m]
                ids = {c.chunk_id for c in law}
                others = [c for c in chunks if c.chunk_id not in ids][: top_n - len(law)]
                ids |= {c.chunk_id for c in others}
                chunks = [c for c in chunks if c.chunk_id in ids]
            t0 = time.perf_counter()
            chunks = self.rerank(q, chunks)
            timings["reranker"] = round((time.perf_counter() - t0) * 1000, 1)
            stages["reranker"] = "on"
        if mixing:
            chunks = self.source_mix(chunks, k)
            stages["source_mix"] = "on"
        chunks = chunks[:k]
        for i, c in enumerate(chunks, start=1):
            c.rank = i
        return RetrievalResult(query=q, chunks=chunks, stages=stages, timings_ms=timings)

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
