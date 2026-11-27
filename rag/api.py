"""FastAPI service.

    uvicorn rag.api:app --port 8000

GET  /health               service, corpus and configuration status
POST /ask                  {"question": str, "mode": "c0"|"c1", "k": int, "history": [{"role", "content"}]}
GET  /chunk/{chunk_id}     full passage + metadata for a source panel
                           (chunk ids contain '#': URL-encode it as %23, e.g. /chunk/GWA-1890:29%231;
                            /chunk?chunk_id=... works too)
"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.config import get_settings
from rag.generation.llm import LLMConfigError

app = FastAPI(title="Law or Fiqh? - basic RAG baseline (R0/C1) and C0 control", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=6000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="FR-QA-01: up to 1,000 characters")
    mode: Literal["c0", "c1"] = "c1"
    k: int = Field(5, ge=1, le=20)
    history: list[Turn] = Field(default_factory=list, max_length=20,
                                description="earlier turns, oldest first; used when answering.history.enabled")


@app.get("/health")
def health():
    s = get_settings()
    out = {"status": "ok", "chunking_variant": s.chunking.name, "collection": s.collection_name,
           "embedding": f"{s.embedding_provider}:{s.embedding_model}",
           "llm_configured": bool(s.llm_provider and s.llm_model), "llm": f"{s.llm_provider}:{s.llm_model}",
           "llm_base_url_host": s.llm_base_url_host, "llm_reasoning_effort": s.llm_reasoning_effort or None}
    try:
        from rag.index.store import get_collection, load_chunk_store

        store = load_chunk_store(str(s.chunks_path))
        out["corpus_version"] = store.corpus_version
        out["chunks"] = len(store)
        col = get_collection(s, check_model=True)
        out["indexed_vectors"] = col.count()
        out["index_corpus_version"] = (col.metadata or {}).get("corpus_version")
        if out["index_corpus_version"] != store.corpus_version:
            out["status"] = "degraded"
            out["warning"] = "index corpus_version differs from chunks.jsonl; run: python -m rag.cli index"
    except Exception as e:
        out["status"] = "degraded"
        out["warning"] = str(e)
    return out


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    from rag.pipeline import ask

    try:
        extra = {"history": [t.model_dump() for t in req.history]} if req.history else {}
        r = ask(req.question, mode=req.mode, k=req.k, **extra)
    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=f"LLM not configured: {e}")
    except RuntimeError as e:
        cause = e.__cause__
        if isinstance(cause, LLMConfigError):
            raise HTTPException(status_code=503, detail=f"LLM not configured: {cause}")
        raise HTTPException(status_code=500, detail=str(e))
    keep = ("request_id", "mode", "question", "k", "status", "error", "final", "verification", "retrieved",
            "latency_ms", "tokens", "cost", "corpus_version", "prompt_version", "chunking_variant",
            "standalone_question", "evidence_gate", "dropped_claims", "verification_first_pass")
    return {k: r.get(k) for k in keep}


def _chunk_payload(chunk_id: str):
    from rag.index.store import load_chunk_store

    s = get_settings()
    store = load_chunk_store(str(s.chunks_path))
    rec = store.get(chunk_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"unknown chunk_id {chunk_id!r}")
    prev_id, next_id = store.neighbours(chunk_id)
    return {"chunk": rec, "prev_chunk_id": prev_id, "next_chunk_id": next_id,
            "notice": "Legal information, not legal advice."}


@app.get("/chunk/{chunk_id:path}")
def chunk_endpoint(chunk_id: str):
    return _chunk_payload(chunk_id)


@app.get("/chunk")
def chunk_query(chunk_id: str = Query(...)):
    return _chunk_payload(chunk_id)
