"""Embed validated chunks and write them to Chroma.

    python -m rag.cli index [--variant default]

The collection is rebuilt when the corpus_version or the embedding model changes; embeddings
themselves come from the on-disk cache when the same text was embedded before.
"""
from __future__ import annotations

import json
import time
from typing import Any

from rag.config import ROOT, Settings, get_settings
from rag.index.embeddings import EmbeddingCache, embed_with_cache, get_embedder
from rag.index.store import chroma_client
from rag.schema import chroma_metadata


def build_index(variant: str | None = None, verbose: bool = True) -> dict[str, Any]:
    s: Settings = get_settings(variant)
    log = print if verbose else (lambda *a, **k: None)
    man_path = s.processed_dir / "corpus_manifest.json"
    if not s.chunks_path.exists() or not man_path.exists():
        raise FileNotFoundError(f"No chunks for variant {s.chunking.name!r}. Run: python -m rag.cli ingest --variant {s.chunking.name}")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    corpus_version = manifest["corpus_version"]
    chunks = [json.loads(l) for l in s.chunks_path.open(encoding="utf-8") if l.strip()]

    client = chroma_client(s)
    meta_wanted = {"hnsw:space": s.raw["vector_store"]["distance"], "embedding_provider": s.embedding_provider,
                   "embedding_model": s.embedding_model, "corpus_version": corpus_version,
                   "chunking_variant": s.chunking.name,
                   "max_seq_length": s.raw["embeddings"]["max_seq_length"]}
    try:
        existing = client.get_collection(s.collection_name)
        em = existing.metadata or {}
        same = all(em.get(k) == meta_wanted[k] for k in ("embedding_provider", "embedding_model", "corpus_version"))
        if same and existing.count() == len(chunks):
            log(f"[index] {s.collection_name} is up to date ({existing.count()} chunks, corpus {corpus_version})")
            return {"collection": s.collection_name, "count": existing.count(), "rebuilt": False}
        client.delete_collection(s.collection_name)
        log(f"[index] rebuilding {s.collection_name} (corpus or model changed)")
    except Exception:
        pass
    col = client.create_collection(s.collection_name, metadata=meta_wanted)

    t0 = time.time()
    log(f"[index] embedding {len(chunks)} chunks with {s.embedding_provider}:{s.embedding_model} ...")
    embedder = get_embedder(s)
    cache = EmbeddingCache(s.path("embedding_cache"), s.embedding_model)
    texts = [c["embed_text"] for c in chunks]
    vecs, stats = embed_with_cache(embedder, cache, texts)
    log(f"[index] embeddings: {stats}")
    batch = 256
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(ids=[c["chunk_id"] for c in part],
                embeddings=vecs[i:i + batch].tolist(),
                documents=[c["embed_text"] for c in part],
                metadatas=[chroma_metadata(c) for c in part])
    info = {"collection": s.collection_name, "count": col.count(), "rebuilt": True,
            "corpus_version": corpus_version, "embedding": stats, "seconds": round(time.time() - t0, 1)}
    (s.processed_dir / "index_info.json").write_text(json.dumps({**info, **meta_wanted}, indent=2), encoding="utf-8")
    log(f"[index] wrote {col.count()} vectors to {s.path('chroma_dir').relative_to(ROOT)}/{s.collection_name}")
    return info
