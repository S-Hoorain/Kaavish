"""Chroma vector store (persistent, cosine) and the in-memory chunk store (full records)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag.config import Settings


class EmbeddingModelMismatch(RuntimeError):
    pass


def chroma_client(s: Settings):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    path = s.path("chroma_dir")
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


def get_collection(s: Settings, check_model: bool = True):
    client = chroma_client(s)
    try:
        col = client.get_collection(s.collection_name)
    except Exception as e:  # collection does not exist
        raise RuntimeError(f"Chroma collection {s.collection_name!r} not found in {s.path('chroma_dir')}. "
                           f"Build it first: python -m rag.cli index --variant {s.chunking.name}") from e
    if check_model:
        meta = col.metadata or {}
        stored = (meta.get("embedding_provider"), meta.get("embedding_model"))
        wanted = (s.embedding_provider, s.embedding_model)
        if stored != wanted:
            raise EmbeddingModelMismatch(
                f"Collection {s.collection_name!r} was built with {stored[0]}:{stored[1]} but the configuration "
                f"says {wanted[0]}:{wanted[1]}. Refusing to query: rebuild the index or change EMBEDDINGS.")
    return col


class ChunkStore:
    """All chunk records of a variant, loaded from chunks.jsonl (lists included)."""

    def __init__(self, path: Path):
        self.path = path
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_parent: dict[str, list[str]] = {}
        self.order: list[str] = []
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run ingestion first: python -m rag.cli ingest")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                self.by_id[c["chunk_id"]] = c
                self.order.append(c["chunk_id"])
                for u in c.get("unit_ids") or [c["parent_id"]]:
                    self.by_parent.setdefault(u, []).append(c["chunk_id"])
        self.corpus_version = next(iter(self.by_id.values()))["corpus_version"] if self.by_id else None

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self.by_id.get(chunk_id)

    def neighbours(self, chunk_id: str) -> tuple[str | None, str | None]:
        """Previous and next chunk of the same document (for the source panel, FR-SRC-03)."""
        if chunk_id not in self.by_id:
            return None, None
        i = self.order.index(chunk_id)
        doc = self.by_id[chunk_id]["doc_id"]
        prev_id = self.order[i - 1] if i > 0 and self.by_id[self.order[i - 1]]["doc_id"] == doc else None
        next_id = self.order[i + 1] if i + 1 < len(self.order) and self.by_id[self.order[i + 1]]["doc_id"] == doc else None
        return prev_id, next_id

    def __len__(self) -> int:
        return len(self.by_id)


@lru_cache(maxsize=8)
def load_chunk_store(path_str: str) -> ChunkStore:
    return ChunkStore(Path(path_str))
