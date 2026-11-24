"""Embedding providers behind one interface, with an on-disk cache keyed by model + text hash.

* bge_m3_local: BAAI/bge-m3 via sentence-transformers (no API key), normalised embeddings,
  max sequence length 512, configurable batch size and device.
* openai: OpenAI embeddings API (OPENAI_API_KEY + OPENAI_EMBED_MODEL).
"""
from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from rag.config import Settings


class Embedder(ABC):
    provider: str
    model_name: str

    @abstractmethod
    def _embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


class BgeM3LocalEmbedder(Embedder):
    provider = "bge_m3_local"

    def __init__(self, model_name: str, max_seq_length: int = 512, batch_size: int = 8,
                 normalize: bool = True, device: str = "cpu", use_safetensors: bool | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import, lazy

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        kwargs = {} if use_safetensors is None else {"model_kwargs": {"use_safetensors": use_safetensors}}
        # Prefer the local cache: when the model is already downloaded, loading online can still make
        # transformers fetch a second 2.3 GB safetensors copy of the same weights from the Hub.
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True, **kwargs)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.model.max_seq_length = max_seq_length

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize,
                                            show_progress_bar=len(texts) > 64, convert_to_numpy=True),
                          dtype=np.float32)


class OpenAIEmbedder(Embedder):
    provider = "openai"

    def __init__(self, model_name: str, api_key: str, batch_size: int = 64):
        from openai import OpenAI

        if not model_name:
            raise ValueError("EMBEDDINGS=openai needs OPENAI_EMBED_MODEL in .env")
        if not api_key:
            raise ValueError("EMBEDDINGS=openai needs OPENAI_API_KEY in .env")
        self.model_name = model_name
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key)

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i + self.batch_size])
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


class EmbeddingCache:
    """One .npy file per (model, text) pair under data/cache/embeddings/<model-slug>/<aa>/<hash>.npy"""

    def __init__(self, root: Path, model_name: str):
        self.dir = root / re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)

    @staticmethod
    def key(model_name: str, text: str) -> str:
        return hashlib.sha256(f"{model_name}\n{text}".encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.npy"

    def get(self, key: str) -> np.ndarray | None:
        p = self._path(key)
        return np.load(p) if p.exists() else None

    def put(self, key: str, vec: np.ndarray) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, vec.astype(np.float32))


def embed_with_cache(embedder: Embedder, cache: EmbeddingCache, texts: list[str]) -> tuple[np.ndarray, dict]:
    keys = [cache.key(embedder.model_name, t) for t in texts]
    vecs: list[np.ndarray | None] = [cache.get(k) for k in keys]
    todo = [i for i, v in enumerate(vecs) if v is None]
    t0 = time.time()
    if todo:
        new = embedder._embed([texts[i] for i in todo])
        for i, v in zip(todo, new):
            cache.put(keys[i], v)
            vecs[i] = v
    stats = {"total": len(texts), "cached": len(texts) - len(todo), "computed": len(todo),
             "seconds": round(time.time() - t0, 1)}
    return np.vstack(vecs), stats


_EMBEDDERS: dict[tuple, Embedder] = {}


def get_embedder(s: Settings) -> Embedder:
    e = s.raw["embeddings"]
    key = (s.embedding_provider, s.embedding_model, s.embed_device)
    if key not in _EMBEDDERS:
        if s.embedding_provider == "bge_m3_local":
            _EMBEDDERS[key] = BgeM3LocalEmbedder(s.embedding_model, e["max_seq_length"], e["batch_size"],
                                                 e["normalize"], s.embed_device, e.get("use_safetensors"))
        elif s.embedding_provider == "openai":
            _EMBEDDERS[key] = OpenAIEmbedder(s.embedding_model, s.secret("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unknown EMBEDDINGS={s.embedding_provider!r}; use bge_m3_local or openai")
    return _EMBEDDERS[key]
