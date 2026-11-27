"""Cross-encoder reranker (R3, FR-RET-05).

A cross-encoder reads the question and one passage together and outputs a relevance score.
Slower than embeddings (one model pass per candidate), so it only re-scores the top candidates
from dense (+ BM25) search. Default model: BAAI/bge-reranker-v2-m3 (multilingual, ~2.3 GB, local).

Scores are returned in 0..1 (sigmoid of the model's logit), so a single threshold can later decide
"not enough evidence" (FR-GEN insufficient-evidence check).
"""
from __future__ import annotations

import math
from functools import lru_cache


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 8, device: str = "cpu"):
        from sentence_transformers import CrossEncoder  # heavy import, lazy

        self.model_name, self.batch_size = model_name, batch_size
        # Prefer the local cache (same pattern as the embedder): no network call once downloaded.
        try:
            self.model = CrossEncoder(model_name, max_length=max_length, device=device, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                 show_progress_bar=False)
        vals = [float(x) for x in raw]
        # Some sentence-transformers versions return logits, others already apply a sigmoid.
        if any(v < 0.0 or v > 1.0 for v in vals):
            vals = [1.0 / (1.0 + math.exp(-v)) for v in vals]
        return vals


@lru_cache(maxsize=2)
def get_reranker(model_name: str, max_length: int, batch_size: int, device: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name, max_length, batch_size, device)
