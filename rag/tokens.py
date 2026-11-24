"""Token counting with the BGE-M3 tokenizer (the same tokenizer the embedder truncates with).

Counts exclude the special tokens <s> and </s>; the embedder adds two, so a chunk of N tokens
uses N + 2 positions of the 512-token window.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

TokenCounter = Callable[[str], int]


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    from transformers import AutoTokenizer  # heavy import, kept lazy

    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)


def bge_m3_counter(model_name: str = "BAAI/bge-m3") -> TokenCounter:
    tok = _tokenizer(model_name)

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


def whitespace_counter(text: str) -> int:
    """Cheap stand-in used by unit tests (1 token per whitespace-separated word)."""
    return len(text.split())
