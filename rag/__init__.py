"""Law or Fiqh? basic RAG baseline.

Importing the package loads rag.config first, so HF_HOME (model cache location) is set before any
Hugging Face library is imported.
"""
from rag import config as _config  # noqa: F401
