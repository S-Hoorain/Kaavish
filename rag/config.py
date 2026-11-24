"""Configuration: config/settings.yaml + config/chunking_variants.yaml + .env.

Everything tunable lives in the YAML files; provider/model choice and secrets live in .env.
`Settings.snapshot()` returns a key-free copy that is written into every request log.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# Windows without developer mode cannot create symlinks in the HF cache; the warning is cosmetic.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Model caches live inside the project (on the same drive as the repo) unless .env sets HF_HOME.
# This must run before transformers / huggingface_hub are imported, which read HF_HOME at import time;
# rag/__init__.py imports this module first for that reason.
load_dotenv(ROOT / ".env", override=False)
if not (os.environ.get("HF_HOME") or "").strip():
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")

SECRET_ENV = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY")
LLM_PROVIDERS = ("openai", "anthropic", "openai_compatible")
REASONING_EFFORTS = ("low", "medium", "high")
DEFAULT_BASE_HOSTS = {"openai": "api.openai.com", "anthropic": "api.anthropic.com"}


@dataclass(frozen=True)
class ChunkingConfig:
    name: str
    target_min_tokens: int
    target_max_tokens: int
    hard_cap_tokens: int
    merge_below_tokens: int
    overlap_tokens: int
    header_in_embed_text: bool
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Settings:
    raw: dict[str, Any]
    chunking: ChunkingConfig
    env: dict[str, str] = field(default_factory=dict)

    # ---- paths -------------------------------------------------------------
    def path(self, key: str) -> Path:
        return ROOT / self.raw["paths"][key]

    @property
    def processed_dir(self) -> Path:
        """Per-variant output folder: data/processed/<variant>/"""
        return self.path("processed_dir") / self.chunking.name

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    # ---- embeddings ----------------------------------------------------------
    @property
    def embedding_provider(self) -> str:
        return (self.env.get("EMBEDDINGS") or "bge_m3_local").strip()

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider == "openai":
            return (self.env.get("OPENAI_EMBED_MODEL") or "").strip()
        return self.raw["embeddings"]["bge_m3_model"]

    @property
    def embed_device(self) -> str:
        return (self.env.get("EMBED_DEVICE") or self.raw["embeddings"].get("device") or "cpu").strip()

    @property
    def collection_name(self) -> str:
        return f'{self.raw["vector_store"]["collection_prefix"]}__{self.chunking.name}'

    # ---- llm -------------------------------------------------------------------
    @property
    def llm_provider(self) -> str:
        return (self.env.get("LLM_PROVIDER") or "").strip().lower()

    @property
    def llm_model(self) -> str:
        return (self.env.get("LLM_MODEL") or "").strip()

    @property
    def llm_base_url(self) -> str:
        """Only used by LLM_PROVIDER=openai_compatible."""
        return (self.env.get("LLM_BASE_URL") or "").strip()

    @property
    def llm_base_url_host(self) -> str:
        """Host part only (what gets logged): never the path, query or any credential."""
        from urllib.parse import urlparse

        if self.llm_provider == "openai_compatible":
            return urlparse(self.llm_base_url).hostname or ""
        # the OpenAI / Anthropic SDKs read OPENAI_BASE_URL / ANTHROPIC_BASE_URL themselves
        sdk_env = {"openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL"}.get(self.llm_provider)
        override = (os.environ.get(sdk_env) or "").strip() if sdk_env else ""
        if override:
            return urlparse(override).hostname or ""
        return DEFAULT_BASE_HOSTS.get(self.llm_provider, "")

    @property
    def llm_reasoning_effort(self) -> str:
        """Optional: low | medium | high. Empty = not sent."""
        return (self.env.get("LLM_REASONING_EFFORT") or "").strip().lower()

    def secret(self, name: str) -> str:
        return (self.env.get(name) or "").strip()

    # ---- snapshot for logs -----------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Configuration that produced a result (NFR-INT-06). Never contains API keys."""
        snap = {
            "llm": {
                "provider": self.llm_provider,
                "base_url_host": self.llm_base_url_host,
                "model": self.llm_model,
                "reasoning_effort": self.llm_reasoning_effort or None,
                **{k: v for k, v in self.raw["llm"].items()},
            },
            "embeddings": {
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "max_seq_length": self.raw["embeddings"]["max_seq_length"],
                "normalize": self.raw["embeddings"]["normalize"],
            },
            "vector_store": {"backend": "chroma", "collection": self.collection_name,
                             "distance": self.raw["vector_store"]["distance"]},
            "retrieval": copy.deepcopy(self.raw["retrieval"]),
            "chunking": self.chunking.as_dict(),
            "prompts": copy.deepcopy(self.raw["prompts"]),
        }
        return snap

    def snapshot_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()[:12]


def load_variants() -> dict[str, dict[str, Any]]:
    with (ROOT / "config" / "chunking_variants.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_env() -> dict[str, str]:
    load_dotenv(ROOT / ".env", override=False)
    keys = ["LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_BASE_URL", "LLM_API_KEY",
            "LLM_REASONING_EFFORT", "EMBEDDINGS", "OPENAI_EMBED_MODEL", "EMBED_DEVICE", "HF_HOME"]
    return {k: os.environ.get(k, "") for k in keys}


@lru_cache(maxsize=8)
def get_settings(variant: str | None = None) -> Settings:
    with (ROOT / "config" / "settings.yaml").open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    variants = load_variants()
    name = variant or raw["chunking"]["variant"]
    if name not in variants:
        raise KeyError(f"Unknown chunking variant {name!r}. Known: {sorted(variants)}")
    chunking = ChunkingConfig(name=name, **variants[name])
    env = _read_env()
    if env.get("HF_HOME"):
        os.environ["HF_HOME"] = env["HF_HOME"]
    return Settings(raw=raw, chunking=chunking, env=env)
