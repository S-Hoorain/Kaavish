"""Load versioned prompt templates and build C0 / C1 messages."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from rag.config import Settings


@dataclass(frozen=True)
class PromptSet:
    version: str
    system: str
    user_c1: str
    user_c0: str
    sha256: str


@lru_cache(maxsize=4)
def _load(prompts_dir: str, version: str) -> PromptSet:
    d = Path(prompts_dir)
    files = [d / f"answer_system.{version}.md", d / f"answer_user_c1.{version}.md", d / f"answer_user_c0.{version}.md"]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Prompt file missing: {f}")
    texts = [f.read_text(encoding="utf-8") for f in files]
    sha = hashlib.sha256("\n\x00\n".join(texts).encode("utf-8")).hexdigest()[:16]
    return PromptSet(version=version, system=texts[0].strip(), user_c1=texts[1].strip(), user_c0=texts[2].strip(), sha256=sha)


def load_prompts(s: Settings) -> PromptSet:
    return _load(str(s.path("prompts_dir")), s.raw["prompts"]["version"])


def source_label(meta: dict[str, Any]) -> str:
    """Human-readable source for a passage tag, from chunk metadata."""
    if meta.get("doc_type") == "judgment":
        para = meta.get("paragraph")
        where = "case caption" if meta.get("unit_type") == "caption" else f"para {para}"
        return f'{meta.get("title") or ""}, {meta.get("case_no") or ""}, {meta.get("court") or ""}, {meta.get("date") or ""}, {where}'
    unit = "Article" if meta.get("doc_type") == "constitution" else "Section"
    num = meta.get("article") or meta.get("section")
    return f'{meta.get("act_title")}, {unit} {num}: {meta.get("title") or ""}'.strip()


def _attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def render_passages(chunks: Sequence[Any]) -> str:
    out = []
    for c in chunks:
        meta = c.metadata if hasattr(c, "metadata") else c["metadata"]
        cid = c.chunk_id if hasattr(c, "chunk_id") else c["chunk_id"]
        text = c.text if hasattr(c, "text") else c["text"]
        # passage text is DATA; neutralise anything that looks like our own delimiter tags
        safe = re.sub(r"</?\s*(passages?|question)\b", lambda m: m.group(0).replace("<", "‹"), text, flags=re.I)
        out.append(f'<passage chunk_id="{_attr(cid)}" source="{_attr(source_label(meta))}">\n{safe}\n</passage>')
    return "\n".join(out)


def build_messages(prompts: PromptSet, mode: str, question: str, chunks: Sequence[Any] = ()) -> tuple[str, str]:
    q = question.replace("</question>", "‹/question>")
    if mode == "c1":
        user = prompts.user_c1.replace("{question}", q).replace("{passages}", render_passages(chunks))
    elif mode == "c0":
        user = prompts.user_c0.replace("{question}", q)
    else:
        raise ValueError(f"mode must be 'c0' or 'c1', got {mode!r}")
    return prompts.system, user
