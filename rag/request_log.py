"""Append-only request log (logs/requests.jsonl). By default the user's question and the answer text are
NOT stored (logging.store_text: false); set it to true only for local evaluation runs. API keys never reach the log: the config snapshot
is built from a whitelist, and every string is scanned for key-like values before writing."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

from rag.config import SECRET_ENV, Settings

_KEY_PATTERNS = [re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), re.compile(r"sk-[A-Za-z0-9_\-]{16,}")]
_lock = threading.Lock()


def _redact(obj: Any, secrets: list[str]) -> Any:
    if isinstance(obj, str):
        for sec in secrets:
            if sec and len(sec) >= 8:
                obj = obj.replace(sec, "[REDACTED]")
        for p in _KEY_PATTERNS:
            obj = p.sub("[REDACTED]", obj)
        return obj
    if isinstance(obj, dict):
        return {k: (_redact(v, secrets) if "key" not in k.lower() or isinstance(v, (dict, list)) else "[REDACTED]")
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v, secrets) for v in obj]
    return obj


def _gone(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return text
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"[not stored: {len(text)} chars, sha256 {h}]"


def strip_user_text(record: dict[str, Any]) -> dict[str, Any]:
    """Remove what the user wrote (and model text that restates it) from a log record (NFR-COMP-03).
    Kept: ids, retrieval results, scores, quotes (corpus text), verification, timings, tokens, cost.
    A short hash lets the same question be matched across runs without storing it."""
    r = dict(record)
    for k in ("question", "standalone_question", "condense_error"):
        if k in r:
            r[k] = _gone(r[k])
    if isinstance(r.get("final"), dict):
        f = dict(r["final"])
        f["answer"] = _gone(f.get("answer"))
        f["claims"] = [{**c, "text": _gone(c.get("text"))} for c in (f.get("claims") or [])]
        r["final"] = f
    if isinstance(r.get("dropped_claims"), list):
        r["dropped_claims"] = [{**c, "text": _gone(c.get("text"))} for c in r["dropped_claims"]]
    if isinstance(r.get("llm_attempts"), list):
        r["llm_attempts"] = [{**a, "raw_reply": _gone(a["raw_reply"])} if "raw_reply" in a else a
                             for a in r["llm_attempts"]]
    if isinstance(r.get("traceback"), str):
        r["traceback"] = _gone(r["traceback"])
    return r


def append_request_log(s: Settings, record: dict[str, Any]) -> Path:
    path = s.path("requests_log")
    path.parent.mkdir(parents=True, exist_ok=True)
    secrets = [s.secret(k) for k in SECRET_ENV]
    if not (s.raw.get("logging", {}) or {}).get("store_text", False):
        record = strip_user_text(record)
    clean = _redact(record, secrets)
    line = json.dumps(clean, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
