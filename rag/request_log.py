"""Append-only request log (logs/requests.jsonl). API keys never reach the log: the config snapshot
is built from a whitelist, and every string is scanned for key-like values before writing."""
from __future__ import annotations

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


def append_request_log(s: Settings, record: dict[str, Any]) -> Path:
    path = s.path("requests_log")
    path.parent.mkdir(parents=True, exist_ok=True)
    secrets = [s.secret(k) for k in SECRET_ENV]
    clean = _redact(record, secrets)
    line = json.dumps(clean, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
