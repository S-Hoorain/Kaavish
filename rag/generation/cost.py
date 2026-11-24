"""Cost estimate from config/pricing.yaml. No price is invented: a missing or null price gives
cost_usd = None with a note."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from rag.config import ROOT


@lru_cache(maxsize=1)
def _pricing() -> dict[str, Any]:
    p = ROOT / "config" / "pricing.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def estimate_cost(provider: str, model: str, input_tokens: int | None, output_tokens: int | None) -> dict[str, Any]:
    entry = ((_pricing().get("models") or {}).get(f"{provider}/{model}")) or {}
    pin, pout = entry.get("input_per_1m"), entry.get("output_per_1m")
    if pin is None or pout is None:
        return {"cost_usd": None, "cost_note": f"price not configured for {provider}/{model} in config/pricing.yaml"}
    if input_tokens is None or output_tokens is None:
        return {"cost_usd": None, "cost_note": "token counts not reported by the provider"}
    cost = input_tokens / 1e6 * float(pin) + output_tokens / 1e6 * float(pout)
    return {"cost_usd": round(cost, 6), "cost_note": f"pricing.yaml checked_on={entry.get('checked_on') or 'unknown'}"}
