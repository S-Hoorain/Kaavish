"""Check that .env has what the chosen setup needs. Never prints key values.

    python scripts/check_env.py            # checks LLM + embeddings
    python scripts/check_env.py --no-llm   # only what retrieval needs (ingest, index, retrieve, smoke test)

LLM_PROVIDER:
    openai             OPENAI_API_KEY, LLM_MODEL
    anthropic          ANTHROPIC_API_KEY, LLM_MODEL
    openai_compatible  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL   (e.g. Gemini's OpenAI-compatible endpoint)
Optional: LLM_REASONING_EFFORT = low | medium | high (openai / openai_compatible only; sent only when set)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("openai", "anthropic", "openai_compatible")
EFFORTS = ("low", "medium", "high")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip LLM variables (retrieval-only use)")
    a = ap.parse_args()
    from dotenv import load_dotenv

    env_file = ROOT / ".env"
    if not env_file.exists():
        print(f"MISSING: {env_file} does not exist. Copy .env.example to .env and fill it in.")
        return 2
    load_dotenv(env_file, override=False)
    get = lambda k: (os.environ.get(k) or "").strip()
    problems: list[str] = []
    notes: list[str] = []
    ok: list[str] = []

    def need(key: str, why: str) -> None:
        (ok if get(key) else problems).append(f"{key} {'is set' if get(key) else f'is missing ({why})'}")

    emb = get("EMBEDDINGS") or "bge_m3_local"
    if emb not in ("bge_m3_local", "openai"):
        problems.append(f"EMBEDDINGS={emb!r} must be bge_m3_local or openai")
    elif emb == "openai":
        need("OPENAI_API_KEY", "needed because EMBEDDINGS=openai")
        need("OPENAI_EMBED_MODEL", "needed because EMBEDDINGS=openai")
    else:
        ok.append("EMBEDDINGS=bge_m3_local (no key needed)")
    hf = get("HF_HOME")
    ok.append(f"model cache (HF_HOME): {hf or ROOT / '.cache' / 'huggingface'}{'' if hf else ' (default, inside the repo)'}")

    if not a.no_llm:
        prov = get("LLM_PROVIDER").lower()
        if prov not in PROVIDERS:
            problems.append(f"LLM_PROVIDER={prov!r} must be one of {', '.join(PROVIDERS)}")
        else:
            ok.append(f"LLM_PROVIDER={prov}")
            if prov == "openai":
                need("OPENAI_API_KEY", "needed because LLM_PROVIDER=openai")
                if get("OPENAI_BASE_URL"):
                    notes.append(f"OPENAI_BASE_URL is set: the OpenAI SDK will send requests to "
                                 f"{urlparse(get('OPENAI_BASE_URL')).hostname}, not api.openai.com. For a non-OpenAI "
                                 f"server use LLM_PROVIDER=openai_compatible with LLM_BASE_URL / LLM_API_KEY.")
            elif prov == "anthropic":
                need("ANTHROPIC_API_KEY", "needed because LLM_PROVIDER=anthropic")
                if get("LLM_REASONING_EFFORT"):
                    problems.append("LLM_REASONING_EFFORT is only used with openai / openai_compatible; leave it empty")
            else:  # openai_compatible
                url = get("LLM_BASE_URL")
                if not url:
                    hint = ""
                    if get("OPENAI_BASE_URL"):
                        hint = (" -- OPENAI_BASE_URL is set instead: rename it to LLM_BASE_URL, and put the key for "
                                "that server in LLM_API_KEY (openai_compatible does not read OPENAI_* variables)")
                    problems.append(f"LLM_BASE_URL is missing (needed because LLM_PROVIDER=openai_compatible){hint}")
                elif not url.lower().startswith(("http://", "https://")):
                    problems.append("LLM_BASE_URL must start with https:// (or http://)")
                else:
                    ok.append(f"LLM_BASE_URL host = {urlparse(url).hostname}")
                need("LLM_API_KEY", "needed because LLM_PROVIDER=openai_compatible")
        need("LLM_MODEL", "model id of the chosen provider")
        if get("LLM_MODEL"):
            ok[-1] = f"LLM_MODEL = {get('LLM_MODEL')}"
        eff = get("LLM_REASONING_EFFORT").lower()
        if eff and eff not in EFFORTS:
            problems.append(f"LLM_REASONING_EFFORT={eff!r} must be one of {', '.join(EFFORTS)} or empty")
        elif eff:
            ok.append(f"LLM_REASONING_EFFORT = {eff}")
        else:
            ok.append("LLM_REASONING_EFFORT not set (not sent)")

    for line in ok:
        print("ok      ", line)
    for line in notes:
        print("note    ", line)
    for line in problems:
        print("MISSING ", line)
    if problems:
        print("\nFix the lines above in .env (see .env.example), then run this again.")
        return 1
    print("\nEnvironment is complete for this setup. Try one real call: python scripts/test_llm.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
