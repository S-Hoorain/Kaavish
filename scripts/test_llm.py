"""Make ONE real call to the LLM configured in .env and print the reply.

    python scripts/test_llm.py
    python scripts/test_llm.py --prompt "Say hello in Roman Urdu as JSON: {\"reply\": \"...\"}"

Prints the provider, base URL host, model id, reasoning effort, latency, token counts, finish reason and
the raw reply, and whether the reply parses as JSON. Never prints the API key. Costs one small request.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_PROMPT = ('Reply with exactly this JSON object and nothing else: '
                  '{"ok": true, "message": "connection works"}')


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--system", default="You are a test endpoint. Follow the user's formatting instruction exactly.")
    a = ap.parse_args()

    from rag.config import SECRET_ENV, get_settings
    from rag.generation.llm import LLMConfigError, get_llm_client

    s = get_settings()
    secrets = [s.secret(k) for k in SECRET_ENV if s.secret(k)]

    def safe(text: str) -> str:
        for sec in secrets:
            text = text.replace(sec, "[REDACTED]")
        return text

    print(f"provider:         {s.llm_provider or '(not set)'}")
    print(f"base URL host:    {s.llm_base_url_host or '(not set)'}")
    print(f"model:            {s.llm_model or '(not set)'}")
    print(f"reasoning effort: {s.llm_reasoning_effort or '(not sent)'}")
    try:
        client = get_llm_client(s)
    except LLMConfigError as e:
        print(f"\nCONFIG ERROR: {e}\nRun: python scripts/check_env.py")
        return 2
    try:
        r = client.complete(a.system, a.prompt, json_mode=True)
    except Exception as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "body", None)
        print(f"\nAPI ERROR ({type(e).__name__}{', HTTP ' + str(status) if status else ''}): {safe(str(e))[:1500]}")
        if body:
            print(f"error body: {safe(str(body))[:1500]}")
        if status in (429, 500, 502, 503, 504):
            print("This is a server-side / rate-limit status (the SDK already retried); try again later.")
        elif status in (401, 403):
            print("Authentication failed: check the API key for this provider in .env.")
        elif status == 404:
            print("Not found: check LLM_MODEL (exact model id) and LLM_BASE_URL.")
        return 1
    print(f"latency:          {r.latency_ms} ms")
    print(f"tokens:           input={r.input_tokens} output={r.output_tokens}")
    print(f"finish reason:    {r.finish_reason}")
    if r.notes:
        print(f"notes:            {r.notes}")
    print(f"model (reply):    {r.model}")
    print("\n--- reply ---")
    print(safe(r.text))
    print("--- end ---")
    try:
        json.loads(r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        print("\nreply parses as JSON: yes")
    except json.JSONDecodeError:
        print("\nreply parses as JSON: no (the pipeline retries up to 2 times with a correction note)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
