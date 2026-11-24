"""OpenAI and Anthropic clients with the SDK call mocked (no network, no key): request shape,
usage parsing, and the fallback when a model rejects `temperature`."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from rag.generation.llm import AnthropicClient, LLMConfigError, OpenAIClient

KW = dict(temperature=0, max_output_tokens=100, max_retries=0, timeout=5)


def test_openai_request_shape_and_usage(monkeypatch):
    c = OpenAIClient("some-model", "sk-test-0000000000000000", **KW)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return NS(id="r1", model="some-model", choices=[NS(message=NS(content='{"a":1}'))],
                  usage=NS(prompt_tokens=11, completion_tokens=7))

    monkeypatch.setattr(c.client.chat.completions, "create", create)
    r = c.complete("SYS", "USER")
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["temperature"] == 0 and seen["response_format"] == {"type": "json_object"}
    assert (r.text, r.input_tokens, r.output_tokens, r.provider) == ('{"a":1}', 11, 7, "openai")


def test_anthropic_request_shape_and_usage(monkeypatch):
    c = AnthropicClient("some-claude-model", "sk-ant-test-000000000000", **KW)
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        return NS(id="m1", model="some-claude-model", content=[NS(type="text", text='{"b":2}')],
                  usage=NS(input_tokens=13, output_tokens=5))

    monkeypatch.setattr(c.client.messages, "create", create)
    r = c.complete("SYS", "USER")
    assert seen["system"] == "SYS" and seen["messages"] == [{"role": "user", "content": "USER"}]
    assert seen["temperature"] == 0 and seen["max_tokens"] == 100
    assert (r.text, r.input_tokens, r.output_tokens, r.provider) == ('{"b":2}', 13, 5, "anthropic")


def test_temperature_rejected_is_retried_without_it(monkeypatch):
    c = AnthropicClient("m", "sk-ant-test-000000000000", **KW)
    calls = []

    def create(**kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise ValueError("temperature is not supported for this model")
        return NS(id="m2", model="m", content=[NS(type="text", text="{}")], usage=NS(input_tokens=1, output_tokens=1))

    monkeypatch.setattr(c.client.messages, "create", create)
    r = c.complete("S", "U")
    assert len(calls) == 2 and "temperature" not in calls[1]
    assert r.notes and "temperature" in r.notes[0]


def test_missing_model_or_key_is_a_config_error():
    with pytest.raises(LLMConfigError):
        OpenAIClient("", "sk-x", **KW)
    with pytest.raises(LLMConfigError):
        AnthropicClient("m", "", **KW)


# ---- openai_compatible ---------------------------------------------------------------------------
import copy  # noqa: E402

from rag.config import Settings, get_settings  # noqa: E402
from rag.generation.llm import OpenAICompatibleClient, get_llm_client  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _settings(**env) -> Settings:
    base = get_settings()
    return Settings(raw=copy.deepcopy(base.raw), chunking=base.chunking, env={**base.env, **env})


def _fake_create(seen, finish="stop"):
    def create(**kwargs):
        seen.append(dict(kwargs))
        return NS(id="c1", model="m-compat", usage=NS(prompt_tokens=21, completion_tokens=9),
                  choices=[NS(message=NS(content='{"answer": "x"}'), finish_reason=finish)])
    return create


def test_openai_compatible_uses_base_url_no_response_format(monkeypatch):
    s = _settings(LLM_PROVIDER="openai_compatible", LLM_BASE_URL=GEMINI_URL, LLM_API_KEY="test-key-123456789",
                  LLM_MODEL="m-compat", LLM_REASONING_EFFORT="")
    c = get_llm_client(s)
    assert isinstance(c, OpenAICompatibleClient)
    assert str(c.client.base_url).startswith(GEMINI_URL.rstrip("/"))
    seen = []
    monkeypatch.setattr(c.client.chat.completions, "create", _fake_create(seen))
    r = c.complete("SYS", "USER", json_mode=True)
    sent = seen[0]
    assert "response_format" not in sent, "JSON is requested in the prompt, not via response_format"
    assert "reasoning_effort" not in sent, "reasoning_effort is sent only when set"
    assert sent["max_tokens"] == s.raw["llm"]["max_output_tokens"] and sent["temperature"] == 0
    assert (r.text, r.input_tokens, r.output_tokens, r.provider) == ('{"answer": "x"}', 21, 9, "openai_compatible")
    info = c.endpoint_info()
    assert info == {"provider": "openai_compatible", "base_url_host": "generativelanguage.googleapis.com",
                    "model": "m-compat", "reasoning_effort": None}
    assert "test-key" not in repr(info)


def test_reasoning_effort_passed_only_when_set(monkeypatch):
    s = _settings(LLM_PROVIDER="openai_compatible", LLM_BASE_URL=GEMINI_URL, LLM_API_KEY="test-key-123456789",
                  LLM_MODEL="m-compat", LLM_REASONING_EFFORT="High")
    c = get_llm_client(s)
    seen = []
    monkeypatch.setattr(c.client.chat.completions, "create", _fake_create(seen, finish="length"))
    r = c.complete("S", "U")
    assert seen[0]["reasoning_effort"] == "high"
    assert any("output-token limit" in n for n in r.notes)
    assert s.snapshot()["llm"]["reasoning_effort"] == "high"
    assert s.snapshot()["llm"]["base_url_host"] == "generativelanguage.googleapis.com"


def test_max_tokens_name_fallback(monkeypatch):
    c = OpenAICompatibleClient("m", "test-key-123456789", "https://example.test/v1", **KW)
    seen = []
    ok = _fake_create(seen)

    def create(**kwargs):
        if "max_tokens" in kwargs:
            seen.append(dict(kwargs))
            raise ValueError("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
        return ok(**kwargs)

    monkeypatch.setattr(c.client.chat.completions, "create", create)
    r = c.complete("S", "U")
    assert "max_completion_tokens" in seen[-1] and r.notes


def test_openai_compatible_config_errors():
    bad = [dict(LLM_BASE_URL="", LLM_API_KEY="k" * 12, LLM_MODEL="m"),
           dict(LLM_BASE_URL="generativelanguage.googleapis.com", LLM_API_KEY="k" * 12, LLM_MODEL="m"),
           dict(LLM_BASE_URL=GEMINI_URL, LLM_API_KEY="", LLM_MODEL="m"),
           dict(LLM_BASE_URL=GEMINI_URL, LLM_API_KEY="k" * 12, LLM_MODEL="m", LLM_REASONING_EFFORT="max")]
    for env in bad:
        with pytest.raises(LLMConfigError):
            get_llm_client(_settings(LLM_PROVIDER="openai_compatible", **env))
    with pytest.raises(LLMConfigError):
        get_llm_client(_settings(LLM_PROVIDER="anthropic", ANTHROPIC_API_KEY="k" * 12, LLM_MODEL="m",
                                 LLM_REASONING_EFFORT="low"))


def test_llm_api_key_is_redacted_from_logs(tmp_path):
    from rag.request_log import append_request_log

    secret = "AIzaSyTESTKEYTHATMUSTNOTAPPEAR123"
    s = _settings(LLM_PROVIDER="openai_compatible", LLM_BASE_URL=GEMINI_URL, LLM_API_KEY=secret, LLM_MODEL="m")
    s.raw["paths"]["requests_log"] = str(tmp_path / "r.jsonl")
    append_request_log(s, {"question": f"leak {secret}?", "config": s.snapshot()})
    text = (tmp_path / "r.jsonl").read_text(encoding="utf-8")
    assert secret not in text and "generativelanguage.googleapis.com" in text
