"""Provider-agnostic LLM client: one interface, three implementations.

* openai            - OpenAI API (chat completions), OPENAI_API_KEY
* anthropic         - Anthropic Messages API, ANTHROPIC_API_KEY
* openai_compatible - any server that speaks the OpenAI chat-completions API (e.g. Gemini's
                      OpenAI-compatible endpoint), via the OpenAI SDK with LLM_BASE_URL + LLM_API_KEY.
                      No `response_format` is sent: JSON is requested in the prompt and parsed by
                      rag.pipeline (up to 2 parse retries), because compatible servers differ in how
                      strictly they support JSON mode.

Provider and model come from .env (LLM_PROVIDER, LLM_MODEL); nothing is hard-coded. Temperature is fixed
by config (0). LLM_REASONING_EFFORT (low | medium | high) is sent only when set. Transient API errors are
retried by the SDK with back-off (`api_max_retries`).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from rag.config import LLM_PROVIDERS, REASONING_EFFORTS, Settings


class LLMConfigError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    model: str
    provider: str
    latency_ms: float
    request_id: str | None = None
    notes: list[str] = field(default_factory=list)   # e.g. "temperature not supported by model; omitted"
    finish_reason: str | None = None


def _host(url: Any) -> str:
    try:
        return urlparse(str(url)).hostname or ""
    except Exception:
        return ""


class LLMClient(ABC):
    provider: str

    def __init__(self, model: str, temperature: float, max_output_tokens: int, max_retries: int, timeout: float,
                 reasoning_effort: str | None = None):
        if not model:
            raise LLMConfigError("LLM_MODEL is empty. Set it in .env (see .env.example).")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort or None
        self._temperature_supported = True

    @property
    def base_url_host(self) -> str:
        return _host(getattr(self.client, "base_url", ""))

    def endpoint_info(self) -> dict[str, Any]:
        """What is logged for every request (never the key)."""
        return {"provider": self.provider, "base_url_host": self.base_url_host, "model": self.model,
                "reasoning_effort": self.reasoning_effort}

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse: ...

    @staticmethod
    def _is_temperature_error(e: Exception) -> bool:
        msg = str(e).lower()
        return "temperature" in msg and ("unsupported" in msg or "not support" in msg or "does not support" in msg
                                         or "only the default" in msg or "deprecated" in msg)

    def _create_with_fallbacks(self, create: Callable[..., Any], kwargs: dict, notes: list[str]) -> Any:
        """Call the SDK; if the model rejects `temperature`, or the server wants the other output-token
        parameter name, adjust once and retry. Any other error is raised unchanged."""
        for _ in range(3):
            try:
                return create(**kwargs)
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in kwargs and self._is_temperature_error(e):
                    self._temperature_supported = False
                    kwargs.pop("temperature")
                    notes.append("model rejected temperature; request sent without it")
                    continue
                if "max_completion_tokens" in kwargs and "max_completion_tokens" in msg:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    notes.append("server rejected max_completion_tokens; sent max_tokens instead")
                    continue
                if "max_tokens" in kwargs and "max_tokens" in msg and "max_completion_tokens" in msg:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    notes.append("server rejected max_tokens; sent max_completion_tokens instead")
                    continue
                raise
        return create(**kwargs)


class _ChatCompletionsClient(LLMClient):
    """Shared by openai and openai_compatible (OpenAI SDK, chat-completions API only)."""

    send_response_format = True
    token_param = "max_completion_tokens"

    def _make_client(self, api_key: str, base_url: str | None):
        from openai import OpenAI

        kw = {"api_key": api_key, "max_retries": self.max_retries, "timeout": self.timeout}
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(**kw)

    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        notes: list[str] = []
        kwargs: dict[str, Any] = {"model": self.model,
                                  "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
                                  self.token_param: self.max_output_tokens}
        if json_mode and self.send_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if self._temperature_supported:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.perf_counter()
        resp = self._create_with_fallbacks(self.client.chat.completions.create, kwargs, notes)
        dt = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish == "length":
            notes.append("reply stopped at the output-token limit (llm.max_output_tokens)")
        u = getattr(resp, "usage", None)
        return LLMResponse(text=(choice.message.content or ""), provider=self.provider,
                           model=getattr(resp, "model", None) or self.model,
                           input_tokens=getattr(u, "prompt_tokens", None),
                           output_tokens=getattr(u, "completion_tokens", None),
                           latency_ms=round(dt, 1), request_id=getattr(resp, "id", None), notes=notes,
                           finish_reason=finish)


class OpenAIClient(_ChatCompletionsClient):
    provider = "openai"

    def __init__(self, model: str, api_key: str, **kw):
        super().__init__(model, **kw)
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=openai needs OPENAI_API_KEY in .env")
        self._make_client(api_key, None)


class OpenAICompatibleClient(_ChatCompletionsClient):
    """Any OpenAI-compatible chat-completions server (Gemini, vLLM, Ollama, ...)."""

    provider = "openai_compatible"
    send_response_format = False      # ask for JSON in the prompt; rag.pipeline parses and retries
    token_param = "max_tokens"        # the most widely supported name on compatible servers

    def __init__(self, model: str, api_key: str, base_url: str, **kw):
        super().__init__(model, **kw)
        if not base_url or not base_url.lower().startswith(("http://", "https://")):
            raise LLMConfigError("LLM_PROVIDER=openai_compatible needs LLM_BASE_URL (http/https) in .env")
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=openai_compatible needs LLM_API_KEY in .env")
        self._make_client(api_key, base_url)


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, **kw):
        super().__init__(model, **kw)
        if not api_key:
            raise LLMConfigError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY in .env")
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=self.max_retries, timeout=self.timeout)

    def complete(self, system: str, user: str, json_mode: bool = True) -> LLMResponse:
        notes: list[str] = []
        kwargs = {"model": self.model, "system": system, "max_tokens": self.max_output_tokens,
                  "messages": [{"role": "user", "content": user}]}
        if self._temperature_supported:
            kwargs["temperature"] = self.temperature
        t0 = time.perf_counter()
        resp = self._create_with_fallbacks(self.client.messages.create, kwargs, notes)
        dt = (time.perf_counter() - t0) * 1000
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        stop = getattr(resp, "stop_reason", None)
        if stop == "max_tokens":
            notes.append("reply stopped at the output-token limit (llm.max_output_tokens)")
        u = getattr(resp, "usage", None)
        return LLMResponse(text=text, provider=self.provider, model=getattr(resp, "model", self.model),
                           input_tokens=getattr(u, "input_tokens", None), output_tokens=getattr(u, "output_tokens", None),
                           latency_ms=round(dt, 1), request_id=getattr(resp, "id", None), notes=notes,
                           finish_reason=stop)


def get_llm_client(s: Settings) -> LLMClient:
    llm = s.raw["llm"]
    effort = s.llm_reasoning_effort
    if effort and effort not in REASONING_EFFORTS:
        raise LLMConfigError(f"LLM_REASONING_EFFORT={effort!r}; use one of {', '.join(REASONING_EFFORTS)} or leave it empty")
    kw = dict(temperature=llm["temperature"], max_output_tokens=llm["max_output_tokens"],
              max_retries=llm["api_max_retries"], timeout=llm["timeout_seconds"], reasoning_effort=effort or None)
    if s.llm_provider == "openai":
        return OpenAIClient(s.llm_model, s.secret("OPENAI_API_KEY"), **kw)
    if s.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(s.llm_model, s.secret("LLM_API_KEY"), s.llm_base_url, **kw)
    if s.llm_provider == "anthropic":
        if effort:
            raise LLMConfigError("LLM_REASONING_EFFORT is only sent with openai / openai_compatible; "
                                 "leave it empty for LLM_PROVIDER=anthropic")
        return AnthropicClient(s.llm_model, s.secret("ANTHROPIC_API_KEY"), **kw)
    raise LLMConfigError(f"LLM_PROVIDER={s.llm_provider!r}; set it to one of {', '.join(LLM_PROVIDERS)} in .env "
                         f"(run: python scripts/check_env.py)")
