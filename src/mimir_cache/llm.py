"""LLM fallback service — Step 5 of the layered flow (cache miss path).

Production path calls Anthropic Claude Haiku. Without an API key (development,
tests, trace replay simulation) a deterministic echo stub stands in so the full
pipeline stays exercisable at zero cost — consistent with the simulation-based
evaluation methodology in the proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(Protocol):
    async def complete(
        self, prompt: str, context_turns: list[str] | None = None
    ) -> LLMResponse: ...


class EchoLLMClient:
    """Deterministic stub for development and trace-replay simulation."""

    model = "echo-stub"

    async def complete(self, prompt: str, context_turns: list[str] | None = None) -> LLMResponse:
        text = f"[simulated response] {prompt}"
        return LLMResponse(
            text=text,
            input_tokens=max(len(prompt.split()), 1),
            output_tokens=max(len(text.split()), 1),
            model=self.model,
        )


class AnthropicLLMClient:
    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, prompt: str, context_turns: list[str] | None = None) -> LLMResponse:
        messages = []
        for i, turn in enumerate(context_turns or []):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self.model,
        )


class OpenRouterLLMClient:
    """Real LLM fallback via OpenRouter's OpenAI-compatible chat completions API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, max_tokens: int = 1024):
        import httpx

        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0
        )
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, prompt: str, context_turns: list[str] | None = None) -> LLMResponse:
        messages = []
        for i, turn in enumerate(context_turns or []):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            self.BASE_URL,
            json={"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", max(len(prompt.split()), 1)),
            output_tokens=usage.get("completion_tokens", max(len(text.split()), 1)),
            model=self.model,
        )


class OllamaLLMClient:
    """Real LLM fallback via a local Ollama server's OpenAI-compatible API. No key needed."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", max_tokens: int = 1024):
        import httpx

        self._client = httpx.AsyncClient(timeout=60.0)
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens

    async def complete(self, prompt: str, context_turns: list[str] | None = None) -> LLMResponse:
        messages = []
        for i, turn in enumerate(context_turns or []):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            self._url,
            json={"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", max(len(prompt.split()), 1)),
            output_tokens=usage.get("completion_tokens", max(len(text.split()), 1)),
            model=self.model,
        )


def build_llm_client(
    api_key: str | None,
    model: str,
    max_tokens: int,
    openrouter_api_key: str | None = None,
    openrouter_model: str = "",
    ollama_model: str = "",
    ollama_base_url: str = "http://localhost:11434",
) -> LLMClient:
    # Real API usage is strictly opt-in: requires BOTH a key and a model name (Ollama needs no key).
    if ollama_model:
        try:
            return OllamaLLMClient(ollama_model, ollama_base_url, max_tokens)
        except ImportError:
            return EchoLLMClient()
    if openrouter_api_key and openrouter_model:
        try:
            return OpenRouterLLMClient(openrouter_api_key, openrouter_model, max_tokens)
        except ImportError:
            return EchoLLMClient()
    if api_key and model:
        try:
            return AnthropicLLMClient(api_key, model, max_tokens)
        except ImportError:
            return EchoLLMClient()
    return EchoLLMClient()
