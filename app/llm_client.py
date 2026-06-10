from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class FakeLLM:
    """Deterministic test double; returns queued responses, records prompts."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0)


class OpenAIChat:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""


class AnthropicChat:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:
        r = self._client.messages.create(
            model=self._model, max_tokens=1024, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        block = r.content[0]
        return getattr(block, "text", "")


class DeepSeekChat:
    """DeepSeek is OpenAI-compatible; just a different base URL."""

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    def complete(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model, temperature=0,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
