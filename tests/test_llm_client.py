from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.llm_client import FakeLLM


def test_fake_llm_returns_queued_and_records_prompt():
    fake = FakeLLM(["resp-1", "resp-2"])
    assert fake.complete("p1") == "resp-1"
    assert fake.complete("p2") == "resp-2"
    assert fake.prompts == ["p1", "p2"]


def test_real_adapters_import():
    # Construction must not require network; only import + attribute presence.
    from app.llm_client import OpenAIChat, AnthropicChat, DeepSeekChat
    for cls in (OpenAIChat, AnthropicChat, DeepSeekChat):
        assert hasattr(cls, "complete")


def test_fake_llm_queue_exhaustion_raises_index_error():
    fake = FakeLLM(["only-one"])
    assert fake.complete("p1") == "only-one"
    with pytest.raises(IndexError):
        fake.complete("p2")
    # the failing prompt was still recorded before the queue was popped
    assert fake.prompts == ["p1", "p2"]


def test_openai_chat_propagates_sdk_timeout():
    from app.llm_client import OpenAIChat
    c = OpenAIChat(api_key="sk-test")
    c._client = Mock()
    c._client.chat.completions.create.side_effect = httpx.ReadTimeout("timed out")
    with pytest.raises(httpx.ReadTimeout):
        c.complete("hello")


def test_openai_chat_none_content_becomes_empty_string():
    from app.llm_client import OpenAIChat
    c = OpenAIChat(api_key="sk-test")
    c._client = Mock()
    c._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    assert c.complete("hello") == ""


def test_anthropic_chat_propagates_sdk_exception():
    from app.llm_client import AnthropicChat
    c = AnthropicChat(api_key="sk-ant-test")
    c._client = Mock()
    c._client.messages.create.side_effect = httpx.ConnectError("boom")
    with pytest.raises(httpx.ConnectError):
        c.complete("hello")


def test_anthropic_chat_non_text_block_becomes_empty_string():
    from app.llm_client import AnthropicChat
    c = AnthropicChat(api_key="sk-ant-test")
    c._client = Mock()
    # e.g. a tool-use block has no .text attribute
    c._client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use")])
    assert c.complete("hello") == ""


def test_deepseek_chat_propagates_sdk_timeout():
    from app.llm_client import DeepSeekChat
    c = DeepSeekChat(api_key="sk-test")
    c._client = Mock()
    c._client.chat.completions.create.side_effect = httpx.ReadTimeout("timed out")
    with pytest.raises(httpx.ReadTimeout):
        c.complete("hello")
