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
