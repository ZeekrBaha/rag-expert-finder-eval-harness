from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("TOP_K", "7")
    s = Settings.from_env()
    assert s.openai_api_key == "sk-x"
    assert s.top_k == 7
    assert s.candidate_openai == "gpt-4o-mini"  # default


def test_top_k_defaults_to_5(monkeypatch):
    monkeypatch.delenv("TOP_K", raising=False)
    assert Settings.from_env().top_k == 5
