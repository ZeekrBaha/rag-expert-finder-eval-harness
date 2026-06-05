from app.config import Settings
from evals.gen_config import select_providers, render_yaml


def test_only_openai_when_only_openai_key():
    s = Settings(openai_api_key="sk-x")
    provs = select_providers(s)
    assert [p["model_provider"] for p in provs] == ["openai"]


def test_all_three_when_all_keys():
    s = Settings(openai_api_key="a", anthropic_api_key="b", deepseek_api_key="c")
    assert [p["model_provider"] for p in select_providers(s)] == \
        ["openai", "anthropic", "deepseek"]


def test_none_when_no_keys():
    assert select_providers(Settings()) == []


def test_render_yaml_includes_only_given_providers_and_judge():
    y = render_yaml([{"label": "gpt-4o-mini", "model_provider": "openai"}],
                    judge_model="openai:gpt-4o",
                    rubric_text="PASS only if correct.\nIgnore answer length entirely.")
    assert "model_provider: openai" in y
    assert "anthropic" not in y
    assert "provider: openai:gpt-4o" in y
    assert "tests.json" in y
    # rubric inlined as a block scalar, not a file reference
    assert "value: |" in y
    assert "Ignore answer length entirely." in y
    assert "file://graders" not in y
