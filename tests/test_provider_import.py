def test_provider_exposes_call_api():
    import evals.provider as p
    assert hasattr(p, "call_api")


def test_config_yaml_parses():
    # minimal YAML sanity without a yaml dep: ensure key markers exist
    text = open("evals/promptfooconfig.yaml").read()
    assert "providers:" in text and "llm-rubric" in text and "tests.json" in text
