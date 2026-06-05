from app.golden import GoldenCase, load_golden, validate_classes


def test_load_golden(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        '{"query":"Who works on solid-state battery electrolytes?",'
        '"kind":"positive","correct_ids":["W1"],"rationale":"direct match"}\n'
        '{"query":"Expert in underwater basket weaving?","kind":"abstain",'
        '"correct_ids":[],"rationale":"no match in corpus"}\n'
    )
    cases = load_golden(p)
    assert len(cases) == 2
    assert cases[0].kind == "positive"
    assert cases[1].correct_ids == []


def test_validate_classes_requires_all_three():
    cases = [GoldenCase("q", "positive", ["W1"], "r")]
    ok, missing = validate_classes(cases)
    assert ok is False
    assert set(missing) == {"hard_negative", "abstain"}


def test_validate_classes_passes_when_all_present():
    cases = [
        GoldenCase("q1", "positive", ["W1"], "r"),
        GoldenCase("q2", "hard_negative", ["W2"], "adjacent field is wrong"),
        GoldenCase("q3", "abstain", [], "none in set"),
    ]
    ok, missing = validate_classes(cases)
    assert ok is True and missing == []
