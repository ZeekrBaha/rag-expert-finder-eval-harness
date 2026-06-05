from app.golden import GoldenCase
from evals.build_tests import golden_to_tests


def test_golden_to_tests_maps_vars_and_metadata():
    cases = [
        GoldenCase("Who works on batteries?", "positive", ["W1"], "direct"),
        GoldenCase("Underwater basket weaving expert?", "abstain", [], "none"),
    ]
    tests = golden_to_tests(cases)
    assert tests[0]["vars"]["query"] == "Who works on batteries?"
    assert tests[0]["vars"]["kind"] == "positive"
    assert tests[0]["vars"]["correct_ids"] == ["W1"]
    # abstain case signals the expected behavior to the rubric
    assert tests[1]["vars"]["kind"] == "abstain"
    assert tests[1]["vars"]["correct_ids"] == []
