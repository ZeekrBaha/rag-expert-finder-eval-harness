from evals.analyze_run import normalize_ids, oracle_correct, analyze
import json


def test_normalize_ids_handles_list_string_and_empty():
    assert normalize_ids(["W1", "W2"]) == {"W1", "W2"}
    assert normalize_ids("W3") == {"W3"}
    assert normalize_ids("[]") == set()
    assert normalize_ids("") == set()


def test_oracle_correct_logic():
    assert oracle_correct("positive", {"W1"}, "W1", False) is True
    assert oracle_correct("positive", {"W1"}, "W2", False) is False        # wrong pick
    assert oracle_correct("hard_negative", {"W1"}, "W9", False) is False   # fooled
    assert oracle_correct("abstain", set(), None, True) is True
    assert oracle_correct("abstain", set(), "W1", False) is False          # should abstain


def _result(kind, cids, eid, abstain, success):
    return {
        "vars": {"kind": kind, "correct_ids": cids},
        "response": {"output": json.dumps({"expert_id": eid, "abstain": abstain})},
        "success": success,
    }


def test_analyze_computes_kappa_and_precision():
    results = [
        _result("positive", "W1", "W1", False, True),       # correct commit
        _result("positive", "W2", "W9", False, False),      # wrong commit
        _result("hard_negative", "W3", "W8", False, False), # fooled commit
        _result("abstain", "[]", None, True, True),         # correct abstain
    ]
    rep = analyze(results)
    assert rep["n"] == 4
    assert rep["per_kind"]["positive"] == {"passed": 1, "total": 2}
    # judge perfectly tracks the oracle here -> kappa 1.0
    assert rep["judge_vs_oracle"]["kappa"] == 1.0
    # 3 commits, 1 correct -> precision 1/3
    assert rep["confident_match"]["commits"] == 3
    assert abs(rep["confident_match"]["precision"] - 1 / 3) < 1e-9
    assert rep["confident_match"]["wrong_confident_matches"] == 2
