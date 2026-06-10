import json

from evals.scorecard import build_scorecard, write_scorecard

REPORT = {
    "n": 10,
    "overall_pass": 0.9,
    "per_kind": {
        "positive": {"passed": 4, "total": 5},
        "hard_negative": {"passed": 3, "total": 3},
        "abstain": {"passed": 1, "total": 2},
    },
    "judge_vs_oracle": {"kappa": 0.87, "pct_agreement": 0.95,
                        "confusion": {"tp": 8, "tn": 1, "fp": 0, "fn": 1}},
    "confident_match": {"commits": 8, "correct_commits": 7,
                        "precision": 0.875, "wrong_confident_matches": 1},
    "by_provider": {},
}


def test_build_scorecard_shape_and_values():
    sc = build_scorecard(REPORT, run_id="run-42")
    assert sc["schema_version"] == "1.0"
    assert sc["run_id"] == "run-42"
    ms = sc["metric_summary"]
    assert ms["pass_rate_positive"] == 4 / 5
    assert ms["pass_rate_hard_negative"] == 1.0
    assert ms["pass_rate_abstain"] == 0.5
    assert ms["judge_kappa"] == 0.87
    assert ms["confident_match_precision"] == 0.875


def test_build_scorecard_nulls_for_missing_metrics():
    sc = build_scorecard({"per_kind": {}}, run_id="empty")
    ms = sc["metric_summary"]
    assert ms["pass_rate_positive"] is None
    assert ms["pass_rate_hard_negative"] is None
    assert ms["pass_rate_abstain"] is None
    assert ms["judge_kappa"] is None
    assert ms["confident_match_precision"] is None


def test_write_scorecard_writes_json_file(tmp_path):
    out = tmp_path / "scorecard.json"
    sc = write_scorecard(REPORT, run_id="run-42", path=str(out))
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == sc
    assert on_disk["metric_summary"]["judge_kappa"] == 0.87
