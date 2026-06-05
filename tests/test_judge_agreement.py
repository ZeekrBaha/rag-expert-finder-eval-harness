from evals.meta.judge_agreement import agreement_report


def test_agreement_report_computes_kappa_and_confusion():
    human = [True, True, False, False]    # should the model have matched?
    judge = [True, False, False, False]   # did the judge pass the model's answer?
    rep = agreement_report(human, judge)
    assert rep["n"] == 4
    assert rep["pct_agreement"] == 0.75
    assert rep["confusion"] == {"tp": 1, "fn": 1, "tn": 2, "fp": 0}
    assert -1.0 <= rep["kappa"] <= 1.0


def test_agreement_report_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        agreement_report([True], [True, False])
