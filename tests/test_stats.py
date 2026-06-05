from evals.meta.stats import cohens_kappa, confusion_matrix, precision_recall


def test_perfect_agreement_kappa_1():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_chance_agreement_kappa_0():
    # judge ignores truth, always says 1 -> agreement == chance -> kappa 0
    k = cohens_kappa([1, 0, 1, 0], [1, 1, 1, 1])
    assert abs(k) < 1e-9


def test_confusion_matrix_counts():
    cm = confusion_matrix(y_true=[1, 1, 0, 0], y_pred=[1, 0, 0, 0])
    assert cm == {"tp": 1, "fn": 1, "tn": 2, "fp": 0}


def test_precision_recall():
    # y=does a correct match exist; pred=model asserted a match
    p, r = precision_recall(y_true=[True, True, False], y_pred=[True, False, True])
    assert p == 0.5   # TP=1, FP=1
    assert r == 0.5   # TP=1, FN=1
