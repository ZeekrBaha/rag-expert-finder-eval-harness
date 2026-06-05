from app.metrics import recall_at_k, mrr


def test_recall_at_k():
    assert recall_at_k(relevant={"W1", "W2"}, ranked=["W3", "W1", "W9"], k=2) == 0.5
    assert recall_at_k(relevant={"W1"}, ranked=["W1", "W2"], k=1) == 1.0
    assert recall_at_k(relevant=set(), ranked=["W1"], k=1) == 1.0  # nothing to find


def test_mrr():
    assert mrr(relevant={"W2"}, ranked=["W1", "W2", "W3"]) == 0.5   # rank 2 -> 1/2
    assert mrr(relevant={"W1"}, ranked=["W1"]) == 1.0
    assert mrr(relevant={"W9"}, ranked=["W1", "W2"]) == 0.0         # not found
