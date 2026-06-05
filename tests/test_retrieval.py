import numpy as np
from app.retrieval import top_k


def test_top_k_ranks_by_cosine_desc():
    q = np.array([1.0, 0.0])
    matrix = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])
    result = top_k(q, matrix, k=2)
    assert [idx for idx, _ in result] == [1, 2]      # exact match, then 45°
    assert result[0][1] > result[1][1]


def test_top_k_caps_at_matrix_size():
    q = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 0.0]])
    assert len(top_k(q, matrix, k=5)) == 1
