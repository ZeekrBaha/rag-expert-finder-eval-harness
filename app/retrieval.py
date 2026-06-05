import numpy as np


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return [(row_index, cosine_score)] for the top-k rows, score-descending.

    Assumes rows may be unnormalized; computes cosine explicitly.
    """
    if matrix.shape[0] == 0:
        return []
    query_vec = np.nan_to_num(np.asarray(query_vec, dtype=np.float64))
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64))
    qn = np.linalg.norm(query_vec)
    row_norms = np.linalg.norm(matrix, axis=1)
    denom = row_norms * qn
    denom[denom == 0] = 1e-12
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = (matrix @ query_vec) / denom
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    k = min(k, matrix.shape[0])
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]
