def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Fraction of relevant ids found in the top-k. Empty relevant set -> 1.0."""
    if not relevant:
        return 1.0
    topk = set(ranked[:k])
    return len(relevant & topk) / len(relevant)


def mrr(relevant: set[str], ranked: list[str]) -> float:
    """Reciprocal rank of the first relevant id (1-indexed); 0.0 if none found."""
    for i, rid in enumerate(ranked, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0
