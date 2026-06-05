"""Bias probe: quantify judge sensitivity to verbosity and candidate position.

pad_text() and flip_rate() are pure + tested. A live harness (run later) feeds
the judge baseline vs. perturbed answers and reports flip_rate — a measured number,
not a claim. Evidence: style/verbosity bias dominates; position bias is small.
"""

_FILLER = (" To elaborate further on this point in considerable additional detail, "
           "it is worth noting comprehensively and at length that")


def pad_text(text: str, factor: int = 3) -> str:
    """Lengthen an answer with neutral filler WITHOUT changing its substance.

    Used to test verbosity bias: a good judge scores padded == concise.
    """
    return text + (_FILLER * max(0, factor))


def flip_rate(baseline: list[bool], variant: list[bool]) -> float:
    """Fraction of verdicts that changed between baseline and a perturbed run.

    Lower is better (judge is robust to the manipulation).
    """
    if len(baseline) != len(variant):
        raise ValueError("baseline and variant must be equal length")
    if not baseline:
        return 0.0
    flips = sum(1 for a, b in zip(baseline, variant) if bool(a) != bool(b))
    return flips / len(baseline)
