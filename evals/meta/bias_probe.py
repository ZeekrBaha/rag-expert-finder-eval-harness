"""Bias probe: quantify judge sensitivity to verbosity and candidate position.

pad_text() perturbs verbosity, swap_positions() perturbs candidate order, and
flip_rate() scores how often verdicts change under a perturbation. All three are
pure + tested. A live harness (run later) feeds the judge baseline vs. perturbed
answers and reports flip_rate — a measured number, not a claim. Expected evidence:
style/verbosity bias dominates; position bias (swap_positions) is small.
"""

_FILLER = (" To elaborate further on this point in considerable additional detail, "
           "it is worth noting comprehensively and at length that")


def pad_text(text: str, factor: int = 3, filler: str = _FILLER) -> str:
    """Lengthen an answer with neutral filler WITHOUT changing its substance.

    Used to test verbosity bias: a good judge scores padded == concise.
    `filler` defaults to the English constant; pass a custom string for
    other languages or styles.
    """
    return text + (filler * max(0, factor))


def swap_positions(items: list) -> list:
    """Return a copy of `items` with the first two elements swapped.

    Used to perturb candidate order for the position-bias flip-rate. Lists with
    fewer than two elements are returned unchanged (as a copy).
    """
    out = list(items)
    if len(out) >= 2:
        out[0], out[1] = out[1], out[0]
    return out


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
