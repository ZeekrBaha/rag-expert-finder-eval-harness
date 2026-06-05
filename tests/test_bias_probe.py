from evals.meta.bias_probe import pad_text, flip_rate, swap_positions


def test_pad_text_adds_filler_but_keeps_core():
    core = "Jane Li works on garnet solid electrolytes."
    padded = pad_text(core, factor=3)
    assert core in padded
    assert len(padded) > len(core) * 2


def test_flip_rate_counts_changed_verdicts():
    # baseline vs variant verdicts (True=pass). 1 of 4 flipped.
    assert flip_rate([True, True, False, False], [True, False, False, False]) == 0.25


def test_flip_rate_zero_when_identical():
    assert flip_rate([True, False], [True, False]) == 0.0


def test_flip_rate_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        flip_rate([True], [True, False])


def test_swap_positions_swaps_first_two():
    assert swap_positions(["a", "b", "c"]) == ["b", "a", "c"]


def test_swap_positions_single_element_unchanged():
    assert swap_positions(["a"]) == ["a"]


def test_swap_positions_empty_unchanged():
    assert swap_positions([]) == []
