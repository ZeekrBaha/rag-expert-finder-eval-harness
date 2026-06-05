import json
from app.corpus import Record
from app.qualify import build_qualify_prompt, parse_qualify_output, QualifyResult

CANDS = [
    Record("W1", "Solid-state electrolytes", "lithium garnet electrolyte", "Jane Li",
           "MIT", "batteries", 40),
    Record("W2", "CRISPR delivery", "lipid nanoparticle", "Sam Ng", "Broad", "crispr", 33),
]


def test_prompt_lists_candidates_and_fences_query():
    p = build_qualify_prompt("Who works on solid-state batteries?", CANDS)
    assert "W1" in p and "Jane Li" in p
    assert "<query>" in p and "</query>"   in p   # query fenced as data
    assert "untrusted" in p.lower()


def test_parse_valid_match():
    raw = json.dumps({"expert_id": "W1", "expert_name": "Jane Li",
                      "abstain": False, "reasoning": "garnet electrolyte work"})
    r = parse_qualify_output(raw)
    assert r == QualifyResult("W1", "Jane Li", False, "garnet electrolyte work")


def test_parse_abstain():
    raw = json.dumps({"expert_id": None, "expert_name": None,
                      "abstain": True, "reasoning": "no candidate fits"})
    r = parse_qualify_output(raw)
    assert r.abstain is True and r.expert_id is None


def test_parse_tolerates_fenced_json():
    raw = "```json\n" + json.dumps({"expert_id": "W2", "expert_name": "Sam Ng",
                                    "abstain": False, "reasoning": "x"}) + "\n```"
    assert parse_qualify_output(raw).expert_id == "W2"


def test_parse_garbage_raises():
    import pytest
    with pytest.raises(ValueError):
        parse_qualify_output("not json at all")
