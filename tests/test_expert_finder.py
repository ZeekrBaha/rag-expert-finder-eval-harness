import json
from app.corpus import Record
from app.embedder import HashingEmbedder
from app.llm_client import FakeLLM
from app.expert_finder import ExpertFinder

CORPUS = [
    Record("W1", "Solid-state electrolytes", "lithium garnet solid electrolyte battery",
           "Jane Li", "MIT", "batteries", 40),
    Record("W2", "CRISPR delivery", "lipid nanoparticle crispr cas9 gene editing",
           "Sam Ng", "Broad", "crispr", 33),
    Record("W3", "Perovskite solar", "perovskite solar cell stability photovoltaic",
           "Mo Park", "KAIST", "solar", 28),
]


def _finder(llm):
    return ExpertFinder(corpus=CORPUS, embedder=HashingEmbedder(dim=256), llm=llm, top_k=2)


def test_retrieval_surfaces_relevant_candidate_for_qualify():
    # FakeLLM picks W1; we assert the prompt it received actually contained W1.
    llm = FakeLLM([json.dumps({"expert_id": "W1", "expert_name": "Jane Li",
                               "abstain": False, "reasoning": "garnet electrolyte"})])
    finder = _finder(llm)
    out = finder.run("Who works on solid-state battery electrolytes?")
    assert out.result.expert_id == "W1"
    assert "W1" in llm.prompts[0]                       # W1 retrieved into the prompt
    assert [c.id for c in out.candidates][:1] == ["W1"] # top candidate is the battery one


def test_run_reports_retrieved_ids_for_metrics():
    llm = FakeLLM([json.dumps({"expert_id": "W2", "expert_name": "Sam Ng",
                               "abstain": False, "reasoning": "x"})])
    finder = _finder(llm)
    out = finder.run("crispr gene editing delivery")
    assert "W2" in out.retrieved_ids
    assert len(out.retrieved_ids) == 2                  # top_k


def test_hallucination_flagged_when_expert_not_in_retrieved():
    # FakeLLM confidently names an id that was never retrieved -> hallucinated.
    llm = FakeLLM([json.dumps({"expert_id": "W99", "expert_name": "Ghost",
                               "abstain": False, "reasoning": "fabricated"})])
    finder = _finder(llm)
    out = finder.run("Who works on solid-state battery electrolytes?")
    assert "W99" not in out.retrieved_ids
    assert out.hallucinated is True


def test_regression_variant_drops_evidence_instruction_from_prompt():
    llm = FakeLLM([json.dumps({"expert_id": "W1", "expert_name": "Jane Li",
                               "abstain": False, "reasoning": "x"})])
    finder = ExpertFinder(corpus=CORPUS, embedder=HashingEmbedder(dim=256), llm=llm,
                          top_k=2, prompt_variant="regression")
    finder.run("Who works on solid-state battery electrolytes?")
    prompt = llm.prompts[0]
    assert "cite concrete evidence" not in prompt.lower()
    assert "if no candidate is a strong match, abstain" not in prompt.lower()


def test_no_hallucination_for_normal_in_set_match():
    llm = FakeLLM([json.dumps({"expert_id": "W1", "expert_name": "Jane Li",
                               "abstain": False, "reasoning": "garnet electrolyte"})])
    finder = _finder(llm)
    out = finder.run("Who works on solid-state battery electrolytes?")
    assert out.hallucinated is False
