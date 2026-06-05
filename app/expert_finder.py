from dataclasses import dataclass

from app.corpus import Record
from app.qualify import build_qualify_prompt, parse_qualify_output, QualifyResult


@dataclass
class FinderOutput:
    retrieved_ids: list[str]
    candidates: list[Record]
    result: QualifyResult
    hallucinated: bool


class ExpertFinder:
    """System-under-test: embed corpus once, retrieve top-k, qualify via LLM."""

    def __init__(self, corpus: list[Record], embedder, llm, top_k: int = 5):
        from app.retrieval import top_k as _topk
        self._corpus = corpus
        self._embedder = embedder
        self._llm = llm
        self._k = top_k
        self._topk = _topk
        self._matrix = embedder.embed_batch([r.text() for r in corpus])

    def run(self, query: str) -> FinderOutput:
        qvec = self._embedder.embed(query)
        ranked = self._topk(qvec, self._matrix, self._k)
        candidates = [self._corpus[i] for i, _ in ranked]
        retrieved_ids = [c.id for c in candidates]
        prompt = build_qualify_prompt(query, candidates)
        raw = self._llm.complete(prompt)
        result = parse_qualify_output(raw)
        hallucinated = (not result.abstain) and (result.expert_id not in retrieved_ids)
        return FinderOutput(retrieved_ids, candidates, result, hallucinated)
