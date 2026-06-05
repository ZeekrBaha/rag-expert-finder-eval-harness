"""Promptfoo python provider. Promptfoo calls call_api(prompt, options, context)
per test case; we ignore `prompt` and run the ExpertFinder for context.vars.query.
Build one finder per model_provider once per process.
"""
import json
import sys
from pathlib import Path

_CORPUS_PATH = Path(__file__).resolve().parent.parent / "data/corpus.jsonl"

_FINDERS = {}


def _get_finder(model_provider: str, variant: str = "good"):
    key = (model_provider, variant)
    if key in _FINDERS:
        return _FINDERS[key]
    from app.config import Settings
    from app.corpus import load_jsonl
    from app.embedder import OpenAIEmbedder, HashingEmbedder
    from app.expert_finder import ExpertFinder
    from app.llm_client import OpenAIChat, AnthropicChat, DeepSeekChat

    s = Settings.from_env()
    corpus = load_jsonl(_CORPUS_PATH)
    if s.openai_api_key:
        embedder = OpenAIEmbedder(s.openai_api_key, s.embed_model)
    else:
        print("WARNING: no OPENAI_API_KEY — using HashingEmbedder; "
              "retrieval is degraded", file=sys.stderr)
        embedder = HashingEmbedder()
    llm = {
        "openai": lambda: OpenAIChat(s.openai_api_key, s.candidate_openai),
        "anthropic": lambda: AnthropicChat(s.anthropic_api_key, s.candidate_anthropic),
        "deepseek": lambda: DeepSeekChat(s.deepseek_api_key, s.candidate_deepseek),
    }[model_provider]()
    finder = ExpertFinder(corpus, embedder, llm, top_k=s.top_k, prompt_variant=variant)
    _FINDERS[key] = finder
    return finder


def call_api(prompt, options, context):
    cfg = (options or {}).get("config", {})
    provider = cfg.get("model_provider", "openai")
    variant = cfg.get("prompt_variant", "good")
    query = context["vars"]["query"]
    finder = _get_finder(provider, variant)
    out = finder.run(query)
    r = out.result
    candidates = [{"id": c.id, "name": c.author, "abstract": c.abstract[:200]}
                  for c in out.candidates]
    return {"output": json.dumps({
        "expert_id": r.expert_id, "expert_name": r.expert_name,
        "abstain": r.abstain, "reasoning": r.reasoning,
        "retrieved_ids": out.retrieved_ids,
        "hallucinated": out.hallucinated,
        "candidates": candidates,
    })}
