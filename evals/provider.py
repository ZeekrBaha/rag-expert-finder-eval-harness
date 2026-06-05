"""Promptfoo python provider. Promptfoo calls call_api(prompt, options, context)
per test case; we ignore `prompt` and run the ExpertFinder for context.vars.query.
Build the embedded corpus once per process.
"""
import json
import os

_FINDER = None


def _get_finder(model_provider: str):
    global _FINDER
    if _FINDER is not None:
        return _FINDER
    from app.config import Settings
    from app.corpus import load_jsonl
    from app.embedder import OpenAIEmbedder, HashingEmbedder
    from app.expert_finder import ExpertFinder
    from app.llm_client import OpenAIChat, AnthropicChat, DeepSeekChat

    s = Settings.from_env()
    corpus = load_jsonl("data/corpus.jsonl")
    embedder = (OpenAIEmbedder(s.openai_api_key, s.embed_model)
                if s.openai_api_key else HashingEmbedder())
    llm = {
        "openai": lambda: OpenAIChat(s.openai_api_key, s.candidate_openai),
        "anthropic": lambda: AnthropicChat(s.anthropic_api_key, s.candidate_anthropic),
        "deepseek": lambda: DeepSeekChat(s.deepseek_api_key, s.candidate_deepseek),
    }[model_provider]()
    _FINDER = ExpertFinder(corpus, embedder, llm, top_k=s.top_k)
    return _FINDER


def call_api(prompt, options, context):
    provider = (options or {}).get("config", {}).get("model_provider", "openai")
    query = context["vars"]["query"]
    finder = _get_finder(provider)
    out = finder.run(query)
    r = out.result
    return {"output": json.dumps({
        "expert_id": r.expert_id, "expert_name": r.expert_name,
        "abstain": r.abstain, "reasoning": r.reasoning,
        "retrieved_ids": out.retrieved_ids,
    })}
