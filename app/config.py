import os
from dataclasses import dataclass


@dataclass
class Settings:
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    deepseek_api_key: str = ""
    candidate_openai: str = "gpt-4o-mini"
    candidate_anthropic: str = "claude-3-5-sonnet-20241022"
    candidate_deepseek: str = "deepseek-chat"
    judge_model: str = "openai:gpt-4o"
    embed_model: str = "text-embedding-3-small"
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            candidate_openai=os.environ.get("CANDIDATE_OPENAI", "gpt-4o-mini"),
            candidate_anthropic=os.environ.get("CANDIDATE_ANTHROPIC",
                                               "claude-3-5-sonnet-20241022"),
            candidate_deepseek=os.environ.get("CANDIDATE_DEEPSEEK", "deepseek-chat"),
            judge_model=os.environ.get("JUDGE_MODEL", "openai:gpt-4o"),
            embed_model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
            top_k=int(os.environ.get("TOP_K", "5")),
        )
