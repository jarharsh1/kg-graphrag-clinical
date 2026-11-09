"""Runtime configuration (env with local defaults). Ollama-first, offline fallback."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("KG_LLM_MODEL", "llama3.2")
    offline: bool = os.getenv("KG_OFFLINE", "0") == "1"
    # How many hops to expand around matched entities when building context.
    hops: int = int(os.getenv("KG_HOPS", "1"))
    # Cap the seed set before k-hop expansion, so a broad query can't expand
    # from dozens of seeds and pull in the whole graph.
    max_seeds: int = int(os.getenv("KG_MAX_SEEDS", "5"))
    # Cap the number of fact lines placed in the LLM prompt (a fact budget), so
    # a large subgraph can't produce an unbounded prompt.
    max_facts: int = int(os.getenv("KG_MAX_FACTS", "40"))


settings = Settings()
