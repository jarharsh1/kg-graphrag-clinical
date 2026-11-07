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


settings = Settings()
