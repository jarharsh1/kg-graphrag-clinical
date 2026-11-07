"""LLM backend (Ollama-first, deterministic offline fallback).

GraphRAG retrieves structured facts from the clinical graph; the LLM only turns
the retrieved fact list into prose. Offline, the extractive LLM returns the fact
lines most relevant to the question, so answers stay grounded with no model.
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from clinicalgraph.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"what", "which", "the", "a", "an", "of", "is", "are", "for", "to",
         "and", "or", "in", "on", "at", "how", "many", "does", "do", "did",
         "who", "whom", "with"}


def keywords(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 2}


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...


class ExtractiveLLM:
    def generate(self, prompt: str) -> str:
        facts, question = _split(prompt)
        lines = [ln.strip("- ").strip() for ln in facts.splitlines() if ln.strip()]
        if not lines:
            return "The knowledge graph has no facts relevant to that question."
        q = keywords(question)
        scored = sorted(lines, key=lambda ln: len(q & keywords(ln)), reverse=True)
        top = [ln for ln in scored if q & keywords(ln)][:5]
        return " ".join(top) if top else "The graph does not contain that information."


class OllamaLLM:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model

    def generate(self, prompt: str) -> str:
        with httpx.Client(timeout=120) as c:
            r = c.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            return r.json()["response"].strip()


def _split(prompt: str) -> tuple[str, str]:
    facts, question = "", prompt
    if "FACTS:" in prompt and "QUESTION:" in prompt:
        facts = prompt.split("FACTS:", 1)[1].split("QUESTION:", 1)[0].strip()
        question = prompt.split("QUESTION:", 1)[1].strip()
    return facts, question


def _ollama_up(host: str) -> bool:
    try:
        with httpx.Client(timeout=2) as c:
            return c.get(f"{host.rstrip('/')}/api/tags").status_code == 200
    except Exception:
        return False


def get_llm() -> LLM:
    if not settings.offline and _ollama_up(settings.ollama_host):
        return OllamaLLM(settings.ollama_host, settings.llm_model)
    return ExtractiveLLM()
