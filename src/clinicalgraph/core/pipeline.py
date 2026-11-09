"""GraphRAG pipeline: load the clinical graph, then answer via graph retrieval."""

from __future__ import annotations

from clinicalgraph.config import settings
from clinicalgraph.core.backends import get_llm
from clinicalgraph.core.graph import PropertyGraph
from clinicalgraph.core.loader import build_graph
from clinicalgraph.core.retriever import retrieve_facts

_PROMPT = """You are a clinical data assistant. Answer the question using ONLY the
graph facts below. Be specific and do not invent facts. If the facts do not
answer the question, say so.

FACTS:
{facts}

QUESTION:
{question}
"""


class ClinicalGraphRAG:
    def __init__(self, graph: PropertyGraph | None = None):
        self.llm = get_llm()
        self.graph = graph if graph is not None else PropertyGraph()

    def load(self, data: dict) -> int:
        """(Re)build the graph from structured records. Returns node count."""
        self.graph = build_graph(data)
        return len(self.graph)

    def ask(self, question: str, hops: int | None = None) -> dict:
        if len(self.graph) == 0:
            return {"answer": "The knowledge graph is empty.", "facts": [], "seeds": []}

        h = settings.hops if hops is None else hops
        facts, seeds = retrieve_facts(self.graph, question, h)
        if not facts:
            return {
                "answer": "No entities in the question matched the knowledge graph.",
                "facts": [],
                "seeds": [],
            }
        # Fact budget: cap how many fact lines reach the prompt, so a large
        # subgraph can't produce an unbounded prompt. Facts are already ordered
        # most-relevant-first (seeds, then their neighbors, then relations).
        if settings.max_facts > 0:
            facts = facts[: settings.max_facts]
        answer = self.llm.generate(_PROMPT.format(facts="\n".join(facts), question=question))
        return {"answer": answer, "facts": facts, "seeds": seeds}
