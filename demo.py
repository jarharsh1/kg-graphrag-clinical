"""End-to-end CLI demo: load the clinical graph and answer graph-aware questions.

    python demo.py            # Ollama if up, else offline
    KG_OFFLINE=1 python demo.py
"""

import json
from pathlib import Path

from clinicalgraph.core.pipeline import ClinicalGraphRAG

QUESTIONS = [
    "Which medications treat hypertension?",
    "What conditions does Patient 001 have?",
    "What did Dr. Rao prescribe and for which conditions?",
]


def main() -> None:
    data = json.loads(Path("data/clinical.json").read_text(encoding="utf-8"))
    rag = ClinicalGraphRAG()
    n = rag.load(data)
    print(f"Loaded graph with {n} nodes (backend: {type(rag.llm).__name__})\n")

    for q in QUESTIONS:
        result = rag.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"   [seeds: {result['seeds'][:3]}, facts retrieved: {len(result['facts'])}]")
        print()


if __name__ == "__main__":
    main()
