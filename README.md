# Clinical Knowledge Graph QA — GraphRAG

An **Ollama-first**, fully local **GraphRAG** service that answers questions
over a **clinical knowledge graph** — patients, encounters, conditions (ICD-10),
medications (drug classes), and providers.

**Problem statement.** Clinical questions are inherently *relational*:
"which drugs treat this condition", "what conditions does this patient have and
what were they prescribed", "which providers manage diabetes". Flat text RAG
retrieves disconnected passages; a knowledge graph captures the *connections*,
and GraphRAG retrieves a connected neighborhood of facts to answer from.

> All sample data is **synthetic** — no real patient data / PHI.

## How GraphRAG works here

```
structured records ──build──► clinical knowledge graph
                                     (Patient ─HAS_ENCOUNTER─► Encounter ─DIAGNOSED─► Condition
                                      Encounter ─PRESCRIBED─► Medication ─TREATS─► Condition ...)
question ──entity match──► seed nodes ──k-hop expand──► subgraph ──serialize facts──► LLM ──► answer
```

- **Graph** (`core/graph.py`) — dependency-free directed property graph with
  typed nodes/edges and k-hop neighborhood expansion.
- **Loader** (`core/loader.py`) — builds the graph from structured records
  (reliable, vs. brittle text extraction).
- **Graph-aware retrieval** (`core/retriever.py`) — matches the question to
  nodes, expands their neighborhood, and serializes the subgraph into fact
  lines. This pulls in *connected* facts a text chunk wouldn't co-locate.
- **LLM** only turns the retrieved facts into prose (Ollama, or a deterministic
  offline extractive fallback so the pipeline + tests run with no model server).

## Graph schema

| Relation | Meaning |
|----------|---------|
| `Patient -HAS_ENCOUNTER-> Encounter` | a visit |
| `Encounter -SEEN_BY-> Provider` | attending provider |
| `Encounter -DIAGNOSED-> Condition` | diagnosis at the visit |
| `Encounter -PRESCRIBED-> Medication` | prescription at the visit |
| `Medication -TREATS-> Condition` | drug knowledge |
| `Patient -HAS_CONDITION-> Condition` | rolled up for convenience |

## Quickstart

```bash
pip install -r requirements.txt

ollama pull llama3.2                     # optional, for natural-language prose
python demo.py                           # or KG_OFFLINE=1 python demo.py

uvicorn clinicalgraph.api.main:app --reload   # API at /docs (loads data/clinical.json)
```

### Example

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Which medications treat hypertension?"}'
```

```json
{
  "answer": "Medication 'Lisinopril' treats Condition 'Hypertension'. Medication 'Amlodipine' treats Condition 'Hypertension'.",
  "facts": ["Condition 'Hypertension' (icd10=I10).", "Medication 'Lisinopril' treats Condition 'Hypertension'.", "..."],
  "seeds": ["cond:htn"]
}
```

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `KG_LLM_MODEL` | `llama3.2` | generation model |
| `KG_OFFLINE` | `0` | `1` forces the offline backend |
| `KG_HOPS` | `1` | neighborhood expansion radius |

## Tests

```bash
pytest -q          # offline; no Ollama needed
```

CI lints, runs tests + the demo offline, and builds the Docker image.
