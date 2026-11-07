# GraphRAG Engine — Knowledge-Graph QA

An **Ollama-first**, fully local, **domain-agnostic** **GraphRAG** engine. It
answers questions over *any* knowledge graph by matching the question to seed
nodes, expanding a **k-hop neighborhood**, serializing that connected subgraph
into facts, and letting an LLM turn those facts into a cited answer.

**Why a graph.** Many questions are inherently *relational* — "which X relate to
Y", "what is connected to this entity and how". Flat text RAG retrieves
disconnected passages; a property graph captures the *connections*, and GraphRAG
retrieves a connected neighborhood of facts to answer from.

**Ships with a worked example: a clinical knowledge graph** — patients,
encounters, conditions (ICD-10), medications (drug classes), and providers — so
the engine runs end to end out of the box. The graph, k-hop retrieval, and
pipeline are **domain-agnostic**: to target a different domain you swap the
records fed to the loader ([see use cases](#use-cases)); you do not change the
engine.

> The shipped sample data is **synthetic** — no real patient data / PHI.

## Use cases

The engine is the same; only the records you feed the loader change. The
`PropertyGraph` + k-hop graph-aware retrieval are domain-agnostic — swap in your
own nodes and relations and the same `/ask` pipeline answers over them.

| Domain | Graph (nodes — relations) | Example question |
|--------|---------------------------|------------------|
| **Healthcare** *(shipped example)* | patients — conditions — medications — providers | "Which drugs treat hypertension?" |
| **Enterprise / HR** | employees — teams — projects | "Who works on project X?" |
| **E-commerce** | customers — orders — products | "What did this customer buy?" |
| **Supply chain** | suppliers — parts — plants | "Which suppliers provide part Y?" |
| **Research** | papers — authors — topics | "Who co-authored on topic Z?" |
| **Fraud / finance** | accounts — transactions — entities | "Which accounts funnel into entity W?" |

Only the loader's **input records** change — the property graph, the k-hop
neighborhood expansion, and the fact-serialization step are identical across all
of these.

## Stack

- **Backend:** FastAPI (an async Flask-equivalent — it serves the HTTP API and
  auto-generates interactive docs at `/docs`).
- **Frontend:** a lightweight browser chat UI (`web/`, plain HTML + JS) served by
  FastAPI at `/` — this is where a user types questions and reads answers.
- **Model:** Ollama (local LLM), with a deterministic offline fallback (so the
  pipeline and tests run with no model server — CI-safe).
- **Deploy:** single Docker image, or `docker compose` (API + Ollama). See
  [DEPLOYMENT.md](DEPLOYMENT.md).

Open **<http://localhost:8000>** for the UI once running.

## How GraphRAG works here

The flow below is the same for any graph; the clinical nodes/edges shown are the
shipped example.

```
structured records ──build──► clinical knowledge graph
                                     (Patient ─HAS_ENCOUNTER─► Encounter ─DIAGNOSED─► Condition
                                      Encounter ─PRESCRIBED─► Medication ─TREATS─► Condition ...)
question ──entity match──► seed nodes ──k-hop expand──► subgraph ──serialize facts──► LLM ──► answer
```

- **Graph** (`core/graph.py`) — dependency-free directed property graph with
  typed nodes/edges and k-hop neighborhood expansion.
- **Loader** (`core/loader.py`) — builds the graph from structured records
  (reliable, vs. brittle text extraction). **This is the only domain-specific
  piece** — point it at your own entity/relation records to retarget the engine.
- **Graph-aware retrieval** (`core/retriever.py`) — matches the question to
  nodes, expands their neighborhood, and serializes the subgraph into fact
  lines. This pulls in *connected* facts a text chunk wouldn't co-locate.
- **LLM** only turns the retrieved facts into prose (Ollama, or a deterministic
  offline extractive fallback so the pipeline + tests run with no model server).

## Graph schema (shipped clinical example)

This is the schema of the bundled clinical example (`data/clinical.json`). Any
graph with its own node/relation types works the same way — the engine does not
hard-code these.

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
