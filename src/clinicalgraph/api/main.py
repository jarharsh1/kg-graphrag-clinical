"""FastAPI serving layer for clinical GraphRAG.

Endpoints:
  GET  /         - the browser chat UI (static)
  GET  /health   - liveness + active backend + graph size
  POST /load      - (re)build the graph from structured records
  POST /ask       - graph-aware question answering with the retrieved facts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clinicalgraph.core.backends import OllamaLLM
from clinicalgraph.core.pipeline import ClinicalGraphRAG

# Locate the bundled web UI (repo-root/web), resilient to the CWD.
_WEB_DIR = Path(__file__).resolve().parents[3] / "web"

app = FastAPI(
    title="Clinical Knowledge Graph QA (GraphRAG)",
    version="0.1.0",
    description="Ollama-first GraphRAG over a clinical knowledge graph "
    "(patients, encounters, conditions, medications, providers).",
)

_pipeline = ClinicalGraphRAG()

# Load the bundled synthetic dataset at startup if present.
_SAMPLE = Path("data/clinical.json")
if _SAMPLE.exists():
    try:
        _pipeline.load(json.loads(_SAMPLE.read_text(encoding="utf-8")))
    except Exception as exc:  # pragma: no cover - startup best-effort
        print(f"[startup] could not load {_SAMPLE}: {exc}", file=sys.stderr)


class LoadRequest(BaseModel):
    data: dict = Field(..., description="Structured records: patients, providers, "
                       "conditions, medications, encounters.")


class LoadResponse(BaseModel):
    nodes: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Which medications treat hypertension?"])
    hops: int | None = Field(None, ge=0, le=3)


class AskResponse(BaseModel):
    answer: str
    facts: list[str]
    seeds: list[str]


@app.get("/health")
def health() -> dict:
    backend = "ollama" if isinstance(_pipeline.llm, OllamaLLM) else "offline"
    return {"status": "ok", "backend": backend, "graph_nodes": len(_pipeline.graph)}


@app.post("/load", response_model=LoadResponse)
def load(req: LoadRequest) -> LoadResponse:
    return LoadResponse(nodes=_pipeline.load(req.data))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**_pipeline.ask(req.question, req.hops))


# --- Browser UI ------------------------------------------------------------ #
# Serve the static chat UI. Mounted last so it never shadows the API routes.
if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_WEB_DIR / "index.html"))
