"""End-to-end: load the clinical graph and answer graph-aware questions."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from clinicalgraph.api.main import app
from clinicalgraph.core.loader import build_graph
from clinicalgraph.core.retriever import retrieve_facts

DATA = json.loads(Path("data/clinical.json").read_text(encoding="utf-8"))


def test_loader_builds_relations():
    g = build_graph(DATA)
    # Medication -> condition treatment edge exists.
    rels = {(e.src, e.rel, e.dst) for e in g.edges}
    assert ("med:lisinopril", "TREATS", "cond:htn") in rels
    # Patient rolled-up condition.
    assert ("pt:001", "HAS_CONDITION", "cond:htn") in rels


def test_retrieve_facts_for_drug_query():
    g = build_graph(DATA)
    facts, seeds = retrieve_facts(g, "which medications treat hypertension", hops=1)
    assert seeds
    joined = " ".join(facts).lower()
    assert "hypertension" in joined
    assert "lisinopril" in joined or "amlodipine" in joined


def test_api_health_and_ask():
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["backend"] == "offline"
    assert h.json()["graph_nodes"] > 0  # sample loaded at startup

    r = client.post("/ask", json={"question": "Which medications treat hypertension?"})
    assert r.status_code == 200
    body = r.json()
    assert body["facts"]
    assert "lisinopril" in body["answer"].lower() or "amlodipine" in body["answer"].lower()
