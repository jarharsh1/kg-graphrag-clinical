"""End-to-end: load the clinical graph and answer graph-aware questions."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from clinicalgraph.api import main as api_main
from clinicalgraph.api.main import app
from clinicalgraph.core.loader import build_graph
from clinicalgraph.core.retriever import match_seed_nodes, retrieve_facts

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


def test_seed_matching_uses_names_not_types():
    # A type word ("conditions") must not seed every Condition node. The patient
    # question should seed only the patient (and resolve its conditions).
    g = build_graph(DATA)
    seeds = match_seed_nodes(g, "What conditions does Patient 001 have?")
    assert seeds == ["pt:001"]
    # None of the Condition nodes are seeds.
    assert not any(s.startswith("cond:") for s in seeds)
    # ...but its conditions are reached by expansion.
    facts, _ = retrieve_facts(g, "What conditions does Patient 001 have?", hops=1)
    joined = " ".join(facts).lower()
    assert "hypertension" in joined and "chronic kidney disease" in joined


def test_seed_set_is_capped():
    # The seed list is capped so a broad query can't expand from many seeds.
    g = build_graph(DATA)
    seeds = match_seed_nodes(g, "patient condition medication provider encounter",
                             max_seeds=2)
    assert len(seeds) <= 2


def test_short_two_char_token_is_kept_and_matches():
    # Lowering the token threshold keeps 2-char tokens usable. A query that only
    # names the id token "pt" (2 chars) now seeds the patient nodes instead of
    # being dropped as too short.
    g = build_graph(DATA)
    seeds = match_seed_nodes(g, "pt 001", max_seeds=5)
    assert "pt:001" in seeds


def test_fact_budget_caps_prompt_facts(monkeypatch):
    import dataclasses

    from clinicalgraph.core import pipeline as pipeline_mod

    # settings is a frozen dataclass; swap in a copy with a tiny fact budget.
    monkeypatch.setattr(
        pipeline_mod, "settings",
        dataclasses.replace(pipeline_mod.settings, max_facts=3),
    )
    rag = pipeline_mod.ClinicalGraphRAG()
    rag.load(DATA)
    out = rag.ask("Patient 001", hops=2)
    assert len(out["facts"]) <= 3
    # Without the budget a 2-hop expansion around a patient yields many facts.
    monkeypatch.setattr(
        pipeline_mod, "settings",
        dataclasses.replace(pipeline_mod.settings, max_facts=0),
    )
    out_full = rag.ask("Patient 001", hops=2)
    assert len(out_full["facts"]) > 3


def test_sample_path_is_cwd_independent():
    # The bundled dataset must resolve from the package location, not the CWD.
    assert api_main._SAMPLE.is_absolute()
    assert api_main._SAMPLE.exists()


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
