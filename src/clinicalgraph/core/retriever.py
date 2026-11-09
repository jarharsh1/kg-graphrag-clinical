"""Graph-aware retrieval: match entities, expand their neighborhood, serialize.

Unlike flat vector RAG, retrieval here is structural: we find the graph nodes
the question mentions (by name/type keyword match), expand k hops around them,
and serialize the resulting subgraph into readable fact lines that the LLM then
summarizes. This naturally pulls in connected facts — a drug's conditions, a
patient's encounters and prescriptions — that a text chunk might not co-locate.
"""

from __future__ import annotations

import re

from clinicalgraph.config import settings
from clinicalgraph.core.graph import Edge, PropertyGraph

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _type_tokens(graph: PropertyGraph) -> set[str]:
    """Tokens that merely name a node *type* (singular and plural).

    These are generic ("condition(s)", "patient(s)", "medication(s)", ...) and
    appear in the type word of a question — and often inside node names too
    (e.g. every Patient is named "Patient NNN"). Excluding them from the query's
    seed tokens stops a type word from seeding every node of that type.
    """
    toks: set[str] = set()
    for t in {n.type for n in graph.nodes.values()}:
        base = t.lower()
        toks.add(base)
        toks.add(base + "s")
    return toks


def match_seed_nodes(graph: PropertyGraph, query: str, max_seeds: int | None = None) -> list[str]:
    """Seed nodes whose NAME/id shares a content token with the query.

    Matches against each node's name (and id tokens) only — NOT its type. Matching
    on type would let a type word ("conditions", "patient", "medication") seed
    every node of that type and balloon the k-hop expansion into the whole graph.
    Generic type-name tokens are also removed from the query's seed tokens, so a
    type word in the question (or embedded in node names like "Patient 001")
    does not by itself seed every node of that type.

    Tokens of length 1 are dropped (pure noise like "a"), but 2-char tokens are
    kept so short identifiers/codes (e.g. an ICD-10 like "I10") can still match.

    The seed list is sorted by token overlap (descending) and capped to
    ``max_seeds`` (``settings.max_seeds`` when None), so a broad query can't
    expand from dozens of seeds.
    """
    cap = settings.max_seeds if max_seeds is None else max_seeds
    q = {t for t in _tokens(query) if len(t) > 1} - _type_tokens(graph)
    seeds: list[tuple[str, int]] = []
    for node in graph.nodes.values():
        name = str(node.attrs.get("name", node.id))
        hay = _tokens(name) | _tokens(node.id)
        overlap = len(q & hay)
        if overlap:
            seeds.append((node.id, overlap))
    seeds.sort(key=lambda t: t[1], reverse=True)
    return [nid for nid, _ in seeds[:cap]]


def edge_to_fact(graph: PropertyGraph, e: Edge) -> str:
    """Render one edge as a readable fact line using node names."""
    src = graph.nodes[e.src]
    dst = graph.nodes[e.dst]
    sname = src.attrs.get("name", src.id)
    dname = dst.attrs.get("name", dst.id)
    rel = e.rel.replace("_", " ").lower()
    return f"{src.type} '{sname}' {rel} {dst.type} '{dname}'."


def node_to_fact(graph: PropertyGraph, node_id: str) -> str:
    n = graph.nodes[node_id]
    name = n.attrs.get("name", n.id)
    extras = {k: v for k, v in n.attrs.items() if k != "name" and v is not None}
    if not extras:
        return f"{n.type} '{name}'."
    detail = ", ".join(f"{k}={v}" for k, v in extras.items())
    return f"{n.type} '{name}' ({detail})."


def retrieve_facts(graph: PropertyGraph, query: str, hops: int) -> tuple[list[str], list[str]]:
    """Return (fact_lines, seed_node_ids) for the query's neighborhood."""
    seeds = match_seed_nodes(graph, query)
    if not seeds:
        return [], []
    node_ids, edges = graph.expand(seeds, hops)

    # Seed descriptions first, then the neighbors they reached (so attributes
    # like a drug's class or a condition's ICD-10 come along), then relations.
    facts = [node_to_fact(graph, nid) for nid in seeds]
    facts += [node_to_fact(graph, nid) for nid in node_ids if nid not in seeds]
    facts += [edge_to_fact(graph, e) for e in edges]
    # Deduplicate, preserve order.
    seen, out = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out, seeds
