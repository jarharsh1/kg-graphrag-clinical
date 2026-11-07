"""Unit tests for the property graph and k-hop expansion."""

import pytest

from clinicalgraph.core.graph import PropertyGraph


def _toy() -> PropertyGraph:
    g = PropertyGraph()
    g.add_node("a", "Patient", name="A")
    g.add_node("b", "Encounter", name="E1")
    g.add_node("c", "Condition", name="HTN")
    g.add_edge("a", "b", "HAS_ENCOUNTER")
    g.add_edge("b", "c", "DIAGNOSED")
    return g


def test_add_and_len():
    g = _toy()
    assert len(g) == 3


def test_add_edge_requires_endpoints():
    g = PropertyGraph()
    g.add_node("x", "Patient")
    with pytest.raises(KeyError):
        g.add_edge("x", "missing", "REL")


def test_neighbors_both_directions():
    g = _toy()
    nb = g.neighbors("b")
    others = {other for _, _, other in nb}
    assert others == {"a", "c"}  # in-edge from a, out-edge to c


def test_expand_one_hop():
    g = _toy()
    nodes, edges = g.expand(["b"], hops=1)
    assert nodes == {"a", "b", "c"}
    assert len(edges) == 2


def test_expand_zero_hops_is_seed_only():
    g = _toy()
    nodes, edges = g.expand(["a"], hops=0)
    assert nodes == {"a"}
    assert edges == []


def test_nodes_of_type():
    g = _toy()
    assert [n.id for n in g.nodes_of_type("Condition")] == ["c"]
