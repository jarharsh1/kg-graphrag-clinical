# Code Review — kg-graphrag-clinical

A reviewer-facing deep-dive on the correctness findings in the GraphRAG engine
and the fixes that landed for them. The engine is a dependency-free property
graph with k-hop retrieval; this review focuses on the **retrieval** layer,
where the real defects were, and records what was traced and left unchanged.

Fixes reviewed here: commits `7fe5bc2` (retrieval bounding) and `819bcea`
(dataset path). Everything below quotes the actual before/after from those
commits.

---

## Finding 1 — Seed over-match: a type word seeds *every* node of that type (High)

**Where:** `src/clinicalgraph/core/retriever.py:40` — `match_seed_nodes`
(the `hay` haystack and the query-token filter at `:58`/`:62`).

**Root cause.** Seed matching built each node's match haystack from its **name
*and* its type**, and only dropped it to name-vs-name afterwards:

```python
# before
def match_seed_nodes(graph: PropertyGraph, query: str) -> list[str]:
    """Nodes whose name or type shares a content token with the query."""
    q = {t for t in _tokens(query) if len(t) > 2}
    ...
        hay = _tokens(name) | _tokens(node.type)   # <-- type tokens in the haystack
```

Because the haystack carried the node's **type** tokens, any ordinary question
word that happens to be a type name seeded *every* node of that type. A question
like *"What conditions does Patient 001 have?"* tokenizes to
`{what, conditions, does, patient, have, 001, ...}`; the token `conditions`
matched the type of **all three** `Condition` nodes, and `patient` matched
**both** `Patient` nodes. Those became seeds, and `graph.expand(seeds, hops)`
then k-hopped out of a dozen unrelated starting points — in a small graph that
is effectively "serialize the whole graph", which is exactly what GraphRAG's
targeted retrieval is supposed to avoid. The intended answer (this one patient's
conditions) was buried in noise, and the LLM prompt ballooned.

A second, subtler leak: even after dropping `node.type` from the haystack, every
`Patient` node is *named* `"Patient NNN"`, so the bare word `patient` still
matched both patients through the **name**. The type word needed to be removed
from the *query* side too, not just the haystack side.

**Fix.** Match names/ids only, **and** strip generic type-name tokens
(singular + plural) from the query's seed tokens:

```python
# after
def _type_tokens(graph: PropertyGraph) -> set[str]:
    """Tokens that merely name a node *type* (singular and plural)."""
    toks: set[str] = set()
    for t in {n.type for n in graph.nodes.values()}:
        base = t.lower()
        toks.add(base)
        toks.add(base + "s")
    return toks

def match_seed_nodes(graph, query, max_seeds=None):
    cap = settings.max_seeds if max_seeds is None else max_seeds
    q = {t for t in _tokens(query) if len(t) > 1} - _type_tokens(graph)
    ...
        hay = _tokens(name) | _tokens(node.id)      # names/ids, NOT type
```

With both changes, *"What conditions does Patient 001 have?"* now seeds exactly
`["pt:001"]`, and its two conditions (Hypertension, Chronic Kidney Disease) are
still reached by the 1-hop expansion — the right answer, without the whole-graph
blast radius.

**Regression test.** `tests/test_graphrag_e2e.py::test_seed_matching_uses_names_not_types`
asserts `seeds == ["pt:001"]`, that no `cond:*` node is a seed, and that the
expansion still surfaces both conditions.

---

## Finding 2 — No seed cap: a broad query k-hops from dozens of starts (Medium)

**Where:** `src/clinicalgraph/core/retriever.py:67` (return), with the cap
plumbed from `src/clinicalgraph/config.py:18` (`max_seeds`).

**Root cause.** `match_seed_nodes` returned **all** overlapping nodes sorted by
overlap — no ceiling. Even with Finding 1 fixed, a deliberately broad or
adversarial query ("patient condition medication provider encounter") could
still match many nodes and expand from all of them. Retrieval cost and prompt
size scaled with how vague the question was.

**Fix.** Cap the sorted seed list before expansion, defaulting to
`settings.max_seeds` (5, env `KG_MAX_SEEDS`):

```python
# before
    return [nid for nid, _ in seeds]
# after
    return [nid for nid, _ in seeds[:cap]]
```

Seeds are already sorted by descending token overlap, so the cap keeps the
best-matching starts and drops the long tail.

**Regression test.** `test_seed_set_is_capped` calls with `max_seeds=2` on a
five-type query and asserts `len(seeds) <= 2`.

---

## Finding 3 — Unbounded prompt: no fact budget (Medium)

**Where:** `src/clinicalgraph/core/pipeline.py:48` — `ClinicalGraphRAG.ask`,
budget from `config.py:19` (`max_facts`).

**Root cause.** `retrieve_facts` returns one fact line per node and per edge in
the expanded subgraph; `ask` joined *all* of them straight into the LLM prompt.
A large subgraph (big graph, high `hops`, or many seeds) produced an unbounded
prompt — a latency, cost, and context-window hazard independent of the seed
count.

**Fix.** Apply a fact budget after retrieval, before prompting:

```python
# after (pipeline.ask)
        # Fact budget: cap how many fact lines reach the prompt...
        if settings.max_facts > 0:
            facts = facts[: settings.max_facts]
        answer = self.llm.generate(_PROMPT.format(facts="\n".join(facts), ...))
```

Facts are ordered most-relevant-first (seed descriptions, then their neighbors,
then relations), so truncating the tail drops the least-relevant lines. The
returned `facts` reflect exactly what the prompt saw, keeping the response
honest. `max_facts <= 0` disables the budget.

**Regression test.** `test_fact_budget_caps_prompt_facts` swaps in a frozen
`Settings` copy via `dataclasses.replace` (the config is frozen, so plain
`setattr` is rejected), asserts `len(facts) <= 3` with a budget of 3, and
`> 3` with the budget disabled — proving the cap, not just a small graph.

---

## Finding 4 — Short-token filter drops valid identifiers (Low)

**Where:** `src/clinicalgraph/core/retriever.py:58` — the `len(t) > 2` filter.

**Root cause.** Query tokens of length `<= 2` were discarded, which silently
dropped short identifiers and codes (a 2-char code, an ICD-10 token like `I10`
after tokenization splits, etc.). The threshold was tuned to kill stop-noise but
also killed legitimate short handles.

**Fix.** Lower the threshold from `> 2` to `> 1`, dropping only single-character
noise while keeping 2-char tokens usable:

```python
# before
    q = {t for t in _tokens(query) if len(t) > 2}
# after
    q = {t for t in _tokens(query) if len(t) > 1} - _type_tokens(graph)
```

**Regression test.** `test_short_two_char_token_is_kept_and_matches` seeds from
`"pt 001"` and asserts `pt:001` is found, proving a 2-char token survives the
filter and participates in matching.

---

## Finding 5 — CWD-relative sample path (Low)

**Where:** `src/clinicalgraph/api/main.py:40` — `_SAMPLE`.

**Root cause.** Startup loaded the bundled dataset from a CWD-relative path:

```python
# before
_SAMPLE = Path("data/clinical.json")
```

Launched from anywhere but the repo root, `_SAMPLE.exists()` was false and the
sample graph **silently** failed to load — `/health` then reported zero nodes
with no error. The sibling `_WEB_DIR` was already resolved from `__file__`, so
the two were inconsistent.

**Fix.** Resolve from the package location via a shared `_REPO_ROOT`:

```python
# after
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WEB_DIR = _REPO_ROOT / "web"
...
_SAMPLE = _REPO_ROOT / "data" / "clinical.json"
```

**Regression test.** `test_sample_path_is_cwd_independent` asserts `_SAMPLE`
is absolute and exists.

---

## Verified correct, not changed

These were traced during review and found correct; no change was made, to keep
the diff scoped to real defects.

- **Graph engine hop directions (`core/graph.py`).** `add_edge` indexes both an
  out-adjacency (`_out[src]`) and an in-adjacency (`_in[dst]`); `neighbors`
  returns `(edge, "out", dst)` for outgoing and `(edge, "in", src)` for
  incoming. All four edge types match the loader's schema. No reversed hop.
- **k-hop expansion (`graph.expand`).** BFS over `max(hops, 0)` rounds, dedupes
  edges by index, tracks `seen` to prevent revisits and infinite loops, handles
  `hops=0` (seeds only, no edges) and missing seeds (filtered up front). Correct.
- **Loader edge construction (`core/loader.py`).** Nodes are added before the
  edges that reference them (conditions before medications' `TREATS`; patients
  before encounters' `HAS_ENCOUNTER`); every relation is built in the schema
  direction. No node referenced before it exists.
- **Fact ordering (`retriever.retrieve_facts`).** Seeds first, then reached
  neighbors (so a drug's class / a condition's ICD-10 come along), then edge
  facts, de-duplicated with order preserved — which is also what makes the
  Finding 3 fact budget safe to truncate from the tail.
