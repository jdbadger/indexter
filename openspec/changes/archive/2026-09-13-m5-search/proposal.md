## Why

After M4 a synced database holds everything retrieval needs — composed-summary vectors, a full-text index over names, docstrings and bodies, and a resolved graph with confidences and degrees — but nothing reads it. Search is the reason the rewrite exists ("let an agent find code when the user can't name the symbol or module"), and M6's MCP tools are thin wrappers over it, so ranking, expansion and the result shape have to be decided and measured here, including the plan's open question of whether embedding composed summaries actually beats embedding raw code (Q2).

## What Changes

- Add `search/hybrid.py`: `search_repo(conn, repo, query, …)` syncs the repository first (settled decision 5), then ranks nodes by Reciprocal Rank Fusion (k=60) of the top 50 nearest vectors and the top 50 full-text matches, with optional `kind`, `language` and `path` filters applied inside both candidate queries. Nodes in test files are demoted in the fused score. External module nodes are never hits. `search(repo, query, …)` opens the repository's database and fails with an actionable error when it has not been indexed.
- Add `search/expand.py`: one-hop graph expansion from the top 5 hits along `contains`, `calls` (both directions), `inherits` and `imports`, skipping neighbors with degree above 40, weighting `ambiguous` edges at half and never including a node reached only through them. Expanded nodes form a separate `related` list with the reason they were reached, never mixed into the ranking. Also per-hit graph context: up to 3 callers, up to 3 callees, and the containing class.
- Add `search/results.py`: two or more methods of one class (or a class and its own methods) roll up into one class entry; hits are grouped by file and carry node ID, qualified name, kind, `path:start-end`, match reasons, signature, docstring, and a snippet read from disk of at most `snippet_max_lines` lines with the middle elided; a total character budget returns fewer intact hits rather than truncating them; a deterministic plain-text rendering.
- Add a `search_max_chars` setting (default 20,000) alongside the existing `search_limit` and `snippet_max_lines`.
- Add a retrieval eval: a question set of 15–20 natural-language questions about `~/dev/indexter` with expected files, scored hit@5, run by `just eval` (a new `justfile`), outside CI. It compares composed versus raw-code embeddings, vector-only versus keyword-only versus hybrid, test demotion on versus off, and embedding models, and records the Q2 decision.
- No CLI search command (the CLI stays setup-only) and no MCP server yet (M6).

## Capabilities

### New Capabilities
- `hybrid-search`: The search entry point — sync before searching, query validation, vector and keyword candidate retrieval, filters, RRF fusion, test-file demotion, exclusion of external modules, deterministic ordering, and errors for unindexed repositories.
- `graph-expansion`: Related results reached through the graph from top hits — edge kinds and directions, hub damping, ambiguous-edge weighting, scoring, reasons, exclusions — and the per-hit caller, callee and container context.
- `search-results`: The shape and budget of a search response — class roll-up, grouping by file, hit fields, snippets read fresh from disk with middle elision and long-line cutting, the result-count and character budgets, and the rendered text.
- `retrieval-eval`: The question set, hit@5 scoring, the compared variants (composed/raw, vector/keyword/hybrid, test demotion, models), isolation from the user's own configuration and databases, and the `just eval` entry point.

### Modified Capabilities
<!-- None: the new setting doesn't change configuration's requirements, and search reuses sync as specified. -->

## Impact

- **New code**: `search/__init__.py`, `search/hybrid.py`, `search/expand.py`, `search/results.py` with co-located tests; `eval/questions.toml` and `eval/run_eval.py` (not packaged, not in coverage); `justfile`.
- **Changed code**: `config.py` (`search_max_chars`).
- **Schema**: none. The full-text tokenizer stays `unicode61`; Porter stemming was measured and did not help (design).
- **Existing databases**: unchanged and searchable as they are.
- **Dependencies**: none (`rust-just` is already a dev dependency).
- **Not touched**: parsing, composition, resolution, sync, CLI. If the eval decides raw-code embeddings win Q2, changing composition is a follow-up change, not part of this one.
- **Performance**: measured on `~/dev/indexter`'s database — a no-op sync takes ~4 ms warm, filtered KNN over 2,711 vectors under 1 ms, a warm query embedding 5–20 ms — so a warm search should complete well under 100 ms excluding the one-time model load.
