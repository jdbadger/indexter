## 1. Schema and models

- [x] 1.1 Add `imported_name TEXT` and `for_type TEXT` to `refs` in `db/schema.sql`; bump `SCHEMA_VERSION` to 2
- [x] 1.2 Add optional `imported_name` and `for_type` to `ParsedRef` and `RawRef`, carried through `parse.ids.link_refs`
- [x] 1.3 Update the M1 schema round-trip tests for the new columns and the version bump; confirm `index_repository` rebuilds a version-1 database

## 2. Extraction changes (`parse/`)

- [x] 2.1 Python: emit one import ref per binding — plain (`raw=a.b.c`, head first segment), aliased plain (head alias), from-import (`raw` = module as written, `imported_name`, head name or alias), relative markers preserved, wildcard (`imported_name='*'`, no head)
- [x] 2.2 JavaScript and TypeScript: emit one import ref per default, named (with alias), and namespace binding; side-effect imports with no head; `require()` with the assigned variable as head; `export … from` and `export * from` re-exports
- [x] 2.3 Rust: split `use` paths into `raw_name` (prefix) and `imported_name` (last segment) with the binding (last segment or alias) as head, for simple, aliased, grouped, and single-segment uses; wildcard `use a::*` with `imported_name='*'` and no head
- [x] 2.4 Rust: `impl Trait for Type` inherits refs carry `for_type` as written
- [x] 2.5 Add `parse/builtins.py` with frozen builtin-name lists for Python, JavaScript/TypeScript, and Rust; drop call and inherits refs whose head is builtin unless the file defines a node with that name or an import binds it
- [x] 2.6 Bump `INDEX_FORMAT_VERSION` to 2
- [x] 2.7 Tests per language for every binding shape in the reference-extraction delta spec, `for_type`, inherent impls yielding nothing, and builtin dropping with local-definition and import shadowing; imports are never dropped
- [x] 2.8 Regenerate the M2 parse snapshots and the M3 composer snapshots; review the diffs are only the intended ref changes

## 3. Sync writes (`index/sync.py`)

- [x] 3.1 Write `imported_name` and `for_type` when inserting refs
- [x] 3.2 Set `project_metadata.resolution_pending` inside the transactions of `write_file`, `remove_file`, and `record_unreadable` (not `touch_file`)
- [x] 3.3 Exclude `external_module` nodes from the embedding backlog
- [x] 3.4 Tests: each structural write sets the marker in its own transaction (rolled back with it on failure); touch does not; backlog never embeds an external node and does not load the model when only externals lack vectors

## 4. Module resolution (`index/resolve.py`)

- [x] 4.1 Define the in-memory inputs (node and ref records loaded from the database) and the language-family mapping
- [x] 4.2 Python module resolution: relative against the importer's package; absolute by path-component suffix match over `.py` and `__init__.py`, tie-broken by longest shared directory prefix then shortest path; external by first segment
- [x] 4.3 JavaScript/TypeScript specifier resolution: relative with exact, appended-extension, `index`, and `.js`→`.ts` family attempts; bare specifiers external by package name (scoped and unscoped)
- [x] 4.4 Rust module tree: crate root discovery, file module paths (`mod.rs`/`lib.rs`/`main.rs`), `crate`/`self`/`super` and child-module-first paths, external crate by first segment
- [x] 4.5 Member lookup over a resolved target: submodule before top-level node; class members plus inherited members; Rust type members across files by scope path with or without trait suffix; Python and JS/TS re-export following, at most 5 hops, cycle-safe; `default` import rules; `*` wildcard modules
- [x] 4.6 Tests: each scenario in the Python, JavaScript/TypeScript, and Rust module-resolution requirements, plus suffix-match tie-breaking and re-export cycles terminating

## 5. Tiers and outcomes (`index/resolve.py`)

- [x] 5.1 Import bindings visible from an origin (origin or ancestor; innermost wins) and resolution of every `imports` ref to its most specific target or external node
- [x] 5.2 Tier 1: `self`/`cls`/`this`/`Self` → enclosing type (Rust via scope segment without `<Trait>`), member lookup including bases through resolved inherits, cycle-safe; fall through on miss or longer chains
- [x] 5.3 Tier 2: enclosing-scope lookup that skips class-like containers unless they are the origin, with chain member lookup, falling through when a chain continues past a non-container
- [x] 5.4 Tier 3: import-bound heads, wildcard-imported modules, and module-qualified paths (Rust `crate`/`self`/`super`/child-module), walked by remaining segments; external targets yield `external`
- [x] 5.5 `for_type` resolution restricted to type kinds, used as the edge source; unresolvable `for_type` yields `failed`
- [x] 5.6 Tier 4/5: candidate kinds per ref kind; narrowed lookup over classes the file defines or imports (with bases) first, repo-wide second; tail stoplist on repo-wide lookups for attribute calls with unresolved receivers; outcomes `unique_name` / `ambiguous` (2–5) / `too_ambiguous` (>5, first 20 IDs) / `failed`; same language family only
- [x] 5.7 Deterministic ordering of candidates and of all outputs
- [x] 5.8 Tests: every scenario in the reference-resolution delta spec's tier, narrowing, stoplist, and `for_type` requirements, built from small in-memory node/ref sets

## 6. Graph derivation and writes (`index/graph.py`)

- [x] 6.1 Derive the desired edge set: `contains` from `parent_id`; one edge per resolved ref; one per candidate for ambiguous refs; `imports` edges to external nodes; none for `too_ambiguous`/`failed`/external non-imports; dedupe on source, target, kind, line
- [x] 6.2 Define `ResolveReport` (refs by kind × status/confidence, edges by kind × confidence, edges inserted/deleted, external node count, elapsed seconds) and `RESOLVER_VERSION`
- [x] 6.3 Implement `resolve_repo(conn) -> ResolveReport`: load nodes and refs, run resolution, diff against stored refs/edges/external nodes, and in one transaction update changed ref outcomes, insert/delete edges, update changed confidences, insert/delete external nodes with their FTS rows, recompute degree (calls/imports/inherits only) for touched nodes, clear `resolution_pending`, store `resolver_version`
- [x] 6.4 Wire into `sync_repo` between pass one and the backlog, gated on the pending marker or a resolver version mismatch; add `resolution: ResolveReport | None` to `SyncReport`
- [x] 6.5 Add `queries.resolution_summary(conn)` and extend the orphan checks to cover `candidates`
- [x] 6.6 Tests: contains edges mirror parents; edge fan-out and confidences; external node creation, sharing across languages, and removal; degree values and updates; unchanged edges keep their row IDs across an unrelated edit; no dangling edges or targets after deletions; summary equals grouped counts
- [x] 6.7 Tests — sync integration: no-op sync runs no resolution and reports `None`; touch does not resolve; an interrupted resolution (pending marker left set) heals on the next sync; a resolver version bump re-resolves without parsing or embedding; failed refs succeed after a definition is added; unique becomes ambiguous; incremental edits produce the same graph as a from-scratch index

## 7. Fixture repository

- [x] 7.1 Build `src/indexter/index/tests/fixtures/graph_repo/` with the Python, JavaScript/TypeScript, and Rust cases listed in design decision 13
- [x] 7.2 Tests: sync the fixture with `FakeEmbedder` and assert known edges for each tier and language (including Rust `Foo<Display>`/`Foo<Debug>`, Python nested functions, JS callbacks, same-name symbols in different files)
- [x] 7.3 Inline snapshot of every edge (kind, source, target, line, confidence) and every ref outcome over the fixture

## 8. CLI (`cli.py`)

- [x] 8.1 Print a resolution line in the `init`/`reindex` summary when resolution ran: edge counts by kind, call-reference outcomes by status and confidence, external module count, resolution time; omit it otherwise
- [x] 8.2 Tests via `CliRunner`: first init prints the resolution line; a no-change init omits it

## 9. Verification

- [x] 9.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 9.2 `uv run --group dev ty check src/indexter` clean
- [x] 9.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green
- [x] 9.4 Real-repo check: `indexter reindex ~/dev/indexter` rebuilds the version-1 database; record node, edge (by kind), and external counts, and resolution time
- [x] 9.5 Real-repo check: record the resolution summary — share of recorded call refs at `exact`/`imported`/`unique_name` or `external` (target ≥75%, plan baseline ~80%) and `ambiguous` (target ≤15%), plus the same shares over call edges — and break down the largest remaining `failed`/`ambiguous` groups
- [x] 9.6 Real-repo spot checks: callers of `Walker.walk`; `inherits` edges into `BaseLanguageParser`; `imports` edges into `external::pydantic`
- [x] 9.7 Real-repo check: immediate re-`reindex` runs no resolution; an in-process warm sync after a one-function edit (including resolution) completes well under a second — record both times
- [x] 9.8 Real-repo check: orphan queries (including candidates) return nothing; `indexter list` shows the rebuilt database
- [x] 9.9 Confirm the suite passes on Python 3.11, 3.12, and 3.13
