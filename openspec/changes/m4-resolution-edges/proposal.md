## Why

M3 writes every call, import and inheritance reference into `refs` with status `unresolved`, and `edges` is still empty — the database holds everything vectors need and nothing the graph needs. The graph is the capability the old indexter lacked ("what calls this", "what breaks if I change it"), and M1–M4 carry the risk: if references can't be resolved to the right nodes at a useful rate, M5's expansion and M6's `neighbors` tool have nothing to stand on.

## What Changes

- Add `index/resolve.py`: resolve each ref to a target node through ordered tiers — `self`/`this`/`Self` members (including inherited ones), definitions in enclosing scope, import bindings and module-qualified paths, then name lookup narrowed first to the classes the file defines or imports and then repo-wide. Outcomes are `resolved` (confidence `exact`, `imported` or `unique_name`), `external`, `ambiguous` (2–5 candidates), `too_ambiguous` (more than 5, no edges), or `failed`.
- Per-language module resolution: Python absolute (source-root agnostic), relative and package re-exports; JavaScript/TypeScript relative specifiers with extension, `index` and `.js`→`.ts` resolution; Rust `crate::`/`self::`/`super::` paths over `mod.rs` and file modules.
- External modules become `external::<name>` nodes: in full-text search by name, never embedded, targets of `imports` edges, removed when nothing refers to them any more.
- Add `index/graph.py`: derive `edges` — `contains` from parent links plus `calls`/`imports`/`inherits` from resolved refs, each with line and confidence — and `nodes.degree`, writing only the difference from what is stored.
- Resolution runs inside every sync that changed anything, over the whole repository, after structural writes and before the embedding backlog. A pending marker written with each structural change makes an interrupted resolution heal on the next sync; failed refs are retried every time it runs. A no-op sync still does no resolution work.
- **BREAKING** (parse output): import references are emitted one per bound name, carrying the local binding as their head and the imported member separately; JavaScript/TypeScript default, named, namespace and `require` bindings and `export … from` re-exports are captured; Rust `impl Trait for Type` records the implementing type so its `inherits` edge starts at the type.
- **BREAKING** (parse output): calls and base classes naming a language builtin (`len`, `console`, `Some`, …) are dropped at extraction unless the file defines or imports that name.
- **BREAKING** (schema): `refs` gains `imported_name` and `for_type` columns; the schema version becomes 2, so existing databases are rebuilt by `init`/`reindex` as M3 already provides.
- The sync report and the `init`/`reindex` summary gain resolution outcomes and edge counts.

## Capabilities

### New Capabilities
- `reference-resolution`: How a ref becomes a target — the tier order, per-language module resolution, import bindings and re-exports, candidate narrowing, the ambiguity cap, the tail-name stoplist, external modules, and ref statuses and confidences.
- `graph-edges`: What `edges` holds and how it stays consistent — `contains` derivation, edges from resolved and ambiguous refs, external-module node lifecycle, `nodes.degree`, and the guarantee that after a sync no edge names a missing node.

### Modified Capabilities
- `reference-extraction`: refs carry `imported_name` and `for_type`; import refs are one per binding with the binding as head; head reduction applies to calls and inheritance only; Rust trait impls record the implementing type; builtin calls and bases are dropped at extraction unless shadowed.
- `index-sync`: resolution runs after structural changes (pending marker, resolver version); the embedding backlog skips external-module nodes; the no-op path includes "no resolution pending"; the sync report includes resolution results.
- `repo-management-cli`: the `init`/`reindex` summary reports edges and resolution outcomes.

## Impact

- **New code**: `index/resolve.py`, `index/graph.py`, `parse/builtins.py`, co-located tests, and a multi-language fixture repository with known imports, calls and inheritance (including the collision cases: Rust trait impls, Python nested functions, JS callbacks, same-name symbols in different files).
- **Changed code**: the Python, JavaScript, TypeScript and Rust reference queries and handlers; `parse/models.py` (`ParsedRef` fields); `parse/base.py` (builtin dropping); `db/schema.sql` and `SCHEMA_VERSION`; `index/sync.py` (write the new ref columns, pending marker, run resolution, skip externals in the backlog, report); `index/compose.py` (`INDEX_FORMAT_VERSION` bump); `db/queries.py` (resolution summary); `cli.py` (summary line).
- **Existing databases**: rebuilt on the next `init`/`reindex` because of the schema version bump; nothing to migrate.
- **Dependencies**: none.
- **Not touched**: search, ranking, hub damping and roll-up (M5, which reads `edges`, `confidence` and `degree` written here); the MCP `neighbors` tool (M6).
- **Performance**: a prototype of these tiers over the 7,815 refs in `~/dev/indexter`'s index ran in about 65 ms; resolution adds that order of cost to syncs that changed something and nothing to no-op syncs.
