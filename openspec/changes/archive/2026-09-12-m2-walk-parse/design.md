## Context

M1 left a database with correct tables and nothing in them. M2 produces the facts that fill it: which files count (`walk.py`) and what each file contains (`parse/`). Nothing is written to SQLite here — M3 owns the writer — so M2's output is in-memory `ParsedNode`/`ParsedRef` values whose fields line up with the M1 `nodes` and `refs` columns.

The ten language parsers and the walker come from `~/dev/indexter` v0.1.2 — the only code carried over. They were written to feed a flat vector store, and a graph exposes three defects, all three of which I reproduced against tree-sitter 0.26 before writing this:

- **Python** `_get_parent_scope` walks up to the nearest `class_definition` and stops. `outer.inner` becomes just `inner`, at file scope.
- **JavaScript/TypeScript** does the same with `class_declaration`. A named callback inside a method (`A.m.cb`) reports scope `A`; an object-literal method (`obj.handler`) reports no scope at all, landing at file scope.
- **Rust** reads only the `type` field of `impl_item`. `impl Display for Foo` and `impl Debug for Foo` both report scope `Foo`, so their two `fmt` methods become one identity.

Two more lifted-code problems: `Query(self.tslanguage, self.query_str)` is constructed inside `parse()`, recompiling the query for every file; and there is no reference extraction at all, so there is nothing for M4 to resolve.

Constraints from the plan: synchronous throughout; nothing lifted except the parsers and walker; the settled node-ID format; the settled kind vocabulary; imports/exports are edges, not nodes; tests co-located, ≥95% coverage.

## Goals / Non-Goals

**Goals:**
- One repository walk yields exactly the files worth indexing, cheaply enough that M3 can walk on every search without reading file contents.
- Every symbol gets a stable identity that survives reformatting and line shifts, and that distinguishes the three collision cases above.
- Every call, import, and inheritance relationship in the four code languages is recorded as a `ParsedRef` with enough information for M4 to resolve it — in particular the head identifier of attribute chains.
- Kinds are a closed vocabulary, not whatever string each parser happened to emit.
- One malformed file cannot abort an index run.
- Per-language snapshot tests over a fixture repo that deliberately contains the collision cases.

**Non-Goals:**
- Writing to the database, composing embed text, embedding, resolving refs into edges (M3/M4).
- The builtin/stdlib stoplist — settled decision 9 drops those at extraction, but the milestone table puts the stoplist in M4. M2 extracts everything and leaves the seam.
- `external_module` nodes: created during resolution in M4, not by parsers.
- New languages beyond the existing ten; LLM summaries; framework or route detection.
- Porting the old repo's ~12,000 lines of parser tests (see Decisions 10).

## Decisions

### 1. The walker yields stat-level candidates; reading and hashing is a separate step

The lifted walker read and hashed every file on every walk. `Walker.walk()` instead yields a `WalkedFile` — relative path, size, mtime, extension — having applied every filter that needs no file contents (ignore patterns, binary extensions, minified names, size limit, empty files). Content and `sha256(relpath:content)` come from an explicit `read_file()` call.

*Why:* settled decision 5 has search sync on every query: mtime+size check, then hash, then re-parse only what changed. If walking read every file, the common case — nothing changed — would still cost a full read of the repository. Splitting the walk lets M3 compare stat data against the `files` table and read only candidates that look different.

*Consequence:* the encoding filter moves to read time, since an undecodable file can only be discovered by reading it. The walk result is therefore a superset of what actually gets indexed, and `read_file()` returns `None` for a file that decodes as neither UTF-8 nor Latin-1. That asymmetry is worth one sentence in the spec rather than a wasted read per file per search.

*Alternative rejected:* keep the eager walker and let M3 ignore the content it doesn't need. Simpler to lift, but it makes the no-op re-index — an explicit M3 success criterion — cost a full repository read.

### 2. `ParsedNode`/`ParsedRef` are frozen dataclasses, not pydantic models

Plain `@dataclass(frozen=True, slots=True)`, with fields named for the M1 columns they land in.

*Why:* these are produced by our own parsers at a rate of thousands per repository and consumed one module away; there is no untrusted input to validate and no serialization boundary. Pydantic's per-instance validation is real cost for no benefit here. Config keeps pydantic (M1) because that *is* untrusted input.

*Consequence:* mistakes in parser output surface as test failures rather than validation errors. The snapshot tests are what catch them.

### 3. Two queries per parser, both compiled in `__init__`

Each code parser holds a compiled definitions `Query` and a compiled references `Query`, built once when the parser instance is created. A fresh `QueryCursor` is created per `parse()` call, since cursors carry match state. Non-code parsers (markdown, json, yaml, toml, html, css) have no references query.

*Why:* fixes the recompile-per-file defect, and keeping the two queries separate keeps match processing honest — a merged query would force every match handler to first work out which pattern it came from. Compilation cost is paid once per parser instance; parser instances are cached per language by the registry.

### 4. Node IDs are assigned file-wide, after parsing, not by individual parsers

Parsers emit nodes carrying `scope_path: tuple[str, ...]`, `name`, and `kind`. A file-level pass in `parse/ids.py` then builds `<relpath>::<scope path>.<name>#<kind>`, sorts same-identity nodes by start line, and appends `~2`, `~3`, … to the second and subsequent ones.

*Why:* `~N` disambiguation is only correct with the whole file in view. Pushing it into each parser would duplicate the logic ten times and make it depend on emission order.

*Consequence:* the ID of a genuine duplicate depends on how many same-named siblings precede it, so inserting a new duplicate above an existing one shifts the lower one's ID. See Risks.

### 5. Scope paths are tuples, and carry the Rust trait qualifier

`scope_path` is `("Handler",)`, `("outer",)`, `("Foo<Display>",)` — joined with `.` only when building the ID. For Rust, an `impl_item` contributes `<type><<trait>>` using the **last `::` segment** of the trait: `impl std::fmt::Display for Foo` → `Foo<Display>`, matching the plan's `Foo<Display>.fmt`. A plain `impl Foo` contributes `Foo`.

*Why tuples:* M2 also computes `parent_id`, and M4 walks ancestors; both want the segments, not a string to re-split.

*Why the last trait segment:* the full path (`Foo<std::fmt::Display>`) makes IDs unwieldy and — worse — makes them churn when someone changes an import style without touching the code. The last segment is what disambiguates in practice.

### 6. Object-literal methods and named callbacks take their enclosing binding as scope

`const obj = { handler(y) {…} }` scopes to `("obj",)`; a named function expression inside a method scopes to the full ancestor path (`("A", "m")` for `cb` inside `A.m`).

*Why:* verified that both currently land wrong — `handler` at file scope, `cb` at `A`. Two `handler` methods in two different object literals in one file are a real pattern in JS config objects, and without the binding name they collide.

### 7. A closed kind vocabulary, normalized at the parser boundary

The lifted parsers emit `"Header 1"`, `"h1"`, `"@media"`, `"mapping"`, `"table"`, `"table_array"`, `"pair"`, `"rule"`. These normalize to the settled set: `file`, `class`, `function`, `method`, `constant`, `interface`, `type_alias`, `enum`, `struct`, `trait`, `section`, `data`, `chunk`. Markdown headings, HTML elements, and CSS rules become `section`; JSON/YAML/TOML blocks become `data`; the fallback parser emits `chunk`. `external_module` is reserved for M4 and never emitted here.

*Why:* the kind is a filter in the M1 `vectors` table (`kind IN (...)` inside the KNN) and part of every node ID. An open-ended vocabulary makes both unusable — no caller can enumerate `"Header 1"` vs `"h1"` vs `"section"`.

*Consequence:* detail is lost (heading level, CSS at-rule type). Where it matters it goes in the node's name or signature, which are searchable, rather than in the kind, which is a filter.

### 8. Imports and exports stop being nodes

Python's `import`/`import_from` and TypeScript's `export_statement` currently produce nodes. They become `ParsedRef`s with `ref_kind="imports"` instead; an export marks the exported symbol rather than creating a second node for it.

*Why:* settled decision 1 — imports/exports are edges. As nodes they were duplicates that competed with the real symbol in ranking, and they gave the graph nothing to traverse.

### 9. Every file yields a `file` node, and `parent_id` is computed here

Each parsed file produces one `file`-kind node spanning the whole file. After IDs are assigned, each node is linked to the innermost enclosing node in the same file by scope path; nodes at file scope get the `file` node as parent.

*Why:* `contains` edges in M4 are then a direct read of `parent_id` rather than a second structural analysis, and the `file` node is what a whole-file match ranks as in M5.

### 10. Snapshot tests over a fixture repo, instead of porting 12,000 lines of assertions

The old repo has ~12,000 lines of parser tests. M2 tests the lifted parsers with `inline-snapshot` over a committed multi-language fixture tree, plus targeted assertions for the three collision cases, reference extraction, and ID stability.

*Why:* the old assertions are written against `NodeMetadata`, a shape that no longer exists, so "porting" means rewriting them anyway. Snapshots capture the whole output of each parser per fixture file, which is both stronger (nothing goes unasserted) and cheaper to maintain when a kind is renamed.

*Consequence:* a snapshot diff is easy to accept without reading. The collision cases, reference extraction, and ID rules get explicit hand-written assertions precisely because those must not be rubber-stamped.

### 11. The fixture repo is committed source plus synthesized edge cases

Parser fixtures live as a real committed tree under `src/indexter/parse/tests/fixtures/` (honouring the co-located-tests convention; `wheel-exclude` already keeps `**/tests/**` out of the wheel). Walker edge cases — symlinks escaping the repo, undecodable bytes, an oversized file, a `.gitignore` — are synthesized into `tmp_path` by a pytest fixture rather than committed.

*Why:* committing a `.gitignore` inside the fixture tree would make git actually ignore the fixture files it names, so they would never be committed at all. Symlinks and binary blobs are similarly awkward in version control. Meanwhile the parser fixtures genuinely want to be readable files a person can open next to the snapshot.

*Consequence:* the fixture tree contains deliberately odd code (nested functions, duplicate names, trait impls) that must not be linted or collected as tests — it needs a ruff `extend-exclude` entry and pytest collection ignore.

### 12. Parse errors are per-file data, not exceptions

`parse_file()` returns a result carrying nodes, refs, and a list of error strings. Tree-sitter is error-tolerant and still yields matches from a file with syntax errors; an unexpected exception inside one parser is caught, recorded, and the file yields whatever it managed.

*Why:* the M1 `files` table has an `errors` column, and a single unparseable file in a large repository must not abort an index run.

## Risks / Trade-offs

- **`~N` IDs shift when a duplicate is inserted above an existing one** → the lower duplicate's ID changes, and edges pointing at it dangle until the referencing file is re-synced. Accepted: genuine duplicates (same name, same kind, same scope, same file) are rare, and M4's failed-ref retry re-resolves dangling references after any later change. The alternative — content-hash suffixes — would churn the ID on every edit to the body, which is far worse.
- **Kind normalization loses detail the old parsers captured** (heading level, at-rule type) → carried in name/signature instead, which are indexed by FTS. If M5's eval shows heading level matters for ranking, it can be promoted without a schema change.
- **HTML/CSS/data-format files can produce many low-value nodes**, inflating embedding cost and cluttering results → they are in the plan's scope and stay for now; M5's eval is the decision point for pruning them, and the kind vocabulary makes them filterable in the meantime.
- **Reference extraction quality is the ceiling on M4** → the measured baseline (1,079 Python call refs, ~80% high-confidence resolvable) came from analysing the old repo; if M2 extracts fewer refs than that analysis assumed, M4 will underperform for reasons that look like resolution bugs. Mitigation: a fixture-based count assertion, and M4 starts by reproducing the baseline percentages.
- **Splitting walk from read (Decision 1) means two syscall passes** for files that do need reading (stat, then open) → negligible next to parsing and embedding, and it saves reading the whole repository on every unchanged search.
- **Lifting code with no lifted tests** → the snapshot suite must be written before the lift is trusted; tasks order it that way (fixtures and models first, then per-language lift-and-snapshot).

## Migration Plan

Nothing to migrate. M2 adds modules and writes nothing to disk; the M1 schema, its version, and existing databases are untouched. The `Settings` additions are new keys with defaults, so existing config files stay valid.

## Open Questions

- **Chunk fallback sizing.** The lifted `ChunkParser` uses 250-character chunks with 25-character overlap — character-based and very small, chosen for a different embedding path. M3's composer is token-budgeted, so these defaults should probably be re-expressed in tokens. M2 keeps the lifted behaviour behind config keys and leaves the retuning to M3.
- **Whether HTML/CSS/data nodes should be embedded at all**, or indexed only in FTS. Settled at M5 by the eval, since it is a retrieval-quality question and costs nothing to defer.
- **Whether TypeScript needs its own scope rules beyond JavaScript's**, or can share the implementation. Resolved during the lift — the two parsers are separate modules in the old repo and stay separate; sharing is an implementation detail, not a contract.
